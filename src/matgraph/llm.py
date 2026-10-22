"""LLM access for Phase 2: OpenRouter free models through LiteLLM.

Every call asks for JSON matching a pydantic model and returns a validated
instance. Designed around the free tier's constraints:

- **Fallback chain.** OPENROUTER_MODEL first, then OPENROUTER_FALLBACK_MODEL on
  rate limits (the free Qwen pool is frequently rate-limited upstream),
  outages, or repeated schema failures.
- **Disk cache** keyed by (model chain, messages, schema): reruns - notably the
  evaluation - cost no requests and reproduce exactly.
- **Client-side throttle** between live requests.
- **Schema-error retry**: an invalid response is sent back with the pydantic
  error so the model can correct it.

Callers catch ``LLMUnavailable`` and fall back to their deterministic
(``--no-llm``) path. Tests never reach the network: they use that path or
inject a fake via ``set_completion_fn``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = ROOT / ".llm_cache"

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    """No configured model produced a valid response."""


def _models() -> list[str]:
    """Primary model, then OPENROUTER_FALLBACK_MODEL (comma-separated list allowed)."""
    fallbacks = (os.getenv("OPENROUTER_FALLBACK_MODEL") or "").split(",")
    models = [os.getenv("OPENROUTER_MODEL"), *fallbacks]
    return [m.strip() for m in models if m and m.strip()]


class EmptyResponse(RuntimeError):
    """The model returned no content - e.g. a reasoning model that spent its
    whole output budget on hidden reasoning tokens. Try the next model."""


def llm_configured() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY") and _models())


def _litellm_completion(model: str, messages: list[dict], schema: dict, name: str) -> str:
    import litellm

    litellm.suppress_debug_info = True
    response = litellm.completion(
        model=model,
        api_key=os.environ["OPENROUTER_API_KEY"],
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": name, "strict": False, "schema": schema},
        },
        temperature=0,
        timeout=float(os.getenv("OPENROUTER_TIMEOUT", "120")),
        # Reasoning models (e.g. Nemotron) can burn the whole output budget on
        # hidden reasoning and return empty content: give them room and ask
        # OpenRouter for low reasoning effort (ignored by non-reasoning models).
        max_tokens=int(os.getenv("OPENROUTER_MAX_TOKENS", "6000")),
        extra_body={"reasoning": {"effort": "low", "exclude": True}},
    )
    content = response.choices[0].message.content or ""
    if not content.strip():
        raise EmptyResponse(f"{model} returned empty content (finish_reason={response.choices[0].finish_reason})")
    return content


# Swappable for tests: (model, messages, json_schema, schema_name) -> raw text.
_completion_fn: Callable[[str, list[dict], dict, str], str] = _litellm_completion


def set_completion_fn(fn: Callable[[str, list[dict], dict, str], str] | None) -> None:
    global _completion_fn
    _completion_fn = fn or _litellm_completion


_throttle_lock = threading.Lock()
_last_call = 0.0

# Free-tier budget protection: a model that just returned a rate-limit error is
# skipped until its cooldown ends instead of burning another request (failed
# requests can count against OpenRouter's daily free-model cap). Hitting the
# daily cap itself disables the LLM for the rest of the process.
_cooldown_until: dict[str, float] = {}
_daily_cap_hit = False
RATE_LIMIT_COOLDOWN_S = float(os.getenv("OPENROUTER_COOLDOWN", "300"))


def _throttle() -> None:
    global _last_call
    min_interval = float(os.getenv("OPENROUTER_MIN_INTERVAL", "3"))
    with _throttle_lock:
        wait = _last_call + min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def _extract_json(text: str) -> str:
    """Strip markdown fences / leading prose some models add around JSON."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def _cache_key(messages: list[dict], schema: dict) -> str:
    # Keyed on the request, not the model chain: changing fallbacks shouldn't
    # re-spend free-tier quota. The entry records which model answered.
    blob = json.dumps({"messages": messages, "schema": schema}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _is_retryable_elsewhere(error: Exception) -> bool:
    """Errors worth moving to the next model for (rate limit, outage, timeout)."""
    name = type(error).__name__
    return any(
        k in name
        for k in ("RateLimit", "ServiceUnavailable", "Timeout", "APIConnection", "InternalServer",
                  "BadGateway", "NotFound", "APIError", "EmptyResponse")
    )


def complete_json(
    messages: list[dict[str, str]],
    output_model: type[T],
    *,
    use_cache: bool = True,
    max_schema_retries: int = 1,
) -> tuple[T, dict[str, Any]]:
    """Return (validated output, meta) where meta = {"model", "cached"}.

    Raises LLMUnavailable if no model in the chain yields a valid response.
    """
    if not llm_configured():
        raise LLMUnavailable("OPENROUTER_API_KEY / OPENROUTER_MODEL not configured")

    models = _models()
    schema = output_model.model_json_schema()
    schema_name = output_model.__name__
    key = _cache_key(messages, schema)
    cache_path = CACHE_DIR / f"{key}.json"

    if use_cache and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        try:
            return output_model.model_validate_json(cached["content"]), {
                "model": cached["model"],
                "cached": True,
            }
        except ValidationError:
            logger.warning("discarding cached LLM response that no longer validates")

    global _daily_cap_hit
    if _daily_cap_hit:
        raise LLMUnavailable("OpenRouter daily free-model request cap reached; try again tomorrow")

    errors: list[str] = []
    for model in models:
        if _cooldown_until.get(model, 0) > time.monotonic():
            errors.append(f"{model}: skipped (rate-limited recently)")
            continue
        attempt_messages = list(messages)
        for attempt in range(max_schema_retries + 1):
            _throttle()
            try:
                raw = _completion_fn(model, attempt_messages, schema, schema_name)
            except Exception as error:  # noqa: BLE001 - provider errors vary widely
                errors.append(f"{model}: {type(error).__name__}: {str(error)[:200]}")
                if "per-day" in str(error) or "per day" in str(error).lower():
                    _daily_cap_hit = True
                    raise LLMUnavailable("OpenRouter daily free-model request cap reached; try again tomorrow") from error
                if "RateLimit" in type(error).__name__:
                    _cooldown_until[model] = time.monotonic() + RATE_LIMIT_COOLDOWN_S
                if _is_retryable_elsewhere(error):
                    logger.warning("LLM %s unavailable (%s); trying next model", model, type(error).__name__)
                    break
                raise LLMUnavailable("; ".join(errors)) from error

            content = _extract_json(raw)
            try:
                result = output_model.model_validate_json(content)
            except ValidationError as error:
                errors.append(f"{model}: invalid JSON for {schema_name} (attempt {attempt + 1})")
                attempt_messages = messages + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": "That response did not match the required JSON schema:\n"
                        f"{error}\nReturn only corrected JSON.",
                    },
                ]
                continue

            if use_cache:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps({"model": model, "content": content}))
            return result, {"model": model, "cached": False}

    raise LLMUnavailable("; ".join(errors) or "no model produced a response")
