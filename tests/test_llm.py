"""llm.py behaviour with an injected completion function (no network)."""

import pytest
from pydantic import BaseModel

from matgraph import llm


class Out(BaseModel):
    value: int


class RateLimitError(Exception):
    pass


@pytest.fixture(autouse=True)
def configured(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("OPENROUTER_MODEL", "primary")
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODEL", "fallback")
    monkeypatch.setenv("OPENROUTER_MIN_INTERVAL", "0")
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm, "_cooldown_until", {})
    monkeypatch.setattr(llm, "_daily_cap_hit", False)
    yield
    llm.set_completion_fn(None)


def test_falls_back_on_rate_limit_and_cools_down_primary():
    calls = []

    def fake(model, messages, schema, name):
        calls.append(model)
        if model == "primary":
            raise RateLimitError("429 temporarily rate-limited upstream")
        return '{"value": 1}'

    llm.set_completion_fn(fake)
    out, meta = llm.complete_json([{"role": "user", "content": "a"}], Out, use_cache=False)
    assert out.value == 1 and meta["model"] == "fallback"
    llm.complete_json([{"role": "user", "content": "b"}], Out, use_cache=False)
    assert calls == ["primary", "fallback", "fallback"]  # primary skipped while cooling down


def test_schema_retry_then_cache():
    responses = iter(['{"value": "not an int"}', '```json\n{"value": 7}\n```'])
    llm.set_completion_fn(lambda *a: next(responses))
    out, meta = llm.complete_json([{"role": "user", "content": "q"}], Out)
    assert out.value == 7 and not meta["cached"]
    llm.set_completion_fn(lambda *a: pytest.fail("should be served from cache"))
    out, meta = llm.complete_json([{"role": "user", "content": "q"}], Out)
    assert out.value == 7 and meta["cached"]


def test_daily_cap_disables_llm():
    def fake(model, messages, schema, name):
        raise RateLimitError("Rate limit exceeded: free-models-per-day")

    llm.set_completion_fn(fake)
    with pytest.raises(llm.LLMUnavailable, match="daily"):
        llm.complete_json([{"role": "user", "content": "x"}], Out, use_cache=False)
    llm.set_completion_fn(lambda *a: pytest.fail("no further requests after the daily cap"))
    with pytest.raises(llm.LLMUnavailable):
        llm.complete_json([{"role": "user", "content": "y"}], Out, use_cache=False)


def test_empty_response_moves_to_next_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODEL", "fallback, third")
    calls = []

    def fake(model, messages, schema, name):
        calls.append(model)
        if model != "third":
            raise llm.EmptyResponse(f"{model} returned empty content")
        return '{"value": 3}'

    llm.set_completion_fn(fake)
    out, meta = llm.complete_json([{"role": "user", "content": "z"}], Out, use_cache=False)
    assert out.value == 3 and meta["model"] == "third"
    assert calls == ["primary", "fallback", "third"]
