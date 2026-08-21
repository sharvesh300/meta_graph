"""Materials Project REST client: chemistry-driven cluster queries, deduped
and cached to disk as one JSON record per material_id."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

MP_BASE_URL = "https://api.materialsproject.org"

SUMMARY_FIELDS = [
    "material_id",
    "formula_pretty",
    "chemsys",
    "elements",
    "nsites",
    "volume",
    "density",
    "band_gap",
    "is_gap_direct",
    "is_metal",
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_stable",
    "total_magnetization",
    "ordering",
    "symmetry",
    "theoretical",
    # Elastic properties (PRD: "Elastic Properties"). Not computed for every
    # material - null when unavailable. Measured coverage across the full
    # 794-material set: 18.0% (143/794).
    "bulk_modulus",
    "shear_modulus",
    "universal_anisotropy",
    "homogeneous_poisson",
    # Dielectric properties (PRD: "Dielectric Properties"). Measured coverage:
    # 19.9% (158/794).
    "e_total",
    "e_electronic",
    "e_ionic",
    "n",
    # Electronic structure scalars (cheap, always paired with band_gap).
    "cbm",
    "vbm",
    "efermi",
    # Magnetic detail beyond total_magnetization/ordering.
    "is_magnetic",
    "num_magnetic_sites",
    # Oxidation states (PRD: "Oxidation States") -> OxidationState nodes.
    "possible_species",
    # Anonymized stoichiometric pattern (e.g. "ABC3") -> FormulaPattern nodes.
    "formula_anonymous",
    # Full crystal structure: cached to disk for Phase 3's CGCNN, deliberately
    # NOT added as a Cognee graph node property (see enrich.py) - it's a bulky
    # nested lattice/coordinates blob, not something an LLM can reason over.
    # The robocrys description already captures the same structural facts in
    # embeddable text form.
    "structure",
]


class MPClientError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get("MP_API_KEY")
    if not key:
        raise MPClientError("MP_API_KEY is not set (check your .env file)")
    return key


def _get(path: str, params: dict[str, Any], retries: int = 3) -> list[dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                f"{MP_BASE_URL}{path}",
                params=params,
                headers={"X-API-KEY": _api_key()},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["data"]
        except (requests.exceptions.RequestException,) as exc:
            last_error = exc
            if attempt < retries:
                wait = 2**attempt
                logger.warning("request failed (attempt %d/%d): %s; retrying in %ds", attempt, retries, exc, wait)
                time.sleep(wait)
    raise MPClientError(f"request to {path} failed after {retries} attempts") from last_error


# --- cluster (bulk) queries --------------------------------------------------


def _query_summary(
    params: dict[str, Any], exclude_elements: list[str], limit: int
) -> list[dict[str, Any]]:
    request_params = {
        **params,
        "_fields": ",".join(SUMMARY_FIELDS),
        "_sort_fields": "energy_above_hull",
        "_limit": limit,
    }
    if exclude_elements:
        # The API caps exclude_elements at 60 chars as a comma-joined string,
        # but accepts it fine as repeated query params - `requests` does this
        # automatically for a list-valued param.
        request_params["exclude_elements"] = exclude_elements
    return _get("/materials/summary/", request_params)


def _cluster_queries(cluster: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a cluster spec into concrete MP query param dicts.

    See config/materials.yaml's `clusters` section for the pattern docs.
    """
    pattern = cluster["pattern"]

    if pattern == "elements_plus_fixed":
        fixed = cluster["fixed"]
        queries = []
        for element in cluster["vary"]:
            queries.append(
                {
                    "elements": ",".join([*fixed, element]),
                    "nelements_max": cluster["nelements_max"],
                    "energy_above_hull_max": cluster["energy_above_hull_max"],
                }
            )
        return queries

    if pattern == "pairwise_chemsys":
        chemsys = [f"{a}-{b}" for a in cluster["group_a"] for b in cluster["group_b"]]
        return [
            {"chemsys": cs, "energy_above_hull_max": cluster["energy_above_hull_max"]}
            for cs in chemsys
        ]

    if pattern == "pairwise_chemsys_with_suffix":
        suffix = cluster["suffix"]
        chemsys = [
            f"{a}-{b}-{suffix}"
            for a in cluster["group_a"]
            for b in cluster["group_b"]
            if a != b
        ]
        return [
            {"chemsys": cs, "energy_above_hull_max": cluster["energy_above_hull_max"]}
            for cs in chemsys
        ]

    if pattern == "chemsys_list":
        return [
            {"chemsys": cs, "energy_above_hull_max": cluster["energy_above_hull_max"]}
            for cs in cluster["chemsys"]
        ]

    raise ValueError(f"unknown cluster pattern: {pattern!r}")


def fetch_cluster(cluster: dict[str, Any], exclude_elements: list[str]) -> list[dict[str, Any]]:
    """Fetch every sub-query for one cluster, tagging each doc with its cluster name."""
    limit = cluster["limit_per_query"]
    results: list[dict[str, Any]] = []
    for params in _cluster_queries(cluster):
        try:
            docs = _query_summary(params, exclude_elements, limit)
        except MPClientError:
            logger.warning("skipping sub-query %s (%s) after repeated failures", params, cluster["name"])
            continue
        for doc in docs:
            doc["_cluster"] = cluster["name"]
        results.extend(docs)
    return results


def dedupe_by_formula(materials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse to one entry per formula_pretty: the lowest-energy_above_hull polymorph."""
    best: dict[str, dict[str, Any]] = {}
    for m in materials:
        formula = m["formula_pretty"]
        if formula not in best or m["energy_above_hull"] < best[formula]["energy_above_hull"]:
            best[formula] = m
    return list(best.values())


def fetch_robocrys_batch(material_ids: list[str], batch_size: int = 100) -> dict[str, str]:
    """Fetch robocrys descriptions for many materials at once (batched, confirmed
    supported by the API: material_ids accepts a comma-separated list)."""
    descriptions: dict[str, str] = {}
    for i in range(0, len(material_ids), batch_size):
        batch = material_ids[i : i + batch_size]
        docs = _get(
            "/materials/robocrys/",
            {
                "material_ids": ",".join(batch),
                "_fields": "material_id,description",
                "_limit": len(batch),
            },
        )
        for doc in docs:
            descriptions[doc["material_id"]] = doc.get("description")
        logger.info("fetched robocrys descriptions %d/%d", min(i + batch_size, len(material_ids)), len(material_ids))
    return descriptions


def fetch_all_clusters(
    clusters: list[dict[str, Any]], exclude_elements: list[str]
) -> list[dict[str, Any]]:
    """Fetch every cluster, dedupe by formula across all of them, attach robocrys text."""
    all_docs: list[dict[str, Any]] = []
    for cluster in clusters:
        docs = fetch_cluster(cluster, exclude_elements)
        logger.info("cluster %-28s -> %d raw docs", cluster["name"], len(docs))
        all_docs.extend(docs)

    materials = dedupe_by_formula(all_docs)
    logger.info("deduped %d raw docs -> %d unique materials", len(all_docs), len(materials))

    material_ids = [m["material_id"] for m in materials]
    descriptions = fetch_robocrys_batch(material_ids)
    for m in materials:
        m["description"] = descriptions.get(m["material_id"])

    return materials


# --- disk cache --------------------------------------------------------------

MANIFEST_NAME = "_manifest.json"


def cache_path(data_dir: Path, material_id: str) -> Path:
    return data_dir / f"{material_id}.json"


def fetch_and_cache_bulk(
    clusters: list[dict[str, Any]],
    exclude_elements: list[str],
    data_dir: Path,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """Fetch every cluster and cache each material as data/raw/<material_id>.json,
    plus a manifest listing all cached ids. Reuses the cache unless refresh=True."""
    manifest_path = data_dir / MANIFEST_NAME
    if manifest_path.exists() and not refresh:
        material_ids = json.loads(manifest_path.read_text())
        logger.info("cache hit: %d materials from %s", len(material_ids), manifest_path)
        return [json.loads(cache_path(data_dir, mid).read_text()) for mid in material_ids]

    materials = fetch_all_clusters(clusters, exclude_elements)

    data_dir.mkdir(parents=True, exist_ok=True)
    material_ids = []
    for m in materials:
        material_ids.append(m["material_id"])
        cache_path(data_dir, m["material_id"]).write_text(json.dumps(m, indent=2))
    manifest_path.write_text(json.dumps(sorted(material_ids), indent=2))

    return materials
