"""Workarounds for bugs in the pinned cognee version (1.3.0).

index_data_points shares one metadata dict across per-field copies
---------------------------------------------------------------------
For a DataPoint with several ``index_fields`` (Material: formula,
description, profile), cognee's ``index_data_points`` makes one
``model_copy()`` per field and sets ``copy.metadata["index_fields"] =
[field]``. ``model_copy`` is shallow, so every copy shares the original's
metadata dict: after the loop all copies say ``[<last field>]`` and every
per-field collection (Material_formula, Material_description,
Material_profile) is embedded from the *same* text. Found in Phase 2 when
all three collections returned identical scores; it silently affected the
Phase 1 build too.

The fix below expands multi-field DataPoints into single-field copies that
each own their metadata dict before handing them to cognee's function, which
then only ever sees one field per point. Remove once cognee fixes it upstream
(check: the Material_* collections should return different top hits for the
same query - see tests in scripts/verify via build_graph.py).
"""

from __future__ import annotations

_applied = False


def apply() -> None:
    global _applied
    if _applied:
        return
    import importlib

    # Not `import cognee.tasks.storage.add_data_points as ...`: the package
    # re-exports a function of the same name, which shadows the module.
    add_data_points_module = importlib.import_module("cognee.tasks.storage.add_data_points")

    original = add_data_points_module.index_data_points

    async def index_data_points_fixed(data_points, vector_engine=None):
        expanded = []
        for data_point in data_points:
            metadata = getattr(data_point, "metadata", None)
            fields = (metadata or {}).get("index_fields") or []
            if len(fields) <= 1:
                expanded.append(data_point)
                continue
            for field in fields:
                single = data_point.model_copy()
                object.__setattr__(single, "metadata", {**metadata, "index_fields": [field]})
                expanded.append(single)
        await original(expanded, vector_engine=vector_engine)
        return data_points

    add_data_points_module.index_data_points = index_data_points_fixed
    _applied = True
