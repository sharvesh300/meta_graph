"""In-memory view of the built knowledge graph.

The whole graph (~1.5k nodes, ~15k edges) is pulled out of cognee's graph
engine once with ``get_graph_data()`` and indexed in Python. Retrieval then
traverses plain dicts instead of issuing per-hop queries, and depends only on
``get_graph_engine``/``get_graph_data`` - so the Kuzu -> Neo4j swap stays a
config change (cognee's Kuzu backend stores every node in one generic table
with a JSON properties blob, so backend-specific Cypher would not port anyway).

Materials added through ``matgraph.ingest`` in preview mode live in an
overlay on top of this view (``add_overlay_node`` / ``add_overlay_edge``)
without being written to cognee.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from cognee import config
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.users.methods import get_default_user

from matgraph.pipeline import COGNEE_SYSTEM_DIR, DATASET_NAME

# Relationship types whose target is a Material (material-material edges).
MATERIAL_TO_MATERIAL = {"similar_to"}


def _is_struct(props: dict[str, Any]) -> bool:
    return str(props.get("material_id", "")).startswith("struct-")


class Graph:
    """Thin in-memory index over cognee's (nodes, edges) tuples."""

    def __init__(self, nodes, edges):
        self.node: dict[str, dict[str, Any]] = {}
        self.out: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
        self.in_: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
        self.by_type: dict[str, list[str]] = defaultdict(list)
        self.material_by_formula: dict[str, str] = {}
        self.material_by_id: dict[str, str] = {}
        self.overlay_nodes: set[str] = set()
        for node_id, props in nodes:
            self._index_node(str(node_id), props)
        for source_id, target_id, rel, props in edges:
            self._index_edge(str(source_id), str(target_id), rel, props or {})

    def _index_node(self, node_id: str, props: dict[str, Any]) -> None:
        self.node[node_id] = props
        type_name = props.get("type", "?")
        self.by_type[type_name].append(node_id)
        if type_name == "Material":
            # Phase 1 dedupes by formula, but a user-supplied structure can be
            # another polymorph of a formula already present: a formula then
            # resolves to the Materials Project node, never the struct- one.
            current = self.material_by_formula.get(props["formula"])
            if current is None or (_is_struct(self.node[current]) and not _is_struct(props)):
                self.material_by_formula[props["formula"]] = node_id
            self.material_by_id[props["material_id"]] = node_id

    def _index_edge(self, source_id: str, target_id: str, rel: str, props: dict) -> None:
        self.out[source_id].append((rel, target_id, props))
        self.in_[target_id].append((rel, source_id, props))

    # --- overlay (preview-mode ingestion) -----------------------------------

    def add_overlay_node(self, node_id: str, props: dict[str, Any]) -> None:
        self._index_node(node_id, props)
        self.overlay_nodes.add(node_id)

    def add_overlay_edge(self, source_id: str, target_id: str, rel: str, props: dict) -> None:
        self._index_edge(source_id, target_id, rel, props)

    # --- lookups -------------------------------------------------------------

    def nodes_of_type(self, type_name: str) -> list[tuple[str, dict[str, Any]]]:
        return [(nid, self.node[nid]) for nid in self.by_type.get(type_name, [])]

    def targets(self, node_id: str, rel: str) -> list[tuple[str, dict[str, Any]]]:
        """(target_id, target_node_props) pairs for outgoing edges of type `rel`."""
        return [(tid, self.node[tid]) for r, tid, _edge_props in self.out[node_id] if r == rel]

    def target_edges(self, node_id: str, rel: str) -> list[tuple[str, dict]]:
        """(target_id, edge_props) pairs, when the edge's own properties matter."""
        return [(tid, ep) for r, tid, ep in self.out[node_id] if r == rel]

    def sources(self, node_id: str, rel: str) -> list[str]:
        """Source ids of incoming edges of type `rel` (e.g. materials -> an Element)."""
        return [sid for r, sid, _ep in self.in_[node_id] if r == rel]

    def find(self, type_name: str, **match: Any) -> str | None:
        """Id of the first node of `type_name` whose props equal every `match` item."""
        for nid in self.by_type.get(type_name, []):
            props = self.node[nid]
            if all(props.get(k) == v for k, v in match.items()):
                return nid
        return None

    def material(self, key: str) -> str | None:
        """Resolve a Material node id from a material_id or an exact formula."""
        return self.material_by_id.get(key) or self.material_by_formula.get(key)


class KnowledgeBase:
    """The in-memory Graph plus access to cognee's vector store for the same
    dataset. Vector ids are DataPoint ids, i.e. the graph's node ids."""

    def __init__(self, graph: Graph, dataset_id, user_id):
        self.graph = graph
        self._dataset_id = dataset_id
        self._user_id = user_id

    async def semantic_scores(
        self, text: str, collections: tuple[str, ...] = ("Material_profile", "Material_description")
    ) -> dict[str, float]:
        """Cosine similarity (1 - cosine distance) of `text` to every node in
        each collection; per node, the best score across collections."""
        if not text.strip() or self._dataset_id is None:
            # No vector store attached (in-memory graphs in tests / evaluation).
            return {}
        async with set_database_global_context_variables(self._dataset_id, self._user_id):
            engine = get_vector_engine()
            vector = (await engine.embedding_engine.embed_text([text]))[0]
            scores: dict[str, float] = {}
            for collection in collections:
                if not await engine.has_collection(collection):
                    continue
                for hit in await engine.search(collection, query_vector=vector, limit=None):
                    node_id = str(hit.id)
                    scores[node_id] = max(scores.get(node_id, 0.0), 1.0 - hit.score)
        return scores


async def _dataset_context():
    config.system_root_directory(str(COGNEE_SYSTEM_DIR))
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets([DATASET_NAME], "read", user)
    if not datasets:
        raise RuntimeError(f"dataset {DATASET_NAME!r} not found (run scripts/build_graph.py first)")
    return datasets[0].id, user.id


async def load_graph() -> Graph:
    """Read the built dataset out of cognee's graph engine into a Graph."""
    dataset_id, user_id = await _dataset_context()
    async with set_database_global_context_variables(dataset_id, user_id):
        engine = await get_graph_engine()
        nodes, edges = await engine.get_graph_data()
    return Graph(nodes, edges)


async def load_knowledge_base() -> KnowledgeBase:
    dataset_id, user_id = await _dataset_context()
    async with set_database_global_context_variables(dataset_id, user_id):
        engine = await get_graph_engine()
        nodes, edges = await engine.get_graph_data()
    return KnowledgeBase(Graph(nodes, edges), dataset_id, user_id)


def graph_from_datapoints(materials: list[Any]) -> Graph:
    """Build a Graph straight from DataPoints (no cognee stores), mirroring
    what add_data_points writes. Used by the ingestion evaluation and tests."""
    from matgraph.ingest import _edge_targets, _node_props

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[tuple[str, str, str, dict]] = []
    seen_edges: set[tuple[str, str, str]] = set()

    def visit(dp: Any) -> str:
        nid = str(dp.id)
        if nid in nodes:
            return nid
        nodes[nid] = _node_props(dp)
        if type(dp).__name__ == "SpaceGroup":
            cs = visit(dp.crystal_system)
            edges.append((nid, cs, "crystal_system", {}))
        for rel, target, props in _edge_targets(dp):
            tid = visit(target)
            if (nid, tid, rel) not in seen_edges:
                seen_edges.add((nid, tid, rel))
                edges.append((nid, tid, rel, props))
        return nid

    for m in materials:
        visit(m)
    return Graph(list(nodes.items()), edges)
