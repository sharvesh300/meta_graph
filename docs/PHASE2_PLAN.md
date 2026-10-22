# MatGraphRAG — Phase 2 Plan: GraphRAG Retrieval and Recommendation Pipeline

> **Status (2026-09-29): M0–M6 implemented. M7 (Neo4j, optional) not done.**
> Results are in README §10–16 and `artifacts/eval/`. Acceptance criteria:
>
> - **M0**
>   - ✓ Counts are unchanged and the collections are verified distinct.
>   - ~ "layered Li cobalt oxide": the top 5 are all Li–Co–O compounds, but
>     LiCoO2 itself isn't among them.
> - **M1:** see the LLM section of the eval. The sample was limited by
>   free-tier quota.
> - **M2**
>   - ✓ 0 hard-constraint violations across the benchmark.
>   - ✓ Every result carries a reasoning path.
>   - ✓ Graph retrieval takes ~5 ms. ~ Embedding the question adds ~1–2 s
>     on local Ollama, so end-to-end p95 is above the 1 s target.
> - **M3:** ✓ Templated explanations are 100% traceable. LLM traceability is
>   in the eval.
> - **M4:** ✓ All criteria met; ingestion leave-one-out: 100% categorical,
>   Jaccard 0.98.
> - **M5:** ✓ Done. Tier 4 (similarity) is the weakest result, reported as
>   is.
> - **M6:** ✓ API and dashboard done; 44 tests pass.
>
> §14 lists the findings made during implementation.

Goal: take a natural-language materials question ("a stable, cobalt-free Li
cathode with a band gap above 2 eV"), retrieve candidates by **traversing the
Phase 1 knowledge graph** (not just semantic text similarity), and return an
**LLM-written recommendation in which every claim cites a specific graph edge
or Materials Project record**. We then measure whether graph retrieval
actually beats plain vector RAG.

Source of truth: the PRD (`MatGraphRAG_Updated.pages`), Phase 2 pipeline:

```
Embedding generation (BGE-M3) → GraphRAG retrieval & graph traversal
  → LLM (Qwen / Llama) → Explainable material recommendation
  → Results & visualization dashboard
```

plus the PRD's `/query`, `/material/{id}`, `/health` endpoints and its
evaluation design (Baseline 1 = Vector RAG over text, Baseline 2 = GraphRAG with
fixed-hop traversal; Recall@10, MRR, nDCG, and the share of answer statements
traceable to a graph edge or record). Phase 2 also adds **ingestion of new
materials** (§6): given a new material as an MP ID or a formula, it fetches
the material's properties and attaches it to the existing graph. Crystal
structure (CIF) input goes to CGCNN, so it waits for Phase 3 along with CGCNN
itself, element substitution, and `/predict` /
`/similar-structures` / `/similar-elements` / `/analyse` belong to
**Phase 3**, not here.

---

## 1. Starting point (Phase 1, as built)

- A Cognee graph with 1,504 nodes and 14,681 edges: 794 `Material`, 46 `Element`,
  106 `SpaceGroup`, 7 `CrystalSystem`, 143 `ChemicalSystem`, 11 `PropertyClass`,
  5 `ApplicationDomain`, 131 `OxidationState`, 261 `FormulaPattern`.
- Numeric properties live as `Material` attributes. Elastic coverage is 18% and
  dielectric coverage is 20%, so **missing data is common**.
- `suitable_for` edges carry the exact rule text that fired, which makes them
  ready-made evidence.
- LanceDB has these vector collections (checked against the current store):
  `Material_description`, `Material_formula`, `PropertyClass_description`,
  `ApplicationDomain_description`, `Element_name`, `ChemicalSystem_chemsys`,
  `SpaceGroup_symbol`, `CrystalSystem_name`, `OxidationState_species`,
  `FormulaPattern_pattern`, `EdgeType_relationship_name`. They are embedded with
  `bge-small-en-v1.5` (384-d). `vector_engine.search()` works and returns node
  UUIDs plus a **distance** (lower is closer), which we join back to graph nodes.
- `get_graph_data()` loads the whole graph (about 1.5k nodes and 15k edges).
  That fits in memory easily, and `verify.py` already has a small in-memory
  `Graph` index.
- There is no LLM configured. `LLM_API_KEY` is unset and
  `COGNEE_SKIP_CONNECTION_TEST=true`. Ollama is installed on the machine (Apple
  M4, 16 GB) but has no models pulled yet. It is used only for local
  embeddings (§2.4). The LLM is an OpenRouter free model (§2.3).
- `tests/` is empty.

## 2. Key design decisions

### 2.1 Custom retriever instead of Cognee's `GRAPH_COMPLETION`
Cognee's built-in `GRAPH_COMPLETION` search does this: vector-match the query
to nodes and edges, collect nearby triplets, and hand them to an LLM. That is
generic text-first GraphRAG. Materials questions are mostly **hard numeric
constraints** such as `band_gap ∈ [2, 3.5]`, `E_hull ≤ 0.05`, or "no Co".
Semantic similarity cannot enforce those, and an LLM reading triplets cannot
do so reliably either. So:

- Constraints are **parsed into a typed schema** and then **enforced
  deterministically** against node attributes and edges.
- The vector store is used for what it is good at: fuzzy entry points such as
  "layered", "octahedral", "similar to olivine", or a misspelled formula.
- The LLM only (a) parses the question and (b) writes the explanation from an
  evidence bundle. **It never selects or filters materials.**

We still run Cognee's `GRAPH_COMPLETION` once, as an extra baseline in the
evaluation (§7), because it is the off-the-shelf GraphRAG we are claiming to
improve on.

### 2.2 Evidence-first generation
Retrieval produces an **evidence bundle**. It is a list of atomic facts, and
each fact has a stable ID:

```
[E12] Material mp-xxxx (LiMnPO4) band_gap = 3.71 eV          (source: MP summary field)
[E13] LiMnPO4 —suitable_for→ battery_cathode                 (rule: "contains Li and a transition metal, …")
[E14] LiMnPO4 —similar_to→ LiFePO4  weight 0.91              (10-d property cosine)
[E15] LiMnPO4 bulk_modulus_vrh: NOT AVAILABLE in Materials Project
```

The LLM has to cite evidence IDs for every claim. A **deterministic grounding
checker** runs after generation. It checks that every cited ID exists, that
every number in a claim matches a value in the evidence it cites (within
tolerance), and that no recommended material is missing from the retrieved
set. Claims that fail are flagged or dropped. The share of claims that pass
is the PRD's traceability metric. Missing properties are listed explicitly,
so the model has no gap to fill with an invented value.

### 2.3 LLM: an OpenRouter free model, behind a provider-agnostic wrapper
**Decision: OpenRouter free tier (API key supplied by you).** OpenRouter is
OpenAI-compatible (`https://openrouter.ai/api/v1`). All calls go through a
single `llm.py` wrapper built on LiteLLM, which Cognee already depends on. The
model ID lives in `.env`, so switching models or providers is a config change.

- **Primary:** `qwen/qwen3.8-27b:free`. It is Qwen, which matches the PRD, has
  a 262k context, and supports `structured_outputs`, so the parser and the
  generator can use JSON-schema-constrained output.
- **Fallback:** `nvidia/nemotron-3-super-120b-a12b:free`, which supports both
  `structured_outputs` and `response_format`. The wrapper switches to it on
  429/5xx errors or when the primary model is unavailable.
- Both IDs came from OpenRouter's live `/models` listing on 2026-09-29. The
  free roster changes often, so M0 re-checks them and pins them.

**Free-tier constraints the design has to absorb:**
- Per-minute and per-day request caps, which are lower without account credits.
- Models that disappear from the free list.
- Free endpoints that may log prompts. That is acceptable here because every
  prompt contains only public Materials Project data and the user's question.
  No secrets are ever put in a prompt.

**Mitigations:**
- A disk-backed **response cache** keyed by `(model, prompt hash, schema)`.
  It makes re-runs free and deterministic, and it is essential for the
  evaluation budget (§7).
- Client-side rate limiting, plus retry with backoff on 429.
- The retry prompt includes the validation error when the JSON fails its
  schema.
- A `--no-llm` mode that uses the rule-based parser and a templated
  explanation. Tests and CI never call OpenRouter, the system still works when
  the quota runs out, and the mode doubles as a clean ablation.

The **retrieval** evaluation needs no LLM at all, because it runs from gold
constraints (§7). The LLM budget is spent only on parser accuracy and
explanation traceability.

### 2.4 Embeddings: BGE-M3, still local
The PRD names BGE-M3 (1024-d, multilingual, handles long text better than
bge-small). OpenRouter serves chat models, not our embedding path, so
embeddings stay **local and offline**. BGE-M3 runs through Ollama
(`ollama pull bge-m3`, 1024-d), with Cognee's `EMBEDDING_PROVIDER` pointed at
it. sentence-transformers is the fallback if Cognee's Ollama embedding adapter
misbehaves, and Phase 3 needs torch anyway. After the switch we rebuild the
graph. We keep `bge-small` as a measured baseline and report the retrieval
difference instead of assuming BGE-M3 wins.

### 2.5 New embeddable `profile` text on `Material`
The robocrys `description` covers structure only. It never mentions band gap,
stability, or applications. We add a deterministic `profile` field: a
templated one-paragraph summary of each material's properties, classes,
domains, and missing fields. It is built from the same data, with no LLM, and
indexed as `Material_profile`. It helps semantic seeding and it is required
for a **fair** Vector RAG baseline (§7). Without it, the baseline would only be
searching crystal-structure prose, which would be a strawman comparison.

### 2.6 Neo4j stays a config swap, outside the Phase 2 critical path
The retriever reads through Cognee's `get_graph_engine()` and `get_graph_data()`
plus its own in-memory index, so it does not depend on the backend. The Neo4j
move (PRD target, README roadmap) is an optional closing milestone (M6), so it
cannot block retrieval work.

## 3. Query understanding

`QueryConstraints` (pydantic), filled by the LLM using schema-constrained
decoding. The rule-based parser also fills it in `--no-llm` mode and serves as
a fallback:

| Field | Example | Grounded against |
|---|---|---|
| `application_domains` | `["battery_cathode"]` | `ApplicationDomain.name` (synonym map: "cathode", "Li-ion" …) |
| `property_ranges` | `{band_gap: [2.0, null], energy_above_hull: [null, 0.05]}` | whitelisted `Material` numeric attrs |
| `property_classes` | `[("stability","stable")]` | `PropertyClass(kind,name)` |
| `include_elements` / `exclude_elements` | `["Li"]` / `["Co","Pb"]` | `Element.symbol` (names → symbols) |
| `crystal_systems`, `space_groups`, `formula_patterns`, `oxidation_states` | `["Hexagonal"]`, `["ABC3"]`, `["Ti4+"]` | respective nodes |
| `anchor_materials` | `["LiFePO4"]` ("similar to …") | `Material.formula` / `material_id` (fuzzy via `Material_formula`) |
| `semantic_text` | "layered, octahedral coordination" | residual free text for vector seeding |
| `top_k`, `sort_by` | 10, `band_gap desc` | — |

**Grounding step:** every value the LLM emits is checked against the graph's
vocabulary. For example, an unknown element or domain is rejected, and
"cubic" is normalised to `Cubic`. Anything that does not map to the vocabulary
is reported back in the response as "ignored constraint: …" and is never
silently dropped.

## 4. Retrieval engine (no LLM)

```
QueryConstraints
   │
   ├─ 1. SEED     structured: nodes named by constraints (domain/class/element/…)
   │              anchors:    resolved Material nodes
   │              semantic:   top-N vector hits on Material_profile/description
   │
   ├─ 2. EXPAND   k-hop traversal from seeds (default k=2, configurable):
   │              domain/class/element → Material; Material —similar_to→ Material;
   │              anchor → shared hubs (Element, OxidationState, FormulaPattern, SpaceGroup)
   │
   ├─ 3. FILTER   hard constraints on attributes + edges; for partial-coverage
   │              fields, material with value missing → "unknown", not pass/fail
   │              (policy configurable: exclude | keep-flagged)
   │
   ├─ 4. RANK     transparent weighted score, each component reported:
   │              constraint margin · graph support (#independent paths, similar_to weight)
   │              · semantic similarity · stability bonus
   │
   └─ 5. BUNDLE   evidence facts [E#] + the actual graph paths for each top-K result
```

- `graph_store.py` loads the graph once and builds indexes: by type, by
  name/formula, and adjacency. The `Graph` class moves here from `verify.py`,
  and `verify.py` imports it.
- Every constraint lists which graph path satisfied it, for example
  `Material —contains→ Li` or `—classified_as→ (stability, stable)`. That path
  is the "reasoning path" the PRD asks for, and the dashboard can draw it.
- The weights and `k` live in `config/retrieval.yaml`, next to
  `materials.yaml`.

## 5. Generation and explanation

- The prompt contains the query, the parsed constraints, and the evidence
  bundle. The instructions are: recommend only materials that appear in the
  bundle, cite `[E#]` for every factual clause, state when a property is
  unavailable, and mention the DFT band-gap underestimation caveat whenever
  band gaps are compared to experimental expectations.
- The output schema is:
  `{summary, recommendations:[{material_id, formula, rank, rationale, claims:[{text, evidence_ids}]}], caveats, ignored_constraints}`.
- Next, `grounding.py` verifies each claim (ID exists, numbers match cited
  facts within tolerance, material is in the retrieved set) and attaches a
  `grounded` flag and a traceability ratio to the response. The API returns
  ungrounded claims marked as such and never hides them.
- The final response contains the ranked candidates, a property comparison
  table (with missing values marked), the evidence bundle, the graph paths,
  the LLM explanation, and the grounding report.

## 6. New material ingestion: get its properties, relate it to the graph

**Goal.** A user names a material that is not in the 794-material graph.
The system gets its properties, works out how it connects to existing nodes
(Elements, SpaceGroup, OxidationStates, FormulaPattern, similar Materials, …),
and explains the connections using the same evidence and grounding format as
§5. The material can then serve as an anchor for queries ("find stable
materials like the one I just added").

**Crystal structures (CIF) are not accepted in Phase 2.** They go to the CGCNN
model in Phase 3 (§6.5). Phase 2 accepts only **identifiers and formulas**.

### 6.1 Accepted inputs

| Input | How properties are obtained | Provenance label |
|---|---|---|
| MP `material_id` (e.g. `mp-…`) | Fetch the summary and robocrys records from Materials Project with the Phase 1 `mp_client` fields, the same ~40 fields as the existing graph | `mp` (verified) |
| Formula in MP (e.g. `LaAlO3`, `MgAl2O4`; neither is in the current graph) | Look up MP by formula and pick the min-`energy_above_hull` polymorph (same rule as Phase 1) | `mp` (verified) |
| Formula **not** in MP | Composition-only entry: elements, atomic fractions, `chemsys`, `formula_anonymous`, and oxidation-state guesses via pymatgen `Composition`. Everything else is reported as **unavailable**, with nothing estimated, because a formula has no structure to feed CGCNN. | `derived` / `guessed` |

For a composition-only entry, rules that depend on properties (`PropertyClass`
buckets, band-gap and E_hull conditions in `ApplicationDomain` rules) are
reported as **"cannot evaluate: missing band_gap"** instead of false. Purely
compositional conditions (for example "contains Li and a transition metal")
are still evaluated and reported as partial matches.

### 6.2 Relating it to the existing graph

The same deterministic builders from `enrich.py` produce the new `Material`
DataPoint, through a new single-material path. Identity keys mean its edges
point at the **existing** hub nodes:

- `contains` → the existing `Element` nodes. An element the graph has not
  used yet (46 of 103 are used) gets a new node from `elements.json`.
- `has_space_group` → `SpaceGroup` → `CrystalSystem`, plus `member_of` →
  `ChemicalSystem`, `has_formula_pattern` → `FormulaPattern`,
  `has_oxidation_state` → `OxidationState`, `classified_as`, and
  `suitable_for`. Each reuses the existing node when one exists, and a new
  node is created only when it does not. MP entries get all of these;
  composition-only entries get only the composition-derived ones.
- **`similar_to` (property similarity).** Uses the same 10-dimension cosine
  as Phase 1. The z-score mean and standard deviation are **frozen at build
  time** in `artifacts/similarity_stats.json`, so a new material is scored on
  the same scale without recomputing the whole graph. The new material gets
  its own top-K outgoing edges, and the existing edges are left unchanged.
  The edges are computed only when the five core dimensions (band gap,
  formation energy, energy above hull, density, magnetization) have real
  values. Every MP entry has them; elastic and dielectric data are optional
  and mean-imputed, as in the build. Composition-only entries have none, and
  a vector of imputed zeros would give meaningless similarity scores.
- **Compositional neighbours.** Cosine over element-fraction vectors finds
  the existing materials with the most similar chemistry. This works for
  every input, and it is the only material-to-material link a
  composition-only entry gets.

### 6.3 Output: a relation report

For the new material, the report contains:
- its properties, each tagged with provenance: `mp` / `derived` / `guessed` /
  `unavailable`
- every edge created
- the hub nodes it shares with existing materials, ranked by how selective
  each hub is. Sharing the rare `ABC3` pattern with 44 materials says more
  than sharing `O` with 756.
- its top property-similar materials and compositional neighbours
- rule evaluations: fired, not fired, or cannot be evaluated

The report is built as an evidence bundle (§2.2), so the LLM can add an
explanation of how the material relates to the graph, and the grounding
checker verifies that explanation.

### 6.4 Persistence

- **Preview by default.** The material goes into an in-memory overlay on
  `graph_store`, so it can be queried and anchored for the session, but
  nothing is written to Cognee.
- **`persist=true`** writes it through the normal `add_data_points` path,
  with `origin="user"` and a timestamp, and refreshes the in-memory index.
  The fetched MP record is cached in `data/raw/`, just as in Phase 1,
  including its `structure`, so Phase 3 can compute CGCNN embeddings for it
  later.
- User-added materials are **excluded from the evaluation** (the benchmark
  pins the 794-material manifest), so evaluation results stay reproducible.

### 6.5 Handoff to Phase 3: structure input goes to CGCNN

When a user supplies a crystal structure (CIF), Phase 3 handles it this way:

1. Parse it (with safety limits: size and `nsites` caps, parse timeout,
   disordered structures rejected).
2. Build the crystal graph.
3. Run CGCNN to get predicted band gap and formation energy, plus the crystal
   embedding.
4. Relate the material to the graph using that embedding
   (`STRUCTURALLY_SIMILAR`) and the ionic-substitution model.

That is the PRD's `/predict` and `/similar-structures` flow. Phase 2 is built
so that this is a clean addition, not a rewrite:
- `ingest.py` takes a pluggable **property source** interface. Phase 2 ships
  `mp` and `composition`, and Phase 3 adds `cgcnn`, whose values carry the
  provenance label `predicted`.
- The relation report, the evidence bundle, the grounding checker, the
  persistence path, and `/materials/ingest` are all reused unchanged. Phase 3
  only adds a new input type and new kinds of evidence.

## 7. Evaluation (PRD: "comparative evaluation of GraphRAG and traditional Vector RAG")

**Benchmark set:** `config/benchmark_queries.yaml`, about 60 queries in five
tiers:
1. single constraint ("wide-gap oxides")
2. multi-constraint ("stable Li cathodes, band gap > 2 eV, no Co")
3. multi-hop / relational ("materials sharing an oxidation state with BaTiO3 that are dielectrics")
4. similarity-anchored ("like LiFePO4 but Mn-based")
5. negation / missing data ("bulk modulus > 150 GPa", where coverage is 18%)

Each query stores its **gold structured constraints**. The ground-truth
relevant set is computed by an **independent oracle** that reads the raw
MP JSON directly and does not use the graph code, so a bug in `enrich.py` or
in the retriever cannot also define the right answer. Relevance is graded for
nDCG by the fraction of constraints satisfied.

**Configurations** (LLM, prompt, and K held fixed; only retrieval varies, as
the PRD requires):

| Config | Retrieval |
|---|---|
| B1 Vector RAG | top-K over `Material_profile` + `description` embeddings only |
| B1b Cognee GRAPH_COMPLETION | off-the-shelf Cognee GraphRAG |
| B2 GraphRAG fixed-hop | seed + fixed 1-hop expansion, no ranking |
| **Proposed** | full seed → expand → filter → rank (§4) |
| Ablations | no semantic seeding · no `similar_to` · bge-small vs BGE-M3 |

**Metrics:**
- Recall@10, MRR, nDCG@10 per tier.
- **Constraint precision**: the share of returned items that violate a hard
  constraint. This should be 0 for Proposed and is where Vector RAG is
  expected to fail.
- **Traceability**: the share of grounded claims.
- **Parser accuracy**: LLM-parsed constraints compared with gold constraints,
  reported separately so parser errors are not mixed into retrieval scores.
- Latency.

Results go in `artifacts/eval/` plus a README section. The results are
reported as they come out, whether they favour the proposed method or not.

**Stated limitation:** the oracle measures whether retrieval is faithful to
the data. It does not measure scientific correctness. A small
**expert-anchored set** of known real-world answers (for example known olivine
cathodes, known transparent conducting oxides, known ferroelectric
perovskites) is added as a sanity check on external validity.

**Ingestion evaluation (leave-one-out).** Hold out about 50 materials,
stratified across the 8 clusters, and build the graph without them.
Re-ingest each one two ways, then compare against the full build:

| Ingestion path | Measured |
|---|---|
| By `material_id` or formula (MP path) | Recovery of categorical edges (`contains`, space group, chemsys, formula pattern, oxidation states, property classes, domains). **Target: 100%**, because it runs the same code on the same data. `similar_to` neighbour overlap against the full build: Jaccard@5 is reported |
| Composition-only (MP lookup disabled) | Oxidation-state guess agreement with MP's `possible_species`; compositional-neighbour overlap with the full build's `ChemicalSystem`/`Element` neighbourhood |

## 8. Interfaces

- **CLI:** `uv run python scripts/ask.py "…" [--no-llm] [--json] [--k 10]`
- **FastAPI** (`src/matgraph/api.py`, PRD endpoint names):
  - `POST /query` takes a question and options and returns the full response
    from §5.
  - `GET /material/{id}` returns stored properties plus the 1-hop graph
    neighbourhood.
  - `GET /health` reports readiness (graph loaded, vector store reachable,
    LLM reachable).
  - `POST /materials/ingest` is an addition to the PRD's endpoint table. It
    takes `{material_id | formula, persist: bool, explain: bool}`
    and returns the §6.4 relation report. The ingested material is then
    usable as an anchor in `/query` and appears in `/material/{id}`.
  - The graph and vector engine are loaded once at startup.
- **CLI for ingestion:**
  `uv run python scripts/add_material.py (--id mp-… | --formula LaAlO3) [--persist] [--no-llm]`
- **Results dashboard** (PRD's last Phase 2 box), with scope set by the
  decision in §11 (minimal FastAPI-served page):
  - query box
  - ranked results with score breakdown
  - property comparison table
  - evidence-path graph view (the query's subgraph only, not the full
    1.5k-node graph)
  - grounded / ungrounded claim highlighting
  - an "Add material" panel (ID or formula) that shows the relation report:
    property table with provenance tags, new edges, shared hubs, and
    neighbours drawn as a small subgraph

## 9. Repo layout (additions)

```
config/
  retrieval.yaml             # k-hop, ranking weights, missing-data policy, synonyms
  benchmark_queries.yaml     # evaluation set with gold constraints
src/matgraph/
  schema.py                  # + Material.profile, origin/provenance fields
  enrich.py                  # + build_profile(); frozen similarity stats; single-material build path
  graph_store.py             # load graph once; indexes; Graph (moved from verify.py); in-memory overlay
  ingest.py                  # resolve id/formula → property source (mp | composition) → DataPoint → relation report (§6)
  query_schema.py            # QueryConstraints, response models
  query_parser.py            # LLM parser + rule-based fallback + vocabulary grounding
  retriever.py               # seed / expand / filter / rank / paths
  evidence.py                # evidence bundle with stable [E#] ids
  llm.py                     # LiteLLM → OpenRouter wrapper: JSON-schema output, cache, throttle, fallback
  generator.py               # prompt + structured explanation
  grounding.py               # claim verification, traceability score
  api.py                     # FastAPI app
  eval/  oracle.py  baselines.py  metrics.py  run.py
scripts/
  ask.py  serve.py  evaluate.py  add_material.py
tests/
  test_query_parser.py  test_retriever.py  test_grounding.py  test_oracle.py  test_api.py
  test_ingest.py
```

## 10. Milestones and acceptance criteria

**M0: Foundations**
- Add `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, and `OPENROUTER_FALLBACK_MODEL` to `.env`
  and `.env.example` (placeholder only).
- Point Cognee's `LLM_*` settings at OpenRouter too, so the B1b baseline works,
  and drop `COGNEE_SKIP_CONNECTION_TEST`.
- Re-check the free roster and pin the models.
- `ollama pull bge-m3`, then switch embeddings to BGE-M3.
- Add `Material.profile`.
- Extract `graph_store.py`.
- Rebuild the graph.

✓ `build_graph.py` passes with the same node and edge counts plus
`Material_profile`. ✓ A vector-search smoke test over profiles returns
sensible results ("layered Li cobalt oxide" ranks LiCoO2 in the top 5).

**M1: Query understanding**
- `QueryConstraints`, the LLM parser, the rule fallback, and vocabulary
  grounding.

✓ Parser accuracy ≥ 90% field-level on the benchmark's gold constraints (LLM
mode). ✓ Unmappable terms show up in `ignored_constraints`.

**M2: Retrieval engine**
- Seed, expand, filter, rank, plus reasoning paths. No LLM.

✓ 0 hard-constraint violations on the whole benchmark using gold constraints.
✓ Every result carries ≥ 1 reasoning path. ✓ p95 retrieval < 1 s.

**M3: Grounded generation**
- Evidence bundle, generator, grounding checker, `ask.py`.

✓ Traceability ≥ 95% on the benchmark. ✓ 0 recommended materials outside the
retrieved set. ✓ Missing properties are stated, never invented (tested with
elastic-constraint queries).

**M4: New material ingestion (§6)**
- Freeze the similarity stats.
- Add the single-material build path and MP resolution (ID and formula).
- Add the composition-only path, the property-source interface, and
  compositional neighbours.
- Add the relation report, the preview overlay with `--persist`, and
  `add_material.py`.

✓ `LaAlO3` and `MgAl2O4`, neither currently in the graph, attach by formula
to the existing `La`/`Mg`/`Al`/`O` and `ABC3`/`AB2C4` nodes without creating
duplicate hub nodes (checked through node counts). ✓ Their `similar_to` edges
use the frozen stats. ✓ A formula that is not in MP is labelled
`derived`/`guessed`, lists band gap and related values as unavailable, and
still gets compositional neighbours. ✓ A preview leaves Cognee's stored counts
unchanged, and `--persist` increases them by exactly the new nodes and edges.

**M5: Evaluation**
- Oracle, baselines, metrics, the ingestion leave-one-out test, and the
  report.

✓ All configurations from §7 run end to end with one command. ✓ The MP-path
ingestion recovers 100% of categorical edges. ✓ The results table is in the
README.

**M6: Service and dashboard**
- FastAPI (`/query`, `/material/{id}`, `/materials/ingest`, `/health`) and the
  dashboard as a single FastAPI-served page (§11), including the
  "Add material" panel.

✓ API contract tests pass. ✓ The dashboard renders the evidence path for a
query and the relation report for an added material.

**M7 (optional): Neo4j**
- Config swap and docker-compose service. Re-run verification and evaluation.

✓ Identical counts and identical eval metrics.

**Definition of done for Phase 2:**
- One command answers a natural-language materials question with ranked
  candidates.
- Every factual claim cites graph or MP evidence, and a checker has verified
  the citations.
- A reproducible evaluation shows how graph retrieval compares with vector
  RAG, with the numbers reported whichever way they come out.
- A new material given by ID or formula gets its properties, each
  labelled with its source, and is connected to the existing graph through
  shared hubs and neighbour edges, with an explanation whose claims are
  checked against the evidence.

## 11. Decisions (confirmed 2026-09-29)

1. **LLM:** an OpenRouter free model (primary `qwen/qwen3.8-27b:free`, with a
   structured-output-capable fallback). You supply the API key, and it stays
   in `.env` only. Embeddings stay local (BGE-M3 via Ollama).
2. **Dashboard:** a minimal single-page dashboard served by FastAPI. The
   Next.js app moves to the deployment phase along with Docker and CI.
3. **Neo4j:** the optional last milestone (M7). Retrieval is built on Kuzu,
   then we swap and re-run the evaluation to confirm identical results.
4. **New material ingestion** (added on request): in Phase 2 scope as §6 and
   M4, for **MP IDs and formulas only**. **Crystal structure (CIF) input goes
   to the CGCNN model and is kept for Phase 3** (§6.5). Phase 2 does not parse
   structures, and it does not estimate properties that are missing.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Free model ignores the JSON schema or citation discipline | Only models with `structured_outputs` support are used; retry with the validation error; rule-based fallback; grounding checker drops uncited claims |
| OpenRouter free-tier rate limits or a model leaves the free list | Response cache; client-side throttle and backoff; fallback model; `--no-llm` mode; retrieval eval needs no LLM calls |
| Evaluation results drift because the free model changes | Model ID and the cached responses are recorded with every eval run; cached reruns reproduce exactly |
| Circular evaluation (graph and ground truth built by the same code) | Independent oracle over raw JSON, plus an expert-anchored subset |
| Sparse elastic/dielectric data looks like a retrieval failure | Explicit missing-data policy; tier-5 queries measure it; answers state what is missing |
| DFT band gaps misread as experimental | Caveat required in the prompt and checked in tests; the `PropertyClass` description already notes it |
| BGE-M3 is slower and larger than bge-small | Embedding happens once at build time; query-time cost is one short string; the ablation reports whether it helped |
| Composition-only entry's `similar_to` built on mostly-imputed vectors | Property similarity only when all 5 core dimensions are real; composition-only entries link through compositional neighbours instead |
| New material shifts similarity scale or duplicates hub nodes | Frozen z-score stats; identity keys reuse existing hubs; node-count assertions in `test_ingest.py` |
| Oxidation-state guesses are wrong | Labelled `guessed`; agreement with MP measured; never used to fire rules on their own |
| Cognee API churn (1.3.0) | Retriever depends only on `get_graph_engine`, `get_graph_data`, and `vector_engine.search`; version stays pinned |

## 13. Non-goals for Phase 2

- CGCNN or ALIGNN training
- Crystal-graph embeddings
- CrystalNN fingerprints
- `Structure` / `Prototype` / `CoordEnv` / `Species` nodes
- `STRUCTURALLY_SIMILAR` / `SUBSTITUTABLE_WITH` edges
- **Crystal structure (CIF) input** of any kind. Structures go to CGCNN for
  property prediction and structural relation in Phase 3; see the §6.5
  handoff.
- `/predict`, `/similar-structures`, `/similar-elements`, `/analyse`

All of the above are Phase 3.

Also out of scope:
- Production deployment, CI/CD, and staging/prod (deployment phase)
- Literature ingestion

Phase 3 will reuse the evidence-bundle and path format designed here, so a
`STRUCTURALLY_SIMILAR` edge becomes just another kind of evidence.

---

## 14. Implementation notes (added during the build)

These are findings from implementing the plan that changed or sharpened it.

- **Cognee 1.3.0 bug: all per-field vector collections held the same text.**
  - `index_data_points` makes one shallow `model_copy()` per index field and
    then mutates the shared `metadata` dict. As a result
    `Material_formula`, `Material_description`, and `Material_profile` were
    all embedded from the last field. Phase 1 was affected too.
  - Patched from our side in `matgraph/cognee_patches.py`, with a unit test.
  - `verify.py` now fails the build if the collections collapse together
    again.
- **Unphysical elastic tensors in Materials Project.**
  - 4 of 794 materials have negative moduli or negative universal anisotropy
    (Li(CuO2)2: bulk modulus −300,604 GPa).
  - They dominated the elastic z-scores in Phase 1's `similar_to` (bulk
    modulus std ≈ 25,000 GPa; ≈ 52 GPa after the fix).
  - The tensors are now discarded as a block (`enrich.elastic_status`) and
    reported as unphysical in the material's profile.
- **The `similar_to` gate for ingested materials is "all 5 core dimensions".**
  - It is not "≥ 6 of 10", as first planned.
  - Only about 20% of MP materials have elastic or dielectric data. Under the
    ≥ 6 rule, most MP records would have got no `similar_to` edges.
  - The leave-one-out check caught this: Jaccard was 0.4, and 1.0 after the
    fix.
- **Embedding throughput.**
  - Ollama serves one embedding at a time. Cognee runs 4 batches concurrently.
  - With the default batch size of 36, long robocrys descriptions queued past
    Cognee's 60 s request timeout.
  - The fix is `EMBEDDING_BATCH_SIZE=8`. A full rebuild takes about 10–15 min.
- **OpenRouter free tier.**
  - The key reports `is_free_tier: true`, which in practice means about 50
    free-model requests per day.
  - The Qwen pool is often rate-limited upstream. `llm.py` puts a model on a
    5-minute cooldown after a 429, so it doesn't burn requests.
  - `llm.py` stops at the daily cap with a clear message and falls back to
    the rule parser and templated explanation.
- **Oxidation-state guesses** (composition-only ingestion) match MP's
  `possible_species` exactly for 74% of the 626 materials checked. They are
  labelled `guessed` and never used to fire rules.
- **Heavy-tailed similarity features: z-scores are now clipped to ±3.**
  - Layered h-BN has a universal anisotropy of 461.6. That is extreme but
    not unphysical, and it gave z ≈ 184, so cosine similarity for any
    material with elastic data was decided by that one field.
  - With clipping, `similar_to` changed slightly: 3,964 → 3,969 edges.
  - Frozen-statistics ingestion now matches a full rebuild at Jaccard 0.98
    (0.967 before). The remaining differences are single near-tie swaps.
