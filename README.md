# MatGraphRAG

**An Explainable Graph-Based Retrieval-Augmented Reasoning System for Intelligent Materials Discovery**

> Project report: Phase 1 (knowledge graph construction) and Phase 2 (GraphRAG retrieval and recommendation)
> Status: **Phase 1 complete** · **Phase 2 complete** · Phase 3 (CGCNN structural embeddings) not yet started

---

## Table of Contents

1. [Overview](#1-overview)
2. [Project Status](#2-project-status)
3. [Dataset](#3-dataset)
4. [Graph Architecture](#4-graph-architecture)
5. [Construction Pipeline](#5-construction-pipeline)
6. [How Materials Interconnect](#6-how-materials-interconnect)
7. [How Materials Are Grouped](#7-how-materials-are-grouped)
8. [Verified Graph Statistics](#8-verified-graph-statistics)
9. [Example Graph Traversals](#9-example-graph-traversals)

**Part II: Phase 2**

10. [Phase 2 Overview](#10-phase-2-overview)
11. [Query Understanding](#11-query-understanding)
12. [Graph Retrieval](#12-graph-retrieval)
13. [Evidence and Grounded Explanations](#13-evidence-and-grounded-explanations)
14. [Adding New Materials](#14-adding-new-materials)
15. [Evaluation](#15-evaluation)
16. [API and Dashboard](#16-api-and-dashboard)

**Reference**

17. [Project Structure](#17-project-structure)
18. [Getting Started](#18-getting-started)
19. [Engineering Notes](#19-engineering-notes)
20. [Roadmap](#20-roadmap)
21. [Tech Stack](#21-tech-stack)

---

## 1. Overview

MatGraphRAG is a decision-support system for materials discovery: it turns the
[Materials Project](https://materialsproject.org) database into a knowledge graph
so that materials, their properties, their crystal chemistry, and their
application domains become traversable and explainable, rather than isolated
rows in a table. On top of the graph, Phase 2 adds graph-based retrieval and
LLM-generated recommendations, where every claim in an answer is traceable to
a concrete graph edge or database record.

Part I of this document reports on **Phase 1**: constructing that knowledge
graph. Part II (§10–16) reports on **Phase 2**: retrieval, explanation,
new-material ingestion, and their evaluation. No LLM is involved anywhere in
Phase 1: every node, edge, and classification
is produced by deterministic rules applied to data pulled directly from the
Materials Project API, using [Cognee](https://github.com/topoteretes/cognee)
as the graph/vector storage and pipeline layer.

## 2. Project Status

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | Knowledge graph construction from Materials Project data | ✅ Complete |
| **Phase 2** | GraphRAG retrieval + LLM-generated, evidence-backed recommendations | ✅ Complete |
| Phase 3 | CGCNN crystal-structure embeddings, structural similarity search, element substitution | Not started |

Phase 1 deliverable, as built and verified: a graph of **794 materials**
spanning **46 elements**, **106 space groups**, **143 chemical systems**,
**131 oxidation states**, **261 stoichiometric formula patterns**, and
**5 application domains**, interconnected by **14,686 edges** — built entirely
from cached, reproducible Materials Project data with zero LLM calls.

Phase 2 deliverable: natural-language questions answered by graph traversal
plus exact constraint filtering. Answers are LLM-written explanations in which
every claim cites a graph edge or Materials Project record, and a
deterministic checker verifies the citations. On a 55-question benchmark,
graph retrieval reaches **nDCG@10 0.910** with **0 constraint violations**,
against 0.308 / 69.6% for vector RAG (§15). New materials can be added by MP
id or formula and related to the existing graph.

## 3. Dataset

### Source

All data comes from the [Materials Project REST API](https://api.materialsproject.org)
(`/materials/summary/` for computed properties, `/materials/robocrys/` for
human-readable structure descriptions). Materials Project currently indexes
**154,377 materials** in total.

### Why not the full catalog

The full catalog was deliberately not used for Phase 1. Two reasons:

- **Material–material similarity is quadratic.** Computing pairwise similarity
  over 154,377 materials means ~11.9 billion pairs — not slow, mathematically
  intractable in this design. Full-scale structural similarity is Phase 3's
  job (CGCNN embeddings + an approximate-nearest-neighbour index), not
  Phase 1's.
- **A random or exhaustive pull produces a sparse, disconnected graph.** The
  design goal for Phase 1 is a *densely interconnected* graph — materials
  that actually share elements, structure types, and application relevance —
  not a flat catalog dump.

### Sampling strategy: chemistry-driven clusters

Instead, the dataset is assembled from eight **chemistry-driven clusters**,
each targeting a materials-science domain and each defined declaratively in
[`config/materials.yaml`](config/materials.yaml). A cluster expands into one
or more Materials Project queries (by element composition or exact chemical
system, always sorted by stability), and results are deduplicated **globally
across all clusters** by formula — keeping the lowest-energy-above-hull
(most stable) polymorph whenever a formula is reachable from more than one
cluster.

| Cluster | Domain | Query pattern | Stability cutoff |
|---|---|---|---|
| `battery_cathode_oxide` | Li–transition-metal oxides (layered/spinel cathodes) | Li + O + {Fe, Co, Mn, Ni, V, Ti, Cr, Zn, Cu, Sc, Nb, Mo}, ≤3 elements | E<sub>hull</sub> ≤ 0.08 eV/atom |
| `battery_cathode_phosphate` | Li–transition-metal phosphates (olivines) | Li + P + O + {Fe, Mn, Co, Ni, V, Ti, Cr}, ≤4 elements | ≤ 0.08 eV/atom |
| `semiconductor_iii_v` | III–V semiconductors | {B, Al, Ga, In} × {N, P, As, Sb} binaries | ≤ 0.08 eV/atom |
| `semiconductor_ii_vi` | II–VI semiconductors | {Zn, Cd, Hg} × {O, S, Se, Te} binaries | ≤ 0.08 eV/atom |
| `semiconductor_group_iv` | Group IV elemental/binary semiconductors | Si, Ge, C, Sn + binary combinations | ≤ 0.15 eV/atom |
| `oxide_binary` | Wide-gap and functional binary oxides | O + 33 candidate elements, exactly 2 elements | ≤ 0.08 eV/atom |
| `perovskite_oxide` | ABO₃ perovskites (dielectric/ferroelectric) | {Ba, Sr, Pb, Ca, K, Na} × {Ti, Zr, Nb, Hf, Sn, Ta} + O | ≤ 0.08 eV/atom |
| `magnetic_spinel_oxide` | AB₂O₄ spinel-type magnetic oxides | {Fe, Co, Ni, Mn, Cr} × {Fe, Co, Ni, Mn, Cr, Zn, Ti} + O | ≤ 0.08 eV/atom |

A global `exclude_elements` list drops radioactive and synthetic elements
(Tc, Pm, Po, At, Rn, Fr, Ra, and the actinides) from every query, since they
add no scientific value to a discovery-oriented graph.

**Result:** 8 clusters → ~2,000 raw candidate documents → **794 unique
materials** after cross-cluster deduplication.

### Properties captured

All of the following come from a single `/materials/summary/` query per
cluster sub-pattern — no additional API endpoints or requests are needed,
since Materials Project exposes elastic, dielectric, oxidation-state, and
structural data as extra fields on the same summary document.

| Category | Fields | Coverage (of 794) |
|---|---|---|
| Core (band gap, stability, density, magnetism, symmetry, …) | `band_gap`, `formation_energy_per_atom`, `energy_above_hull`, `density`, `total_magnetization`, `ordering`, `symmetry`, `theoretical`, … | 100% |
| Electronic structure | `cbm`, `vbm`, `efermi`, `is_magnetic`, `num_magnetic_sites` | 100% |
| **Elastic** (PRD: "Elastic Properties") | `bulk_modulus` (VRH), `shear_modulus` (VRH), `universal_anisotropy`, `homogeneous_poisson` | **18.0%** (143/794) |
| **Dielectric** (PRD: "Dielectric Properties") | `e_total`, `e_electronic`, `e_ionic`, refractive index `n` | **19.9%** (158/794) |
| **Oxidation states** (PRD: "Oxidation States") | `possible_species` → `OxidationState` nodes | 93.1% (739/794) |
| Full crystal structure | `structure` (lattice + atomic positions) — cached to disk for Phase 3, not added to the graph (see §19) | 100% |

Elastic and dielectric tensors require an extra DFPT calculation Materials
Project hasn't run for every entry, so real coverage is much lower than core
properties — this was measured directly against the full 794-material set,
not estimated (an earlier spot-check on 50 materials suggested ~84%/~70%
coverage, which turned out to be a sampling-bias artifact; see §19).

## 4. Graph Architecture

### Design principle

> Continuous numeric values stay as **attributes** on the `Material` node
> (exact values, needed for filtering and comparison). Anything used for
> **grouping or traversal** becomes its own **node**. Derived material–material
> relationships become **weighted edges**.

This was a deliberate, revisited decision: an earlier iteration turned every
property (including raw floats like density) into its own node type. That
was reverted — singleton value-nodes for near-unique floats add graph
complexity without adding groupability, which is the entire point of a
knowledge graph over a flat table. `band_gap` and `energy_above_hull`
*are* additionally represented as categorical nodes (`PropertyClass`, see
below) precisely because their **bucketed** form is what's actually shared
and traversable — the raw float isn't. Elastic and dielectric properties
follow the same rule: they stay as attributes (§3), since bucketing them
usefully would need domain-specific thresholds not yet defined, whereas
`possible_species` and `formula_anonymous` *are* inherently categorical
(a small, shared vocabulary) and became nodes directly — no bucketing needed.

### Node types

| Node type | Represents | Identity (dedup key) | Count |
|---|---|---|---|
| `Material` | One Materials Project entry (ground-state polymorph per formula, per cluster) | `material_id` | 794 |
| `Element` | A chemical element and its periodic properties | `symbol` | 46 |
| `SpaceGroup` | A crystallographic space group | `symbol` + `number` | 106 |
| `CrystalSystem` | One of the 7 crystal systems (cubic, hexagonal, …) | `name` | 7 |
| `ChemicalSystem` | A material's element set as a system (e.g. `Fe-Li-O-P`) | `chemsys` | 143 |
| `PropertyClass` | A categorical bucket a material falls into (band-gap class, stability class, magnetic ordering) | `kind` + `name` | 11 |
| `ApplicationDomain` | A materials-science use case (battery cathode, semiconductor device, …) | `name` | 5 |
| `OxidationState` | An ionic species (e.g. `Fe2+`, `O2-`) a material contains | `species` | 131 |
| `FormulaPattern` | Anonymized stoichiometric pattern (e.g. `ABC3` for perovskites) | `pattern` | 261 |

Every shareable node type declares an identity key, so re-running the
pipeline — or two different materials both containing Oxygen — resolves to
the **same graph node** rather than creating duplicates. This identity
mechanism is what makes the graph's hub structure possible (see §6).

### Relationship types

| Edge | Direction | Meaning | Weight / properties |
|---|---|---|---|
| `contains` | Material → Element | Material's constituent elements | — |
| `has_space_group` | Material → SpaceGroup | Crystallographic space group | — |
| `member_of` | Material → ChemicalSystem | Which chemical system it belongs to | — |
| `classified_as` | Material → PropertyClass | Band-gap / stability / magnetic-ordering bucket | — |
| `suitable_for` | Material → ApplicationDomain | Which application rule(s) it satisfies | `rule`: the exact rule text that fired (explainability) |
| `similar_to` | Material → Material | Property-vector similarity to its nearest neighbours | `weight`: cosine similarity (0–1) |
| `crystal_system` | SpaceGroup → CrystalSystem | Which crystal system a space group belongs to | — |
| `has_oxidation_state` | Material → OxidationState | Ionic species present in the material | — |
| `has_formula_pattern` | Material → FormulaPattern | Which stoichiometric pattern it matches | — |

## 5. Construction Pipeline

```
config/materials.yaml (clusters, thresholds, rules)
        │
        ▼
┌───────────────────────┐
│ 1. FETCH               │  mp_client.py
│  - one query per       │  → Materials Project /materials/summary/
│    cluster sub-pattern │  → sorted by energy_above_hull, capped per query
│  - dedupe by formula    │
│    across all clusters │
│  - batch-fetch robocrys│  → /materials/robocrys/ (up to 100 material_ids/call)
│    structure text      │
└──────────┬─────────────┘
           ▼
  data/raw/<material_id>.json × 794  +  _manifest.json   (cached; reproducible)
           │
           ▼
┌───────────────────────┐
│ 2. ENRICH (no LLM)     │  enrich.py
│  - classify band_gap,  │  → PropertyClass nodes
│    stability, ordering │
│  - apply domain rules  │  → ApplicationDomain edges + fired-rule text
│  - compute similar_to  │  → top-K nearest neighbours per material
│    (cosine, z-scored)  │
│  - build DataPoint     │  → Material/Element/SpaceGroup/... objects
│    graph objects       │     (schema.py)
└──────────┬─────────────┘
           ▼
┌───────────────────────┐
│ 3. WRITE TO COGNEE     │  pipeline.py
│  Task(build_materials) │  → cognee's low-level pipeline
│  Task(add_data_points) │  → writes graph store + vector store together,
│                        │     deduplicating on each node's identity_fields
└──────────┬─────────────┘
           ▼
  Cognee-managed stores: embedded Kuzu (graph) + LanceDB (vectors) + SQLite (metadata)
           │
           ▼
┌───────────────────────┐
│ 4. VERIFY              │  verify.py
│  - node/edge counts    │  → reads the graph directly via get_graph_data(),
│  - structural queries  │     no LLM / no natural-language search needed
│  - visualization       │  → artifacts/graph.html
└────────────────────────┘
```

The entry point is [`scripts/build_graph.py`](scripts/build_graph.py), which
runs steps 2–4 (step 1 is a separate script since fetching is the expensive,
rate-limited part and should be cached independently of graph rebuilds).

## 6. How Materials Interconnect

Interconnection isn't a side effect here — it's what the identity-key +
shared-node design in §4 produces directly. Concretely, in the built graph:

- **Elements act as hubs.** Every material's `contains` edges point at the
  *same* `Element` node for a given symbol. Oxygen, for instance, is one
  node — but **756 of the 794 materials** (95%) point to it, meaning any two
  oxide materials are two hops apart via Oxygen regardless of how different
  their other elements are.
- **Chemical systems group materials by exact composition.** All ternary
  Ba–Ti–O compounds converge on one `ChemicalSystem` node, distinct from
  binary Ba–O materials — this is a finer-grained grouping than individual
  elements.
- **Crystal symmetry groups by structure family.** 106 distinct space groups
  collapse onto just 7 `CrystalSystem` nodes, so e.g. every cubic material —
  regardless of composition — is reachable from every other cubic material
  in two hops.
- **PropertyClass nodes group by behavior, not composition.** A `wide_gap`
  material and another `wide_gap` material may share no elements at all, but
  are still one hop apart through the same bucket node.
- **Oxidation states group by ionic chemistry.** `O2-` alone connects **695
  of 794 materials** (87%); `Li+` connects 227. Two materials with no shared
  elements can still be one hop apart if they share an oxidation state —
  this is also the anchor Phase 3's ionic-substitution model will reason over.
- **Formula patterns group by stoichiometric shape, independent of
  composition.** `BaTiO3` and `CaHfO3` share nothing element-wise but are
  both `ABC3` — 44 materials share that one node. 261 distinct patterns
  cover 794 materials, so this dimension is denser than `ChemicalSystem`
  (143) but sparser than `Element` (46), a genuinely different grouping axis.
- **`similar_to` edges add a direct, weighted shortcut.** Rather than a flat
  similarity threshold (which stops scaling once the dataset is larger than
  a handful of materials — see §19), each material links to its **top-5**
  nearest neighbours by cosine similarity over a **10-dimensional**, z-scored
  property vector spanning electronic (`band_gap`), thermodynamic
  (`formation_energy_per_atom`, `energy_above_hull`), structural-density
  (`density`), magnetic (`total_magnetization`), mechanical (`bulk_modulus_vrh`,
  `shear_modulus_vrh`, `universal_anisotropy`, `homogeneous_poisson`), and
  dielectric (`dielectric_total`) behavior — keeping only pairs with
  similarity ≥ 0.6. Fields with partial coverage (elastic, dielectric) are
  mean-imputed rather than dropped, so a missing value contributes neutrally
  instead of skewing the comparison (see §19 for which fields were
  deliberately left out, and why). Z-scores are clipped to ±3 so no single
  heavy-tailed field (e.g. h-BN's elastic anisotropy of 461) dominates the
  cosine, and unphysical elastic tensors are discarded (both Phase 2 fixes,
  §19). This produced 3,969 directed `similar_to` edges in the current graph.

The net effect: **no material in the graph is an isolated island.** Every
material has at least a `contains` edge (verified structurally on every
build — see §8), and the shared-node design means the graph's connectivity
grows *faster* than its material count as clusters overlap in elements and
chemistry.

## 7. How Materials Are Grouped

Grouping happens through three independent, fully deterministic mechanisms.
The first two are configured in [`config/materials.yaml`](config/materials.yaml),
so every bucket/rule decision is inspectable and traceable to a concrete
threshold; the third comes directly from categorical fields Materials
Project already provides, with no thresholds to define.

### 7.1 Property classification (buckets)

| Property | Buckets | Thresholds |
|---|---|---|
| **Band gap** | `metal` / `narrow_gap` / `semiconductor` / `wide_gap` | 0 / 0–1 / 1–3 / >3 eV (DFT-scale; Materials Project band gaps are systematically underestimated relative to experiment, so thresholds are tuned for DFT values, not experimental ones) |
| **Stability** | `stable` / `metastable` / `unstable` | E<sub>hull</sub> ≤ 1×10⁻⁶ / ≤ 0.05 / > 0.05 eV/atom |
| **Magnetic ordering** | one `PropertyClass` node per distinct value Materials Project reports (FM, AFM, FiM, …) | — (skipped when reported as "Unknown") |

### 7.2 Application domain rules

Each rule fires independently — a material can satisfy several domains at
once — and the exact rule text that fired is stored **on the edge itself**
(`suitable_for.rule`), so a recommendation downstream can point at precisely
why a material was included rather than asserting it opaquely.

| Domain | Rule |
|---|---|
| `battery_cathode` | Contains Li **and** a transition metal, **and** E<sub>hull</sub> ≤ 0.05 eV/atom |
| `semiconductor_device` | 0 < band gap ≤ 3.5 eV **and** not metallic |
| `photocatalyst` | Contains O **and** 1.5 ≤ band gap ≤ 3.5 eV |
| `dielectric` | Band gap > 2 eV **and** |total magnetization| < 0.1 (non-magnetic) |
| `magnetic_material` | Ordering ∈ {FM, FiM} **and** |total magnetization| ≥ 0.1, **or** ordering = AFM |

The transition-metal set used by `battery_cathode` is: Sc, Ti, V, Cr, Mn, Fe,
Co, Ni, Cu, Zn, Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd.

Note the `battery_cathode` rule's 0.05 eV/atom cutoff is intentionally
*tighter* than the 0.08 eV/atom used to fetch cluster candidates in the
first place — the fetch stage casts a slightly wider net so metastable
near-misses are still present in the graph (reachable via `contains`,
`member_of`, etc.), even when they don't pass the stricter domain-membership
bar.

### 7.3 Categorical fields (no thresholds needed)

Two Materials Project fields are already categorical, so they group
materials directly rather than through a bucketing rule:

- **Oxidation states** (`possible_species`) — e.g. `BaTiO3` groups with every
  other material containing `Ti4+`, independent of what else those materials
  contain.
- **Formula patterns** (`formula_anonymous`) — e.g. every `AB2C4`-pattern
  material groups together regardless of which elements A, B, and C actually
  are, capturing stoichiometric shape (spinel-like 1:2:4 ratios, in this case)
  as a grouping dimension distinct from both composition and symmetry.

## 8. Verified Graph Statistics

Numbers below are from the current build (`uv run python scripts/build_graph.py`),
read directly out of Cognee's graph store — not estimated.

**Nodes — 1,504 total**

| Type | Count |
|---|---|
| Material | 794 |
| FormulaPattern | 261 |
| ChemicalSystem | 143 |
| OxidationState | 131 |
| SpaceGroup | 106 |
| Element | 46 |
| PropertyClass | 11 |
| CrystalSystem | 7 |
| ApplicationDomain | 5 |

**Edges — 14,686 total**

| Relationship | Count |
|---|---|
| `similar_to` | 3,969 |
| `has_oxidation_state` | 2,319 |
| `contains` | 2,269 |
| `classified_as` | 2,135 |
| `suitable_for` | 1,506 |
| `has_formula_pattern` | 794 |
| `has_space_group` | 794 |
| `member_of` | 794 |
| `crystal_system` | 106 |

**Application domain distribution** (a material can belong to more than one; sums to 1,506, matching `suitable_for` edge count exactly)

| Domain | Materials |
|---|---|
| `semiconductor_device` | 474 |
| `magnetic_material` | 328 |
| `photocatalyst` | 249 |
| `battery_cathode` | 237 |
| `dielectric` | 218 |

**Structural integrity checks** (asserted on every build, in `verify.py`):

- ✅ Every `Material` node has at least one `contains` edge (no disconnected materials)
- ✅ 794 `Material` nodes present, matching the fetch manifest exactly
- ✅ `Material_formula` / `_description` / `_profile` vector collections embedded from distinct text (guards a cognee bug, §19)

## 9. Example Graph Traversals

These are real outputs from `verify.py` against the current graph, not
illustrative pseudo-queries:

- **Oxygen hub** — 756 of 794 materials trace back to the same `O` node.
- **"Which Li-containing materials are thermodynamically stable?"** → 80
  materials (`Li2O`, `Li2MnO3`, `Li(CoO2)2`, …), found by intersecting
  `contains → Li` with `classified_as → (stability, stable)`.
- **"What connects LiFePO4 and Fe2O3?"** → shared `Element` nodes `Fe` and
  `O`, found in two hops each.
- **"Which materials are wide-band-gap?"** → 143 materials, top of the list
  led by `Al2O3` (5.87 eV) and `SiO2` (5.68 eV).
- **"Which materials are battery cathode candidates?"** → 237 materials,
  each with the exact rule text that qualified it stored on its edge.
- **"What connects BaTiO3 to CaHfO3 despite sharing no elements?"** → both
  match the `ABC3` `FormulaPattern` node (44 materials total).
- **"Which materials contain Ti4+?"** → 107 materials, found in one hop via
  `has_oxidation_state`.
- **Nearest-neighbour similarity** — e.g. `KNb13O33 ↔ NaNb13O33` at cosine
  similarity 1.000 over the full 10-dimensional property vector (near-identical
  behavior despite being different formulas/polymorphs).

---

# Part II — Phase 2: GraphRAG Retrieval and Recommendation

## 10. Phase 2 Overview

Phase 2 answers natural-language materials questions with the Phase 1 graph.
For example:

> *"stable, cobalt-free Li cathode materials with a band gap above 2 eV"*

The answer has three parts:

- **ranked candidates**, found by **traversing and filtering the graph**
- a **property comparison table**, with values Materials Project doesn't have
  shown as "n/a" rather than guessed
- an **LLM-written explanation** in which every claim cites a numbered fact.
  A deterministic checker has verified each citation against the graph or the
  Materials Project record.

```
question
   │
   ▼  1. PARSE      LLM (OpenRouter, JSON-schema output) → typed QueryConstraints
   │                 └ rule-based fallback; every value grounded against graph vocabulary
   ▼  2. RETRIEVE   seed constraint nodes → reverse-traverse to Materials → intersect
   │                 → enforce numeric ranges / flags / exclusions on attributes
   │                 → rank (constraint margin · anchor proximity · BGE-M3 similarity · stability)
   ▼  3. EVIDENCE   flatten into atomic facts [E1] [E2] … (MP field or graph edge + rule text)
   │                 incl. explicit "NOT AVAILABLE" facts for missing properties
   ▼  4. EXPLAIN    LLM writes claims citing fact ids (or a templated, no-LLM explanation)
   ▼  5. GROUND     checker: ids exist · every number matches a cited fact · no unretrieved materials
                     → traceability = grounded claims / all claims
```

**Design choice: the LLM never selects or filters materials.** It does two
jobs only:

- turn the question into constraints
- phrase the explanation

Numeric requirements such as `band_gap ≥ 2 eV`, `E_hull ≤ 0.05`, or "no
cobalt" are enforced exactly against node attributes and edges. Neither
semantic search nor an LLM reading triplets can enforce them reliably.

## 11. Query Understanding

`QueryConstraints` (`src/matgraph/query_schema.py`) is the contract between
parsing and retrieval. It covers:

- application domains
- numeric property ranges over 13 properties, with units
- boolean flags (direct gap, metallic, …)
- property classes
- required and excluded elements
- crystal systems, space groups, formula patterns
- oxidation states
- anchor materials ("similar to X")
- sorting ("highest bulk modulus")

The LLM fills this schema through **JSON-schema-constrained output**. The
enumerated fields are `Literal`s, so the model sees the allowed values
directly. After parsing, `ground_constraints` checks every free-form value
against the graph's vocabulary. For example, "lithium" becomes `Li` and
`Ti+4` becomes `Ti4+`. Anything that doesn't map is reported back to the user
as an *ignored constraint* with a reason; it is never silently dropped.

`parse_with_rules` is a conservative regex/synonym parser. It handles:

- "Co-free" and "free of lead"
- "Li-ion" is not a lithium requirement
- "dielectric constant" is a property, not the dielectric domain

It runs in `--no-llm` mode, in tests, and whenever OpenRouter is unavailable.

**Hybrid parsing.** The free LLMs often return only *part* of the schema. For
example, from "Manganese cathode materials with energy above hull below 0.02
eV/atom" one returned just `include_elements: [Mn]`, dropping the domain and
the range. So when the LLM parse succeeds, the rule parser **fills fields the
LLM left empty. It never overrides a value the LLM returned**, and the answer
is labelled `llm+rules`. Measured results are in §15.

## 12. Graph Retrieval

`src/matgraph/retriever.py`, no LLM:

1. **Seed and expand.** Each node constraint becomes a graph node:
   - ApplicationDomain, PropertyClass, Element, OxidationState, FormulaPattern,
     SpaceGroup
   - CrystalSystem, which is two hops away via SpaceGroup

   Each node reverse-traverses to the Materials that reach it, and the
   resulting sets are intersected.
2. **Filter.** Numeric ranges, boolean flags, and element exclusions are
   checked against Material attributes. The missing-data policy is
   configurable (`config/retrieval.yaml`). The default excludes materials that
   lack the value and **reports how many were excluded**, which matters
   because elastic and dielectric data cover only about 20% of materials.
3. **Rank.** Every candidate that passes gets a transparent weighted score,
   with each component reported:
   - constraint margin
   - **anchor proximity**: IDF-weighted overlap of shared hub nodes plus
     direct or 2-hop `similar_to` weight
   - **BGE-M3** embedding similarity to the question, over the material's
     profile and robocrys description
   - stability
4. **Reasoning paths.** Every satisfied constraint carries the graph path that
   proves it, e.g. `Material:AlN -has_space_group-> SpaceGroup:P6_3mc
   -crystal_system-> CrystalSystem:Hexagonal`. The dashboard draws these.

Retrieval runs in a few milliseconds. The graph (about 1.5k nodes and 15k
edges) is loaded into memory once, through backend-independent
`get_graph_data()`, so the Neo4j swap remains a config change. Embedding the
question adds about 1 s on the local Ollama BGE-M3.

## 13. Evidence and Grounded Explanations

`evidence.py` flattens the retrieval result into atomic facts, each tagged
with its source:

```
[E4]  Li2MnP2O7 band_gap = 4.3677 eV.                   (source: Materials Project summary for mp-aaacqmek, field band_gap)
[E11] Li2MnP2O7: NOT AVAILABLE in Materials Project: bulk_modulus_vrh, shear_modulus_vrh, …
[E12] Material:Li2MnP2O7 -suitable_for-> ApplicationDomain:battery_cathode
      (rule: "contains Li and a transition metal, and energy_above_hull <= 0.05 eV/atom")
[E2]  Band gaps are DFT (GGA/GGA+U) values … systematically underestimate experimental band gaps
```

The LLM (`generator.py`) must cite fact ids for every claim, copy numbers
exactly, recommend only retrieved materials, and state explicitly when data
is missing. `grounding.py` then checks each claim deterministically:

- **Citations.** Every cited id must exist.
- **Numbers.** Every number in the claim must match a value in the cited
  facts, within the precision it was written with, or come from the user's
  own question.
- **Materials.** Every formula the claim mentions must be covered by the
  evidence.

Claims that fail are shown **flagged, not hidden**. The share that passes is
the PRD's *traceability* metric.

## 14. Adding New Materials

`scripts/add_material.py` and `POST /materials/ingest` take a **Materials
Project id or a formula**. The pipeline gets the material's properties and
relates it to the existing graph.

**Crystal-structure (CIF) input is deliberately not accepted.** Structures go
to the CGCNN model in Phase 3.

Where the properties come from:

| Input | Properties | Provenance |
|---|---|---|
| MP id, or formula found in MP | full summary record (min-E<sub>hull</sub> polymorph for a formula) | `mp` |
| formula not in MP | only what the formula determines: elements, fractions, chemsys, formula pattern, oxidation-state *guesses*; every computed property **unavailable, never estimated** | `derived` / `guessed` |

How the new material connects to the graph:

- **Shared hubs.** It is built with the same `NodeFactory` as the full graph.
  Identity-keyed hubs (Element, SpaceGroup, OxidationState, FormulaPattern, …)
  get the *same node ids* as the existing ones, so it attaches to them instead
  of duplicating them.
- **Relation report.** The report lists:
  - shared hubs, ranked by selectivity (sharing `ABC3` with 44 materials says
    more than sharing `O` with 756)
  - `similar_to` neighbours on the **frozen build-time z-score scale**
  - compositional neighbours
  - every application rule as fired, not fired, or *cannot evaluate*
- **Explanation.** The report can be explained by the LLM and grounded like a
  query answer.
- **Persistence.** The default is preview only: the material goes into an
  in-memory overlay and is usable as a query anchor. `--persist` writes it to
  Cognee, and the next full rebuild replays it from the local cache.

## 15. Evaluation

`uv run python scripts/evaluate.py [--llm-sample N] [--ingestion N]`
writes `artifacts/eval/results.{json,md}`.

- **Benchmark** (`config/benchmark_queries.yaml`): 55 questions in five
  tiers:
  1. single constraint
  2. multi-constraint
  3. multi-hop / relational
  4. similarity-anchored
  5. sparse elastic/dielectric data

  Each question has hand-written **gold constraints**.
- **Independent oracle** (`eval/oracle.py`): the relevant set is recomputed
  from the raw Materials Project JSON by code that imports nothing from the
  graph build or the retriever. A bug there can't also define the right
  answer.
- **Configurations.** The LLM, prompt, and K are fixed; only retrieval varies:
  - **B1** vector RAG over profile and description embeddings
  - **B2** generic fixed-hop GraphRAG (entity seeds, one-hop expansion, no
    attribute filtering)
  - **Proposed**
  - two ablations

### Retrieval results (gold constraints, no LLM)

| Config | Recall@10 | MRR | nDCG@10 | Top-10 constraint violations |
|---|---|---|---|---|
| B1 Vector RAG | 0.326 | 0.386 | 0.308 | 69.6% |
| B2 fixed-hop GraphRAG | 0.613 | 0.663 | 0.598 | 41.8% |
| **Proposed** | **0.926** | **0.945** | **0.910** | **0.0%** |
| Ablation: no semantic ranking | 0.911 | 0.907 | 0.890 | 0.0% |
| Ablation: no `similar_to` in anchor score | 0.953 | 0.991 | 0.935 | 0.0% |

nDCG@10 by tier:

| Config | T1 single | T2 multi-constraint | T3 multi-hop | T4 similarity | T5 sparse data |
|---|---|---|---|---|---|
| B1 Vector RAG | 0.654 | 0.294 | 0.238 | 0.187 | 0.046 |
| B2 fixed-hop GraphRAG | 0.847 | 0.729 | 0.870 | 0.290 | 0.056 |
| **Proposed** | **1.000** | **1.000** | **1.000** | 0.517 | **1.000** |
| no `similar_to` | 1.000 | 1.000 | 1.000 | **0.648** | 1.000 |

What these numbers do and don't show:

- **Filter tiers (1, 2, 3, 5).**
  - With correct constraints, graph retrieval matches the independent
    oracle *exactly*. This validates that the graph build and the retriever
    are faithful to the raw data.
  - Most of the gap to the baselines is **constraint enforcement**. Vector
    RAG returns wide-gap *look-alikes* for "band gap above 4 eV". Fixed-hop
    GraphRAG reaches the right entities but can't apply numeric ranges or
    exclusions (tier 5: 0.056).
  - Because these configurations start from gold constraints, the end-to-end
    quality of the system rests on **question parsing**. That is measured
    separately below.
- **Similarity (tier 4) is the open problem.**
  - Relevance there means *structural-stoichiometric analogue*: the same
    anonymized formula, plus the same crystal system for full credit.
  - The Phase 1 `similar_to` edges measure *property* similarity, and they
    **hurt** on this definition (0.517 with them, 0.648 without). The
    ablation beats the default configuration here.
  - This is reported as found rather than tuned away on the test set. It is
    exactly the gap Phase 3's structural (CGCNN) similarity is meant to close.
- **Circularity.** The oracle checks fidelity to the data, not the scientific
  correctness of the rules or of DFT values.

<!-- LLM_RESULTS -->

### New-material ingestion (leave-one-out, 50 materials stratified across clusters)

| Check | Result |
|---|---|
| MP path: categorical edges recovered vs. the full build | **100%** |
| MP path: `similar_to` overlap vs. a rebuild including the material (frozen-scale z-scores) | Jaccard **0.98**. The remaining differences are single near-tie swaps. |
| Composition-only path: formula pattern / chemical system match MP | 100% / 100% |
| Composition-only path: guessed oxidation states exactly equal MP's `possible_species` | 70%. This is why they are labelled `guessed` and never fire rules. |


## 16. API and Dashboard

`uv run python scripts/serve.py`, then open <http://localhost:8000>.

| Endpoint | Purpose |
|---|---|
| `POST /query` | question → parsed constraints, ranked candidates, comparison table, evidence, explanation, grounding report |
| `GET /material/{id}` | stored properties and 1-hop graph neighbourhood |
| `POST /materials/ingest` | MP id or formula → relation report (optionally explained, optionally persisted) |
| `GET /health` | graph loaded, embeddings reachable, LLM configured |

The dashboard is one static page with no build step. It shows:

- how the question was understood (constraint chips, ignored constraints)
- the ranked candidates with their score breakdown
- an interactive **evidence-path graph** for the selected candidate
- the property comparison table
- the explanation, with each claim marked grounded ✓ or ✗ and its cited
  facts on hover
- an **Add material** tab that renders the relation report and a subgraph of
  how the new material connects

---

# Reference

## 17. Project Structure

```
meta_graph/
├── config/
│   ├── materials.yaml          # clusters, buckets, domain rules, similarity config
│   ├── elements.json           # full 103-element periodic table (generated)
│   ├── retrieval.yaml          # Phase 2: ranking weights, missing-data policy, anchor scoring
│   └── benchmark_queries.yaml  # Phase 2: 55-question evaluation set with gold constraints
├── data/raw/                    # cached MP records, one JSON per material_id (gitignored)
├── src/matgraph/
│   ├── schema.py                # Cognee DataPoint node/edge types
│   ├── mp_client.py             # Materials Project REST client (clusters + single lookups)
│   ├── enrich.py                # rules, similarity, profiles, NodeFactory graph builder
│   ├── pipeline.py              # Cognee pipeline wiring (build + add_to_graph)
│   ├── cognee_patches.py        # workaround for a cognee 1.3.0 indexing bug (§19)
│   ├── verify.py                # structural + vector-index verification
│   ├── graph_store.py           # in-memory Graph, KnowledgeBase (graph + vector search)
│   ├── query_schema.py          # QueryConstraints contract
│   ├── query_parser.py          # LLM parser, rule-based fallback, vocabulary grounding
│   ├── retriever.py             # seed / expand / filter / rank / reasoning paths
│   ├── evidence.py              # evidence bundle with [E#] facts
│   ├── llm.py                   # OpenRouter via LiteLLM: fallback, cache, throttle, cooldown
│   ├── generator.py             # LLM + templated explanations
│   ├── grounding.py             # claim verification, traceability
│   ├── answer.py                # end-to-end question answering
│   ├── ingest.py                # new-material ingestion + relation report
│   ├── api.py, static/          # FastAPI service + dashboard
│   └── eval/                    # oracle, baselines, metrics, runner, ingestion LOO
├── scripts/
│   ├── fetch_materials.py       # fetch + cache clusters from Materials Project
│   ├── build_graph.py           # build, write to Cognee, visualize, verify
│   ├── ask.py                   # ask a question (CLI)
│   ├── add_material.py          # add/relate a material by MP id or formula
│   ├── evaluate.py              # run the evaluation -> artifacts/eval/
│   ├── serve.py                 # API + dashboard
│   └── generate_elements.py     # regenerates config/elements.json via pymatgen
├── tests/                       # 44 unit/API tests (synthetic graph; no network)
├── artifacts/eval/              # evaluation results (json + md)
├── docs/PHASE1_PLAN.md, docs/PHASE2_PLAN.md
└── .cognee_system/              # Cognee's embedded stores (gitignored)
```

## 18. Getting Started

### Setup

```bash
uv sync
cp .env.example .env    # fill in MP_API_KEY and OPENROUTER_API_KEY
ollama serve & ollama pull bge-m3   # local embeddings (BGE-M3, 1024-d)
```

### Build the graph

```bash
uv run python scripts/fetch_materials.py   # fetch + cache from Materials Project (once)
uv run python scripts/build_graph.py       # build + embed + verify (~15 min with BGE-M3)
```

`build_graph.py` prunes and rebuilds from scratch by default. Options:

- `--no-reset` adds on top of the existing state.
- `--skip-visualize` skips the HTML render.
- `--skip-user-materials` stops it from re-adding materials persisted with
  `add_material.py --persist`.

### Use it

```bash
uv run python scripts/ask.py "stable cobalt-free Li cathodes with band gap above 2 eV"
uv run python scripts/ask.py "..." --no-llm         # rule parser + templated explanation
uv run python scripts/add_material.py LaAlO3         # preview how a new material relates
uv run python scripts/add_material.py mp-2920 --persist
uv run python scripts/serve.py                       # API + dashboard on :8000
uv run python scripts/evaluate.py --llm-sample 10 --ingestion 50
uv run pytest
```

Embeddings are local (Ollama BGE-M3), so building the graph needs no API
key. The LLM is an OpenRouter free model (`OPENROUTER_MODEL`, with
`OPENROUTER_FALLBACK_MODEL` as fallback).

- **Without a key,** or when the free tier is rate-limited, everything still
  works. Questions go to the rule-based parser, and explanations are
  templated from the evidence.
- **LLM responses are cached** in `.llm_cache/`, so reruns cost nothing.

## 19. Engineering Notes

A running log of non-obvious decisions and fixes made while building this,
kept here rather than scattered across commit messages:

- **Node-vs-attribute design was tested both ways.** An iteration that made
  every property (including raw floats like density) its own graph node was
  implemented, verified working, then deliberately reverted — see §4's
  design principle. Singleton value-nodes don't add groupability for
  near-unique floats; only their bucketed form does.
- **Flat similarity thresholds don't scale.** The original design used a
  single global cosine cutoff (0.7) for `similar_to` edges. At a few hundred
  materials over only 4 dimensions, that produced tens of thousands of
  "similar enough" pairs — noise, not signal. Switched to top-K nearest
  neighbours per material (K=5), which keeps edge density meaningful
  regardless of dataset size.
- **Cognee's provenance-stamping walk is recursive with no depth cap.**
  Once materials are cross-linked via hundreds of `similar_to` edges, that
  walk exceeds Python's default 1,000-frame recursion limit. Fixed by
  raising `sys.setrecursionlimit(20000)` before invoking the pipeline.
- **The Materials Project `exclude_elements` parameter caps at 60
  characters as a comma-joined string** — silently rejects a full
  radioactive/synthetic element exclusion list if joined naively. Fixed by
  passing it as repeated query parameters instead (`requests` does this
  automatically for a list-valued param).
- **YAML 1.1 boolean coercion bit twice**: `No` (element symbol for
  Nobelium) parses as the boolean `False` unless quoted; `no`/`yes`/`on`/`off`
  are all reserved words. Both instances are now explicitly quoted in
  `materials.yaml`.
- **Cluster query batching bug**: an early version joined every chemical
  system pattern in a cluster into one comma-separated query and applied
  `limit_per_query` to the *combined* result — meaning 16 III-V binary
  systems shared a single limit of 10, starving most of them. Fixed by
  issuing one query per chemical-system pattern, each with its own limit.
- **Robocrystallographer descriptions batch-fetch**: confirmed the
  `/materials/robocrys/` endpoint accepts a comma-separated list of
  `material_ids`, cutting ~794 individual calls down to ~8 batched ones.
- **Elastic/dielectric/oxidation-state/structure data all come from the same
  endpoint already being queried** — widening `_fields=` on the existing
  `/materials/summary/` calls added ~20 new attributes and two new node
  types at zero extra HTTP requests. No new endpoints were needed.
- **Full crystal `structure` is cached but deliberately not a graph node
  property.** It's fetched (needed for Phase 3's CGCNN) and written to
  `data/raw/<material_id>.json`, but never passed into a `Material` node:
  it's a bulky nested lattice/coordinates blob, not something meaningful to
  embed or traverse the way scalar attributes are — and the robocrys
  `description` field already captures the same structural facts in
  embeddable text form, which is what actually serves LLM-facing relational
  reasoning.
- **A 50-material spot-check overestimated elastic/dielectric coverage.**
  An early sample suggested ~84% elastic and ~70% dielectric coverage; the
  real number across all 794 materials is 18.0% and 19.9% respectively (see
  §3). The sample happened to be drawn disproportionately from
  well-characterized battery-cathode compounds fetched first. Lesson: don't
  extrapolate data coverage from an early slice of a clustered fetch order —
  measure against the full set.
- **`similar_to`'s similarity vector was redesigned from 4 to 10 fields**
  once elastic/dielectric/thermodynamic data became available, with each
  inclusion and exclusion justified explicitly (see the comment block above
  `SIMILARITY_FIELDS` in `enrich.py`): intensive/size-normalized fields only
  (excludes `volume`, `nsites`, `num_magnetic_sites`, which are confounded by
  unit-cell choice), fields on a universal scale only (excludes `cbm`/`vbm`/
  `efermi`, referenced to each calculation's own internal zero), and no
  redundant fields (excludes `is_magnetic`, which duplicates
  `total_magnetization`; excludes `e_electronic`/`e_ionic`/refractive index,
  which are derived from/correlated with `dielectric_total`). Missing values
  in the partially-covered fields (elastic, dielectric) are mean-imputed
  rather than dropped, so a material with no elastic data still gets
  meaningful `similar_to` edges from its other 8 dimensions.
- **Raw-dict vs. DataPoint field name mismatch caught before it silently
  broke similarity.** The similarity vector is computed from *raw* MP JSON
  (before `Material` objects exist), where `bulk_modulus`/`shear_modulus`
  are nested `{voigt, reuss, vrh}` dicts and dielectric's raw key is
  `e_total` — not the flattened `bulk_modulus_vrh`/`dielectric_total` names
  used on the `Material` DataPoint. A naive `dict.get(field_name)` would
  have silently returned `None` for every material on these fields. Fixed
  with an explicit `_similarity_value()` extraction shim rather than
  renaming fields to match (the raw JSON shape isn't ours to change).

### Phase 2 engineering notes

- **A cognee 1.3.0 bug had put the same text in every vector collection.**
  - `index_data_points` makes one *shallow* `model_copy()` per index field,
    then mutates the metadata dict the copies share. So `Material_formula`,
    `Material_description`, and `Material_profile` were all embedded from
    the last field, and Phase 1's "formula" and "description" collections
    were identical.
  - Found when all three returned identical scores. Patched from our side
    (`cognee_patches.py`, with a unit test).
  - `verify.py` now fails the build if it recurs. Afterwards, description
    search finds genuine ferrite spinels for "ferrimagnetic spinel oxide".
- **Materials Project ships some unphysical elastic tensors.**
  - 4 of 794 materials have negative moduli or negative universal anisotropy,
    e.g. Li(CuO2)2 with a bulk modulus of −300,604 GPa.
  - They had dominated Phase 1's elastic z-scores: bulk-modulus std was about
    25,000 GPa, and about 52 GPa after the fix.
  - Such tensors are now discarded as a block and flagged in the material's
    profile.
- **Heavy-tailed features need clipped z-scores.**
  - Layered h-BN's universal anisotropy (461.6) is extreme but real. It gave
    z ≈ 184, which made cosine similarity a function of that single field.
  - Z-scores are clipped to ±3.
  - Found by the ingestion leave-one-out check: BN's frozen-scale and rebuild
    neighbours were disjoint.
- **The evaluation's first nDCG definition was wrong.**
  - It gave near-miss materials partial relevance, which put
    constraint-violating materials into the *ideal* ranking. A method
    returning exactly the 2 correct answers scored below one that padded its
    list with violations.
  - Filter-query relevance is now binary; near-misses are measured by the
    violation rate.
- **"≥ 6 of 10 similarity dimensions" was the wrong gate for new materials.**
  - Only about 20% of MP records have elastic or dielectric data, so most
    would have got no `similar_to` edges.
  - The gate is now "all 5 core dimensions present", which every MP record
    meets and a composition-only entry never does.
- **Local embeddings are slow; keep batches small.**
  - Ollama serves one embedding at a time, and cognee runs 4 batches
    concurrently. With the default batch size (36), long robocrys
    descriptions queued past cognee's 60 s timeout.
  - `EMBEDDING_BATCH_SIZE=8` fixes it. A full rebuild takes about 15 min.
- **Free-tier LLM operations.**
  - The OpenRouter key is free-tier: about 50 free-model requests per day,
    and the Qwen pool is often rate-limited upstream.
  - `llm.py` puts a model on a 5-minute cooldown after a 429, falls back to
    the second model, stops cleanly at the daily cap, and caches every
    response on disk.

## 20. Roadmap

- **Phase 3: crystal structure encoding.**
  - Train or reproduce a CGCNN, with a pretrained ALIGNN fallback.
  - Use it for structural embeddings and for property prediction on
    **uploaded CIF structures**. Structure input is deliberately routed there,
    not handled in Phase 2.
  - Add `STRUCTURALLY_SIMILAR` / `SUBSTITUTABLE_WITH` edges.
  - Phase 2's evaluation motivates this directly: property-based
    `similar_to` *hurts* on structural-analogue queries (tier 4, §15).
  - `ingest.py`'s pluggable property source gets a `cgcnn` source, with
    values labelled `predicted`.
- **Neo4j migration.** This is a config swap
  (`GRAPH_DATABASE_PROVIDER=neo4j`). Retrieval depends only on
  `get_graph_data()` and vector search, so no retrieval code changes. It was
  deferred as Phase 2's optional last milestone and not yet done.
- **Deployment.** Next.js frontend, Docker Compose, and GitHub Actions CI/CD,
  per the PRD.

## 21. Tech Stack

| Layer | Technology |
|---|---|
| Data source | [Materials Project REST API](https://api.materialsproject.org) |
| Graph/pipeline framework | [Cognee](https://github.com/topoteretes/cognee) (low-level `DataPoint` pipeline) |
| Graph store | Kuzu (embedded; Neo4j-ready via config swap) |
| Vector store | LanceDB (embedded) |
| Embeddings | BGE-M3 (1024-d) via local [Ollama](https://ollama.com) (no API key) |
| LLM | OpenRouter free models via LiteLLM (`qwen/qwen3.8-27b:free`, fallback `nvidia/nemotron-3-super-120b-a12b:free`) |
| API / dashboard | FastAPI + a static page with Cytoscape.js |
| Element property data | [pymatgen](https://pymatgen.org) (full 103-element periodic table) |
| Language / tooling | Python 3.11–3.14, [`uv`](https://github.com/astral-sh/uv), pytest |
