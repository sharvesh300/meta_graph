# MatGraphRAG — Phase 3 Plan: CGCNN Property Prediction and Structure Ingestion

> **Status (2026-10-22): M0 in progress.** The §9 recommended defaults were
> accepted on 2026-10-22: own PyTorch CGCNN, matgl MEGNet as the fallback
> encoder, a current MP pull for production with Matbench only for the gate,
> opt-in elastic gap-filling after M3, and `structurally_similar` (M6)
> instead of a lower `similar_to` threshold.

Goal: a user supplies a **crystal structure** (CIF / POSCAR, or an MP id whose
structure we already cache). A CGCNN trained on Materials Project predicts its
properties. The material is then **added to the existing knowledge graph as a
new, clearly labelled node**, related to existing nodes through the same
edges Phase 2 ingestion builds. Every predicted number carries the provenance
`predicted` plus the model id and its measured error, so retrieval and the
grounding checker never confuse a prediction with a DFT value.

This is the PRD's `/predict` flow and the `cgcnn` property source promised in
PHASE2_PLAN §6.5. Structural embeddings, `structurally_similar` edges and the
ionic-substitution model are later milestones (M6–M7) that reuse the same
encoder.

---

## 1. The dataset we have (measured 2026-10-22 from `data/raw/`)

| | |
|---|---|
| Records | **795** MP summary records (794 curated plus 1 user-added), 46 distinct elements |
| Clusters | perovskite_oxide 247, oxide_binary 155, battery_cathode_oxide 154, magnetic_spinel_oxide 108, battery_cathode_phosphate 88, semiconductor_ii_vi 17, semiconductor_iii_v 17, semiconductor_group_iv 8 |
| Composition | 4 unary, 193 binary, 510 ternary, 88 quaternary |
| Crystal systems | Monoclinic 224, Orthorhombic 161, Triclinic 109, Trigonal 96, Cubic 88, Tetragonal 87, Hexagonal 30 |
| `nsites` | min 2, median 24, max 198 |
| **`structure`** | **795 / 795**: full pymatgen `Structure` dict (lattice and sites), cached but not in the graph |
| band_gap | 795/795, 0 – 5.87 eV (median 1.16); 231 metals |
| formation_energy_per_atom | 795/795, −4.02 – 0.15 eV/atom (median −2.40) |
| energy_above_hull | 795/795, 0 – 0.148 eV/atom; 365 stable; 462 flagged `theoretical` |
| density / volume / magnetization | 795/795 |
| cbm / vbm | 575/795 |
| Elastic (bulk, shear, anisotropy, Poisson) | **143/795 (18%)**. Includes unphysical values (bulk modulus min −3.0e5 GPa, anisotropy −3.7e4) |
| Dielectric (e_total, e_ionic, e_electronic, n) | **159/795 (20%)** |

What this means for the feature:

- **795 structures are far too few to train a CGCNN.** The published models
  train on ~10⁵ structures. Our graph is the **inference target and the
  real-world test set**, not the training set.
- Structures are already cached for every node, so embeddings and
  predictions for existing nodes need no extra API calls.
- The data is chemically narrow (oxides, phosphates, III-V and II-VI). A
  model trained on all of MP is the right choice; a model trained only on
  our clusters would not generalise to user uploads.
- Elastic data includes garbage values. Any elastic head (optional, §9)
  must filter them, e.g. require K, G > 0 and |anisotropy| < 100.

## 2. Research summary: CGCNN and alternatives

**CGCNN** (Xie & Grossman, PRL 2018; reference code `txie-93/cgcnn`, MIT).
- **Graph:** atoms are nodes. Each node gets a 92-d element feature
  (`atom_init.json`: group, period, electronegativity, covalent radius,
  valence electrons, ionisation energy, electron affinity, …, one-hot binned).
- **Edges:** the 12 nearest neighbours within 8 Å, with periodic images
  allowed, so this is a multigraph. Each distance is expanded in a Gaussian
  basis: 0–8 Å with a 0.2 Å step gives 41 dimensions.
- **Model:** 3 gated convolution layers
  `v_i ← v_i + Σ_j σ(z_ij W_f + b_f) ⊙ softplus(z_ij W_s + b_s)`,
  with batch norm, then mean pooling over atoms and an MLP head. Atom
  feature size is 64 and hidden size is 128, roughly 10⁵ parameters.
- **The embedding is the pooled vector before the head.** That vector is the
  PRD's "crystal embedding".
- **Published accuracy:**

  | Benchmark | Formation energy MAE | Band gap MAE |
  |---|---|---|
  | Original paper (~28k MP) | 0.039 eV/atom | 0.388 eV |
  | Matbench v0.1 `mp_e_form` (132,752 structures), CGCNN v2019 | **0.0337 eV/atom** | — |
  | Matbench v0.1 `mp_gap` (106,113 structures), CGCNN v2019 | — | **0.2972 eV** |

  For reference on the same tasks: ALIGNN scores 0.0215 / 0.1861, MEGNet
  (kgcnn) 0.0252 / 0.1934, and coGN 0.0170 / 0.1559.
- **PRD gate:** formation-energy MAE of about 0.03–0.04 eV/atom. CGCNN's
  published 0.0337 sits inside it, so a faithful reproduction should pass.

**Implementation options**

| Option | Pros | Cons |
|---|---|---|
| **A. Own minimal CGCNN in PyTorch** (recommended) | About 300 lines. Depends only on torch, pymatgen and numpy. Runs on Apple M4 via MPS/CPU without PyG or DGL. Embedding extraction is trivial. Fits the repo's Python 3.13 env. It is exactly the PRD's model. | We own correctness, so it needs the invariance tests and the reproduction gate. |
| B. Vendor `txie-93/cgcnn` | Reference implementation. Ships pre-trained `formation-energy-per-atom.pth.tar`, `band-gap`, etc. | Torch 1.0-era code that reads one CIF file per crystal plus `id_prop.csv`. The data loader is slow, and the pre-trained weights come from a 2018 MP snapshot. **Use its `atom_init.json` and its pre-trained weights as a cross-check only.** |
| C. matgl (MEGNet / M3GNet) | v3.0.0 (May 2026) is PyG-only. Pre-trained `MEGNet-Eform-MP-2018.6.1` and `M3GNet-Eform-MP-2018.6.1` are on Hugging Face (`materialyze`). | It is not CGCNN. Its weights are retrained after a v3 message-passing fix, so check the names on HF first. It adds a PyG dependency. **Recommended fallback encoder.** |
| D. ALIGNN (PRD's named fallback) | Most accurate of the four (0.0215 eV/atom). | Requires DGL pinned to torch 2.1 and recommends Python 3.10. **Incompatible with our Python 3.13 env** unless it runs in a separate venv or container. |

The PRD names ALIGNN as the fallback. Option D's environment conflict is the
reason to swap it for matgl (C); see §9.

## 3. Training data

Two datasets with two different jobs.

1. **Reproduction gate: Matbench `mp_e_form` and `mp_gap`.** Load with
   `matminer.datasets.load_dataset`. Run the official 5-fold CV (or at least
   fold 0) so our MAE is directly comparable with the 0.0337 and 0.2972
   leaderboard values. This is an MP 2018 snapshot with the old energy
   corrections, so it is **only** used to validate the implementation.
2. **Production model: current MP snapshot.** Pull it from the API with the
   same client and correction scheme as the graph (MP2020-compatible
   formation energies). Use paged `/materials/summary/` requests with fields
   `material_id, structure, formation_energy_per_atom, band_gap,
   energy_above_hull, is_metal, nsites` and optional `bulk_modulus,
   shear_modulus` (§9). Filters match Matbench:
   - E_form ≤ 2.5 eV/atom
   - for the gap target, e_hull ≤ 0.15 eV/atom
   - no deprecated entries
   - none of the radioactive elements in `config/materials.yaml →
     exclude_elements`
   - `nsites` ≤ 200

   Cache the result to `data/training/mp_<YYYY.MM>/` (gitignored), compressed
   and sharded.

**Leakage rule.** All 795 graph `material_id`s **and every MP entry sharing a
graph material's reduced formula** go into a fixed **graph-holdout split**.
The formula rule also catches polymorphs. The remaining data splits
80/10/10 into train, validation and test with a fixed seed. Save the split
file next to the model. Reporting MAE on the graph-holdout tells us how
accurate predictions are *for the chemistry this graph covers*. It is the
number the evidence bundle will quote.

**Featurisation cost.** We precompute neighbour lists once with pymatgen
`get_all_neighbors(r=8)`, keep the top 12 per atom, and store
`(nbr_index:int32, distance:float16)` per crystal. That is a few hundred MB
for ~130k crystals. Gaussian expansion happens on the fly in the collate
function, because storing the 41-d edge features would take several GB,
which is too much for 16 GB RAM. We run this with `multiprocessing` and
measure the wall time in M0. Training a model this small should take hours
per run on the M4, not days; M0 measures it before we commit to ensembles.

## 4. Naming conventions

### 4.1 Material ids for new nodes

Today we have `mp-…` for Materials Project records and `user-<reduced
formula>` for composition-only entries. The new form is:

```
struct-<reduced_formula>-sg<spacegroup_number>-<hash8>
e.g.  struct-LiFePO4-sg62-3f9a1c2e
```

- **`struct-`** says the node came from a user-supplied structure. It names
  where the structure came from, not the model, because predictions can be
  re-run with a newer model while the id stays stable.
- **`reduced_formula` and `sg<number>`** make the id readable. The number
  comes from `SpacegroupAnalyzer(symprec=0.1)`, the same tolerance as MP.
- **`hash8`** is the first 8 hex characters of the SHA-256 of the canonical
  structure. To canonicalise, standardise to the conventional cell, sort the
  sites, round fractional coordinates to 1e-3 and the lattice to 1e-3 Å.
  This makes the same crystal uploaded twice in different settings resolve
  to the same id. A `StructureMatcher` check (§5) catches anything the hash
  misses.

### 4.2 Node attributes

| Field | Values | Meaning |
|---|---|---|
| `origin` (exists) | `mp` \| `user` | Who added the node. Structure uploads are `user`. |
| `property_source` (new) | `mp` \| `composition` \| `cgcnn` | Where the property values came from. Mirrors the ingest `PropertySource.name`. |
| `predicted_<property>` (new) | float \| None | e.g. `predicted_formation_energy_per_atom`, `predicted_band_gap`, `predicted_is_metal` |
| `predicted_<property>_std` (new) | float \| None | Ensemble standard deviation (§6) |
| `prediction_model` (new) | str | e.g. `cgcnn-mp2026.10-v1` |

**Predicted values never go into the DFT fields** (`band_gap`,
`formation_energy_per_atom`, …). Those fields stay `None` on a `struct-`
node. Retrieval hard constraints read DFT fields by default and use
`predicted_*` only when the caller opts in (§7). This avoids a quiet failure
in which "band_gap > 2" returns a material whose 2.1 eV is a ±0.3 eV guess.

### 4.3 Model artifacts

```
models/cgcnn/<model_id>/        # gitignored; released as versioned build artifacts (PRD)
  config.yaml                   # hyperparameters, featuriser settings, targets
  member_{0..4}.pt              # ensemble weights
  normalizer.json               # per-target mean/std
  split.json                    # ids per split, incl. graph-holdout
  metrics.json                  # MAE per split, per cluster, per target - quoted in evidence
  atom_init.json                # element features used (copied from txie-93/cgcnn)
model_id = cgcnn-<data tag>-v<n>            e.g. cgcnn-matbench-v1 (gate), cgcnn-mp2026.10-v1 (production)
```

### 4.4 New node and edge types (M6–M7)

New nodes and edges follow the repo's existing style: **PascalCase node types
and snake_case edges**, not the PRD's UPPER_CASE. A mapping table goes in the
README.

| Node | Identity | Description |
|---|---|---|
| `Structure` | `material_id` | Holds `nsites`, the CIF and `crystal_embedding`. |
| `Prototype` | prototype name | e.g. perovskite or spinel |
| `CoordinationEnvironment` | — | e.g. octahedral or tetrahedral |

The existing `OxidationState` node already plays the PRD's `Species` role.

| Edge | Connects |
|---|---|
| `has_structure` | Material → Structure |
| `matches_prototype` | Structure → Prototype |
| `has_coordination` | Structure → CoordinationEnvironment |
| `structurally_similar` | Structure → Structure, weighted by embedding cosine |
| `substitutable_with` | OxidationState → OxidationState, weighted by Hautier probability |

## 5. Ingestion flow (structure in, labelled node out)

This adds `CGCNNSource` to `ingest.py` next to `MPSource` and
`CompositionSource`. The input is a structure, or an MP id with a cached
structure when we want "predict anyway" mode for model checks.

1. **Parse with safety limits.**
   - Size limits: at most 1 MB of input and at most 200 sites.
   - A 10 s parse timeout.
   - Reject partial occupancy or disordered sites (CGCNN needs an ordered
     cell).
   - Reject elements outside `atom_init.json` or in `exclude_elements`.
   - Reject atoms closer than 0.5 Å.
2. **Standardise** with `SpacegroupAnalyzer`, then compute the canonical
   hash and the `struct-…` id.
3. **Deduplicate.**
   - Same hash already in the graph: return the existing node.
   - Otherwise, run `StructureMatcher` against graph materials with the same
     reduced formula, using their cached structures. A match means it is
     already known. In that case return the existing MP node, **plus the
     model's prediction next to the DFT value** as a free sanity check.
   - This replaces the formula-only lookup that `ingest_material` does today
     for this input type. A new polymorph of an existing formula is a
     **different** material and must not collapse onto the MP node.
     `Graph.material_by_formula` keeps preferring the MP node for formula
     lookups.
4. **Predict.** Run the ensemble forward pass and return mean ± std for each
   target, plus the 128-d embedding (stored from M6).
5. **Check out-of-distribution status.** Flag `low_confidence` when any of
   these hold:
   - an element is rare in training (fewer than N occurrences)
   - `nsites` is above the 99th percentile of training
   - the ensemble std is above 2× the test MAE

   The flag is shown, and it never blocks ingestion.
6. **Build the node.** Use `NodeFactory.material(raw, origin="user")`.
   - The raw record is assembled from the structure: elements, chemsys,
     `formula_anonymous`, space group, crystal system, volume, density,
     `nsites`, and oxidation-state guesses from `BVAnalyzer`. These are
     **derived exactly**, not predicted. That gives the node real
     `contains`, `member_of`, `has_space_group`, `has_formula_pattern` and
     `has_oxidation_state` edges straight away.
   - `classified_as` band-gap and metal classes come from the prediction,
     with edge property `provenance: predicted`. A class is assigned only if
     the predicted value is more than 1 MAE from a bucket boundary;
     otherwise the edge is omitted and a note says why.
   - The stability class is **not** assigned, because CGCNN gives E_form,
     not E_hull. An optional step estimates E_hull against the MP phase
     diagram of the chemsys, labelled `derived-from-predicted`.
   - `suitable_for` rules are evaluated on predicted values and marked the
     same way.
7. **Relate.**
   - `similar_to` uses the existing 10-d cosine on the frozen scale, with
     predicted values substituted. That is about 4 of 10 dimensions, below
     `MIN_SIMILARITY_DIMENSIONS = 5`. Decided (§9): keep the threshold, so
     structure nodes get no `similar_to` edges and are related through
     `structurally_similar` from M6.
   - Compositional neighbours work as they do today.
8. **Persist.** Save to `data/user_structures/<material_id>.json`: the
   structure, the predictions and `prediction_model`. Record the material in
   `_user_materials.json` with `source: "cgcnn"`.
   `build_graph.py`'s replay re-adds it **from the stored predictions**, so a
   rebuild is deterministic. `scripts/add_material.py --repredict` refreshes
   them after a model upgrade.

**Interfaces.**
- `scripts/add_material.py --structure path.cif [--persist]`
- `POST /predict` (multipart CIF): predictions only, no graph change, as in
  the PRD.
- `POST /materials/ingest`: now also accepts a CIF upload.
- The dashboard gets a structure upload panel showing values with a
  "predicted" badge.

## 6. Model and training

- **Architecture:** CGCNN as in §2 (3 conv layers, atom features 64, hidden
  128, 12 neighbours, 8 Å, 41 Gaussians). The trunk is shared, with
  **multi-task heads** for E_form (L1 loss), band gap (L1 loss) and
  `is_metal` (BCE), as the PRD specifies. The band-gap head is trained only
  on the e_hull ≤ 0.15 subset.
- **Optimisation:** AdamW with cosine LR, early stopping on validation MAE,
  and targets normalised with the training mean and std.
- **Uncertainty:** a 5-seed ensemble. Matbench Discovery uses a 10-model
  CGCNN ensemble for the same purpose. Report mean ± std.
- **Device:** MPS when available, CPU otherwise. Scatter is avoided because
  neighbours are padded to a fixed 12, so the convolution is plain
  `index_select` + `sum`, which MPS supports.
- **Invariance tests** (PRD: "verified explicitly before training"):
  - a supercell of a structure gives the same prediction
  - a different but equivalent setting gives the same prediction
  - permuting sites gives the same prediction
  - a rigid rotation gives the same prediction

  Tolerance is 1e-4 in tests/test_cgcnn.py.

## 7. Retrieval, evidence and grounding integration

- `query_schema`: add `include_predicted: bool = False`.
  - When it is true, a numeric constraint on `band_gap` falls back to
    `predicted_band_gap`, but only for nodes where the DFT value is `None`.
  - The retrieval path marks that hop `predicted`.
- `evidence.EvidenceFact` gets `provenance` and `model_id`. Predicted facts
  render like this:
  `[E7] struct-LiFePO4-sg62-3f9a1c2e predicted band_gap = 3.4 ± 0.2 eV (cgcnn-mp2026.10-v1, graph-holdout MAE 0.29 eV)`.
- `grounding`: numbers in a claim are matched against predicted facts the
  same way as DFT facts. A claim that states a predicted number **without**
  hedging words ("predicted", "estimated", "~") is flagged.
- `generator` prompt: predicted values must be called predictions.

## 8. Milestones and acceptance criteria

| | Milestone | Done when |
|---|---|---|
| **M0** | Deps and data. Add a `[ml]` extra (`torch`, `matminer`); write `scripts/fetch_training_data.py` (MP API pull, paged and cached) and `scripts/build_crystal_graphs.py` (neighbour lists). | Counts per filter logged. Graph-holdout split written. Featurisation wall time and size measured and recorded in this doc. |
| **M1** | `src/matgraph/cgcnn/` with `featurize.py`, `model.py`, `data.py` (collate and Gaussian expansion) and `train.py` (`scripts/train_cgcnn.py`). | Invariance tests pass. Overfits 100 structures to near-zero loss. |
| **M2** | **Reproduction gate** on Matbench `mp_e_form` fold 0 (fold 0–4 if time allows), plus `mp_gap`. | E_form MAE ≤ 0.040 eV/atom (published 0.0337) and gap MAE ≤ 0.35 eV (published 0.297). **Hard gate:** if it misses, fix the model before M3. Cross-check against txie-93's pre-trained E_form model on the same fold. |
| **M3** | Production ensemble `cgcnn-mp<YYYY.MM>-v1` on the current MP snapshot. | `metrics.json` has MAE on test and graph-holdout, broken down per cluster. Predictions for all 795 graph materials are compared against DFT, and that table goes in the README. |
| **M4** | `CGCNNSource` and structure ingestion (§5), with the naming conventions in §4, CLI, `/predict` and CIF upload on `/materials/ingest`. | Uploading a known MP structure in a different setting dedups onto the MP node and shows prediction vs DFT. A hypothetical polymorph creates a `struct-…` node with derived edges and predicted values. A rebuild replays it identically. Tests cover parse limits, dedup and naming. |
| **M5** | Retrieval, evidence and grounding integration (§7). Dashboard badge. | With `include_predicted=false`, results are byte-identical to Phase 2 on the benchmark. With true, the predicted node is retrievable, and every predicted claim is hedged and grounded. |
| **M6** | Crystal embeddings on `Structure` nodes, `structurally_similar` edges (top-k cosine), `/similar-structures`. Includes the CrystalNN 61-d site-fingerprint baseline (PRD). | Tier-4 (structural analogue) benchmark nDCG measured for embedding vs fingerprint vs the current 0.517, and reported honestly (PRD). |
| **M7** | Hautier ionic substitution (`pymatgen.analysis.structure_prediction`), `substitutable_with` edges, `/similar-elements`. | Held-out compound recovery rate (PRD eval). |

M0–M5 is the feature asked for. M6–M7 complete the PRD's Phase 3 and reuse
the M3 encoder.

## 9. Decisions (accepted 2026-10-22: the recommendation in each item)

1. **Implementation.** Recommended: **own minimal PyTorch CGCNN (option A)**,
   with txie-93's weights as a cross-check.
2. **Fallback encoder.** The PRD says ALIGNN, which is incompatible with
   Python 3.13 because of its DGL pin. Recommended: **matgl MEGNet (option
   C)**.
3. **Training data.** Recommended: **current MP pull for production, and
   Matbench only for the gate**. The alternative, Matbench only, is faster to
   start but leaves the formation energies inconsistent with the graph's.
4. **Gap-filling existing nodes.** Should the model also add
   `predicted_bulk_modulus_vrh` / `predicted_shear_modulus_vrh` to the
   82% of graph materials with no elastic data? CGCNN's paper reports 0.054
   and 0.087 log₁₀ GPa MAE. This needs an elastic head trained on MP's
   smaller elastic set (~10⁴). Phase 2 deliberately did not estimate missing
   values. Recommendation: yes as an opt-in after M3, stored only in
   `predicted_*` fields.
5. **`similar_to` for structure nodes.** Lower `MIN_SIMILARITY_DIMENSIONS`
   for `cgcnn` sources, or rely on `structurally_similar` (M6).
   Recommendation: rely on M6, and use composition neighbours until then.

## 10. Sources

- Xie & Grossman, *Crystal Graph Convolutional Neural Networks…*, PRL 120, 145301 (2018), https://arxiv.org/abs/1710.10324
- Reference code (MIT): https://github.com/txie-93/cgcnn
- Matbench leaderboards: https://matbench.materialsproject.org/ (`mp_e_form`, `mp_gap` per-task pages)
- Matbench / matminer dataset table: https://hackingmaterials.lbl.gov/matminer/dataset_summary.html
- matgl v3 (PyG backend, HF weights): https://github.com/materialyzeai/matgl
- ALIGNN install requirements (DGL, torch 2.1): https://github.com/usnistgov/alignn
- Matbench Discovery (CGCNN ensemble): https://arxiv.org/abs/2308.14920
- PRD: `~/Documents/MatGraphRAG_Updated.pages`, sections "Crystal Structure Graph Representation", "Structure Aware Retrieval", Phase 3
