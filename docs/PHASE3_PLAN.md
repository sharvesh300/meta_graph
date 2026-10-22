# MatGraphRAG — Phase 3 Plan: CGCNN Inference and Structure Ingestion

> **Status (2026-10-22): plan revised, nothing implemented.** The reference
> is [`txie-93/cgcnn`](https://github.com/txie-93/cgcnn) at commit `f42ab23`
> (MIT). Scope is **inference** with its published pre-trained models, wired
> into the existing Phase 2 ingestion feature. Training our own CGCNN is no
> longer in the main scope; it moves to the optional M6.

Goal: a user supplies a **crystal structure** (CIF / POSCAR file, or an MP id
whose structure we already cache). The pre-trained CGCNN models predict its
properties. The material is then **added to the existing knowledge graph as a
new, clearly named node**, related to existing nodes through the same edges
Phase 2 ingestion builds (`scripts/add_material.py`, `POST /materials/ingest`).
Every predicted number carries the provenance `predicted`, the model id and
its measured error. That way retrieval and the grounding checker never
confuse a prediction with a DFT value.

The same inference also **completes the existing graph**. 82% of the current
nodes have no elastic data, and CGCNN ships bulk/shear/Poisson models. Those
predictions are added as separately labelled `predicted_*` attributes (M4).

---

## 1. The dataset we have (measured 2026-10-22 from `data/raw/`)

| | |
|---|---|
| Records | **795** MP summary records (794 curated plus 1 user-added), 46 distinct elements. IDs are MP's new format (`mp-aaaaaabg`); legacy ids still resolve through the API. |
| Clusters | perovskite_oxide 247, oxide_binary 155, battery_cathode_oxide 154, magnetic_spinel_oxide 108, battery_cathode_phosphate 88, semiconductor_ii_vi 17, semiconductor_iii_v 17, semiconductor_group_iv 8 |
| Composition | 4 unary, 193 binary, 510 ternary, 88 quaternary |
| Crystal systems | Monoclinic 224, Orthorhombic 161, Triclinic 109, Trigonal 96, Cubic 88, Tetragonal 87, Hexagonal 30 |
| `nsites` | min 2, median 24, max 198 |
| **`structure`** | **795 / 795**: full pymatgen `Structure` dict, cached on disk but not in the graph |
| band_gap / E_form / E_hull | 795/795. Gap 0–5.87 eV (231 metals); E_form −4.02–0.15 eV/atom; 365 stable |
| efermi | 795/795 |
| Elastic (bulk, shear, anisotropy, Poisson) | **143/795 (18%)**. Includes unphysical values (bulk modulus min −3.0e5 GPa) |
| Dielectric | **159/795 (20%)**. No CGCNN model exists for it, so it stays `unavailable`. |

Every node already has its structure, so we can run inference on the whole
graph without any API call. The graph is also the **evaluation set** for the
pre-trained models (§2.4).

## 2. The reference repository: `txie-93/cgcnn`

### 2.1 What it contains

The repo was last updated 2021-09-06 (commit `f42ab23`). It has about 1,350
lines of Python:

| File | Contents |
|---|---|
| `cgcnn/model.py` (187 lines) | `ConvLayer` and `CrystalGraphConvNet` |
| `cgcnn/data.py` (352 lines) | `GaussianDistance`, `AtomCustomJSONInitializer` and `CIFData` |
| `main.py` | Training |
| `predict.py` | Inference over a directory of CIFs plus `id_prop.csv` |
| `data/sample-*/atom_init.json` | 92-d element features for Z = 1–100 |
| `pre-trained/*.pth.tar` | 8 models, about 0.8 MB each |
| `data/material-data/mp-ids-*.csv` | The legacy MP ids used in the paper (46,744 / 27,430 / 3,402) |

### 2.2 How the crystal graph is built (`CIFData.__getitem__`)

For each site:
1. Take the 92-d `atom_init.json` vector for its atomic number.
2. Run `structure.get_all_neighbors(r=8, include_index=True)` and sort by
   distance.
3. Keep the **12 nearest neighbours**. If fewer than 12 are found, pad with
   index 0 and distance r + 1 = 9 Å, and emit a warning.
4. Expand each distance with Gaussians over `arange(0, 8 + 0.2, 0.2)`, which
   gives **41 dimensions**, with var = 0.2.

The model takes `atom_fea (N, 92)`, `nbr_fea (N, 12, 41)`, `nbr_fea_idx (N,
12)` and `crystal_atom_idx` (the atoms belonging to each crystal).

### 2.3 Model (`CrystalGraphConvNet`)

The layers run in this order:
1. `embedding` (92→64)
2. n_conv gated `ConvLayer`s, each computing
   `softplus(v_i + BN(Σ_j σ(BN(W[v_i‖v_j‖u_ij])_filter) ⊙ softplus(…_core)))`
3. **mean pooling** over atoms
4. softplus → `conv_to_fc` (64→h) → softplus
5. `fc_out` (h→1, or h→2 with LogSoftmax for classification)

A regression output is denormalised with the checkpoint's `normalizer`
(mean, std).

**The pre-trained models do not use the repo defaults.** Their checkpoint
`args` say `n_conv=4`, `atom_fea_len=64`, `h_fea_len=32`, `n_h=1`. The
architecture must therefore be built **from the checkpoint args**, never from
defaults. We verified this from the tensor shapes: `convs.0..3`,
`fc_full (128, 169)` where 169 = 2·64 + 41, `conv_to_fc (32, 64)` and
`fc_out (1, 32)`.

### 2.4 Pre-trained models

We read these values from each checkpoint's metadata with a restricted
unpickler that runs no code. All of them were trained on an **MP snapshot of
2017-04-19**.

| Checkpoint | Our attribute | Units | Train size | Best **validation** score | Epoch |
|---|---|---|---|---|---|
| `formation-energy-per-atom` | `predicted_formation_energy_per_atom` | eV/atom | 28,046 | MAE **0.0397** | 968 |
| `band-gap` | `predicted_band_gap` | eV | 16,458 (`20170419_semi.json`) | MAE **0.369** | 457 |
| `semi-metal-classification` | `predicted_is_metal` | metal = 1 | 37,394 | AUC **0.959** | 92 |
| `efermi` | `predicted_efermi` | eV | 28,046 | MAE 0.355 | 804 |
| `bulk-moduli` | `predicted_bulk_modulus_vrh` | log(GPa) | 2,041 | MAE 0.0553 | 759 |
| `shear-moduli` | `predicted_shear_modulus_vrh` | log(GPa) | 2,041 | MAE 0.0908 | 118 |
| `poisson-ratio` | `predicted_homogeneous_poisson` | — | 2,041 | MAE 0.0293 | 66 |
| `final-energy-per-atom` | *not used*: raw uncorrected total energy, not in our schema | eV/atom | 28,046 | MAE 0.079 | 953 |

Things the code and checkpoints imply, which M1 has to verify:

- **The band-gap model was trained only on non-metals.** It used the "semi"
  file, which has about 16k entries against 28k for formation energy. So
  `predicted_band_gap` is gated by the classifier:
  - if P(metal) ≥ 0.5, the gap is 0
  - otherwise it is the regressor output, clipped at 0

  M1 checks this on the 231 graph metals.
- **The log base for the moduli is not documented.** We settle it
  empirically on the 143 graph materials that have elastic data. Their median
  bulk modulus is 106 GPa, which is 2.03 in log₁₀ and 4.66 in ln, so the two
  are easy to tell apart. After that we store values in GPa.
- **Data drift.** The 2017 MP formation energies use the legacy
  correction scheme, while the graph uses current MP2020-compatible values.
  Most of the graph is transition-metal oxides, where the corrections
  changed, so expect a **systematic offset**. We measure it and report it,
  and we do not hide it with a fitted correction.
- **Contamination.** Many graph materials were probably in CGCNN's training
  data. M1 maps the 46,744 legacy ids to new ids through the API, then
  reports graph MAE separately for *seen in training* and *unseen*. The
  unseen number is the honest one, and it is the one quoted in evidence.
- **Validation MAE is not test MAE.** We quote our own graph-measured MAE,
  not the checkpoint's `best_mae_error`.
- **Loading.** The checkpoints are in the legacy (pre-1.6) torch format.
  Modern torch's `weights_only=True` default rejects the numpy scalar in the
  classifier's metadata. A one-time **conversion** step loads them with an
  allowlist and re-saves each model as a plain `state_dict` plus `meta.json`.
  After that, runtime loading never unpickles anything.

### 2.5 What we reuse and what we change

| Reference | In this repo |
|---|---|
| `model.py` | **Vendored almost verbatim** as `src/matgraph/cgcnn/model.py`, with the MIT notice and commit hash. Weight compatibility depends on keeping parameter names identical. The changes are: drop `Variable`; vectorise `pooling` with `index_add_` (results identical, tested); add `forward(..., return_embedding=True)` returning the 64-d pooled vector, which M5 uses. |
| `data.py` featurisation | Re-implemented as `featurize(structure: Structure)` in `src/matgraph/cgcnn/featurize.py`. It takes pymatgen objects instead of files, so there is no `id_prop.csv` or CIF directory, and it uses the same r, max_nbr, Gaussians and padding. The repo's `atom_init.json` is copied in. |
| `predict.py` | Replaced by `CGCNNPredictor`. It loads all 7 models once (as the PRD requires for the API), featurises a structure **once** and shares it across models, and returns a typed `Prediction` per property. No CSV output. |
| `main.py` (training) | Not used. It becomes the reference for optional retraining (M6). |
| `sklearn` dependency | Not needed, because we don't compute metrics at inference time. |

Dependencies: **only `torch`**, in a new `[ml]` optional extra. pymatgen and
numpy are already installed. Inference on ≤200 sites should take well under
a second on CPU (measured in M0), so MPS is not needed. The weights come from
`scripts/fetch_cgcnn_weights.py`: it downloads the 8 checkpoints pinned to
commit `f42ab23`, checks their sha256, converts them, and writes them to
`models/cgcnn/txie93-f42ab23/` (gitignored, so the weights are build
artifacts as the PRD asks). Committing the converted 5.6 MB instead is a
§9 decision.

## 3. Inference module

```
src/matgraph/cgcnn/
  __init__.py
  model.py        # vendored CrystalGraphConvNet (+ return_embedding)
  featurize.py    # Structure -> (atom_fea, nbr_fea, nbr_fea_idx); atom_init.json
  predictor.py    # CGCNNPredictor: load models/meta, predict(structure) -> Predictions
  validate.py     # parse limits + standardisation for user structures
  atom_init.json  # from txie-93/cgcnn data/sample-regression
models/cgcnn/txie93-f42ab23/<property>/{state_dict.pt, meta.json}
  meta.json = {property, units, task, args, normalizer, val_score, source_commit, sha256, graph_eval}
```

`Prediction` holds:
- `property` and `value`
- `units`
- `model_id`, e.g. `cgcnn-txie93-f42ab23/formation-energy-per-atom`
- `expected_error`: our graph-measured MAE on unseen materials
- `flags`, e.g. `out_of_distribution`, `few_neighbours`, `near_bucket_boundary`

## 4. Naming conventions

### 4.1 Material ids for new nodes

Today we have `mp-…` for Materials Project records and `user-<reduced
formula>` for composition-only entries. The new form is:

```
struct-<reduced_formula>-sg<spacegroup_number>-<hash8>
e.g.  struct-LiFePO4-sg62-3f9a1c2e
```

- **`struct-`** marks a node built from a user-supplied structure. It names
  where the structure came from, not the model, so the id stays stable if the
  predictions are re-run with a different model.
- **`sg<number>`** comes from `SpacegroupAnalyzer(symprec=0.1)`, the same
  tolerance as MP.
- **`hash8`** is the first 8 hex characters of the SHA-256 of the canonical
  structure. To canonicalise, take the conventional standard cell, sort the
  sites, and round fractional coordinates and the lattice to 1e-3. That way
  the same crystal uploaded in different settings gets one id.
  `StructureMatcher` (§5) catches what the hash misses.

### 4.2 Attributes

| Field | Values | Meaning |
|---|---|---|
| `origin` (exists) | `mp` \| `user` | Who added the node. Structure uploads are `user`. |
| `property_source` (new) | `mp` \| `composition` \| `cgcnn` | Where the main property values came from. Mirrors `PropertySource.name`. |
| `predicted_<property>` (new) | float/bool \| None | One per model in §2.4, e.g. `predicted_band_gap` |
| `prediction_model` (new) | str | e.g. `cgcnn-txie93-f42ab23` |

**Predictions never go into the DFT fields.** On a `struct-` node, `band_gap`,
`formation_energy_per_atom` and the other DFT fields stay `None`. On existing
MP nodes, the DFT fields are untouched and `predicted_*` sits next to them.
Retrieval reads DFT fields unless the caller opts in (§7).

The `profile` text gets one sentence per predicted value, for example "CGCNN
predicts a band gap of about 2.1 eV (expected error ±0.4 eV)". This keeps
semantic search and the Vector RAG baseline honest about it.

### 4.3 Provenance labels (extends Phase 2's `mp | derived | guessed | unavailable`)

| Label | Used for |
|---|---|
| `predicted` | CGCNN outputs |
| `derived` | Exact values computed from the uploaded structure: formula, chemsys, `formula_anonymous`, space group, crystal system, volume, density, nsites |
| `guessed` | Oxidation states from `BVAnalyzer` / `oxi_state_guesses` |

### 4.4 Later node and edge types (M5+)

These follow the repo's style: PascalCase node types and snake_case edges,
not the PRD's UPPER_CASE. The README will carry a mapping table.

| New | Kind |
|---|---|
| `Structure` | Node, holding the CIF, nsites and `crystal_embedding` |
| `Prototype` | Node |
| `has_structure` | Edge |
| `matches_prototype` | Edge |
| `structurally_similar` | Edge |
| `substitutable_with` | Edge |

## 5. Integration with the existing ingestion feature

`ingest.py` gains `CGCNNSource` next to `MPSource` and `CompositionSource`.
The rest of the pipeline stays as it is: `NodeFactory`, `RelationReport`, the
overlay preview, `--persist`, `_user_materials.json` replay, and
`explain_relation` with the grounding checker.

1. **Parse with safety limits** (`validate.py`). Reject the input when any
   of these hold:
   - the file is over 1 MB, or the parse takes more than 10 s
   - it has more than 200 sites (the graph's max is 198)
   - it has partial occupancy, which CGCNN needs ordered sites for
   - an element is outside `atom_init.json` or in `exclude_elements`
   - two atoms are closer than 0.5 Å
2. **Standardise and name.** Run `SpacegroupAnalyzer`, then build the
   `struct-…` id (§4.1).
3. **Deduplicate.** These checks replace the formula-only lookup that
   `ingest_material` does today, for structure input only.
   - If the hash already exists in the graph, return that node.
   - Otherwise, run `StructureMatcher` against graph materials with the same
     reduced formula, using their cached structures. If one matches, return
     the **existing MP node**, with the CGCNN prediction shown next to its
     DFT value as a free model check.
   - If nothing matches, the structure is a new polymorph and becomes a new
     node, even when its formula already exists. `Graph.material_by_formula`
     keeps resolving a formula to the MP node.
4. **Predict.** `CGCNNPredictor.predict(structure)` runs all models on one
   featurisation.
5. **Build the raw record**, shaped like an MP summary so the factory works
   unchanged.
   - Derived fields: `formula_pretty`, `elements`, `chemsys`,
     `formula_anonymous`, `symmetry`, `volume`, `density`, `nsites`, and
     `possible_species` (guessed).
   - `predicted_*` values are added alongside.
   - The DFT keys are left out.

   The node then gets real `contains`, `member_of`, `has_space_group`,
   `has_formula_pattern` and `has_oxidation_state` edges, exactly as an MP
   material does.
6. **Predicted classes and rules.** `NodeFactory.classifications` gets a
   predicted path.
   - **band_gap class:** assigned from `predicted_band_gap` only when the
     value is more than one expected-error away from a bucket boundary.
     Otherwise the edge is omitted and a note explains why.
   - **stability class:** not assigned. CGCNN predicts formation energy, not
     energy above hull. An optional step estimates E_hull against the MP
     phase diagram of the chemsys, labelled `derived-from-predicted`.
   - **application rules:** evaluated on predicted values and reported in
     `RelationReport.rules` as `predicted`.
   - **edge marking:** these `classified_as` and `suitable_for` edges carry
     `properties={"provenance": "predicted", "model": …}`.
7. **Relate.** Compositional neighbours work as today. `similar_to` needs 5
   DFT dimensions, so structure nodes get none. This keeps the accepted
   decision: the structural relation comes from `structurally_similar` in
   M5.
8. **Persist.** Save to `data/user_structures/<material_id>.json`: the
   structure, the predictions and the `prediction_model`. Add an entry to
   `_user_materials.json` with `source: "cgcnn"`. `build_graph.py` replays
   from the stored predictions, so a rebuild is deterministic.
   `add_material.py --repredict` refreshes them after a weights change.

**Interfaces**

| Interface | Change |
|---|---|
| `scripts/add_material.py --structure path.cif [--persist] [--explain]` | Structure input |
| `scripts/predict.py path.cif …` | Prints predictions only. Replaces the reference `predict.py`. |
| `POST /predict` | Multipart CIF upload. Returns predictions only, with no graph change (PRD). |
| `POST /materials/ingest` | Also accepts a CIF upload. The 422 message "crystal structures go to CGCNN in Phase 3" goes away. |
| `GET /health` | Adds `cgcnn_loaded` and runs a fixed-structure smoke test (PRD) |
| Dashboard | Structure upload panel, with a "predicted" badge and expected error on each value |

## 6. Completing existing nodes (gap-filling)

`scripts/predict_graph.py` runs the predictor over all 795 cached structures.
It writes a `predicted_*` attribute for every model on every node, with DFT
fields untouched, plus `prediction_model`. The batch is expected to take minutes on CPU
(measured in M1). It is written back through `pipeline.add_to_graph`, which upserts by
`material_id`. It is also wired into `build_graph.py` behind
`--with-predictions`, so a rebuild reproduces it.

This is what fills the 82% elastic gap with labelled predictions. Retrieval
does not use them unless asked (§7).

The same run produces the **graph evaluation** (M1), because every node also
has DFT values to compare against.

## 7. Retrieval, evidence and grounding

- **`query_schema`.** Add `include_predicted: bool = False`.
  - When it is true, a numeric constraint falls back to `predicted_<prop>`,
    but only where the DFT value is `None`.
  - The hop is marked `predicted` in the reasoning path.
  - With it false, results are unchanged from Phase 2. The benchmark proves
    this.
- **`EvidenceFact`.** Add `provenance` and `model_id`. A predicted fact
  renders like this:
  `[E7] struct-LiFePO4-sg62-3f9a1c2e predicted band_gap ≈ 2.1 eV (CGCNN txie93 band-gap, expected error ±0.4 eV on unseen graph materials)`.
- **Grounding.** A claim that states a predicted number without hedging
  ("predicted", "estimated", "≈") is flagged. Predicted numbers are matched
  within their expected error, not the DFT tolerance.

## 8. Milestones and acceptance criteria

| | Milestone | Done when |
|---|---|---|
| **M0** | Add `torch` in an `[ml]` extra. `scripts/fetch_cgcnn_weights.py` handles download, sha256 check and conversion. Vendor `model.py` and write `featurize.py` and `predictor.py`. | **Parity test:** for the repo's 10 `sample-regression` CIFs, our predictor matches the reference `predict.py` (run once against the vendored code) to 1e-5. **Invariance tests:** a supercell, an equivalent setting, permuted sites and a rotated cell all give the same prediction within 1e-4. Everything runs offline once the weights are fetched. |
| **M1** | Graph evaluation via `scripts/predict_graph.py --eval-only`. Map the legacy ids to find seen versus unseen materials. | `artifacts/eval/cgcnn_graph_eval.md` reports, per model and per cluster, MAE, bias and seen/unseen splits against the 795 DFT values. The moduli log base and the band-gap gating are settled. `meta.json.graph_eval` holds the expected errors that evidence will quote. |
| **M2** | `CGCNNSource` and structure ingestion (§5), with naming, validation, dedup, persistence and replay. CLI `--structure` and `scripts/predict.py`. | A known MP structure in a different setting dedups onto its MP node and shows prediction vs DFT. A new polymorph creates a `struct-…` node with derived edges, predicted values and predicted classes. A rebuild replays it identically. Tests cover the parse limits, naming and hash stability, and dedup. |
| **M3** | API and dashboard: `/predict`, CIF on `/materials/ingest`, `/health`, the upload panel. Retrieval, evidence and grounding integration (§7). | With `include_predicted=false` the benchmark is identical to Phase 2. With true, struct nodes are retrievable, and every predicted claim is hedged and grounded. API tests stay offline. |
| **M4** | Gap-fill existing nodes (§6). | Every node has `predicted_*` attributes. The README documents coverage of DFT vs predicted per property. |
| **M5** | (PRD Phase 3 continuation) Crystal embedding: the 64-d pooled vector from the formation-energy model, on a `Structure` node. Add `structurally_similar` edges and `/similar-structures`, plus the CrystalNN site-fingerprint baseline. | Tier-4 nDCG for embedding vs fingerprint vs the current 0.517, reported honestly. |
| **M6** | (Optional) Retrain on the current MP snapshot with the reference `main.py` recipe, to remove the 2017 drift. Gate on Matbench `mp_e_form` (published CGCNN 0.0337 eV/atom). Then Hautier substitution, `substitutable_with` and `/similar-elements` (PRD). | Done only if M1 shows the drift matters. |

M0–M4 is the requested feature: complete CGCNN inference wired into the
existing ingestion and the existing nodes.

## 9. Decisions

Accepted 2026-10-22. They carry over unchanged:
- **Own code over a vendored pipeline.** The model is vendored; the
  featuriser and predictor are ours.
- **Opt-in predicted values.** They are used only when the caller asks.
- **No `similar_to` for structure nodes.** Their structural relation waits
  for M5.

Changed by this revision:
- **Training.** Was "own training is the main path". Now **inference with
  the txie-93 pre-trained models is the main path**, and retraining is the
  optional M6.
- **Fallback encoder.** Was matgl MEGNet. It is now only needed if M1 shows
  CGCNN is unusable on this chemistry.

Still open:
1. **Weights in git?** The default is gitignored, with a fetch script and
   sha256 pins. The alternative is to commit the 5.6 MB of converted weights
   so the repo works offline from a clone.
2. **Gap-filling scope.** The default is all 7 models on every node. The
   alternative is elastic and Poisson only, where DFT coverage is 18%.
   Formation energy, band gap and efermi are 100% DFT-covered, so those
   predictions only serve evaluation.

## 10. Sources

- Reference code and pre-trained models (MIT): https://github.com/txie-93/cgcnn, commit `f42ab23`. The checkpoint metadata in §2.4 was read from the files themselves.
- Xie & Grossman, *Crystal Graph Convolutional Neural Networks…*, PRL 120, 145301 (2018), https://arxiv.org/abs/1710.10324
- Matbench `mp_e_form` / `mp_gap` leaderboards (CGCNN 0.0337 eV/atom, 0.2972 eV): https://matbench.materialsproject.org/
- matgl v3 (possible fallback): https://github.com/materialyzeai/matgl
- PRD: `~/Documents/MatGraphRAG_Updated.pages`, sections "Crystal Structure Graph Representation", "Structure Aware Retrieval", Phase 3
