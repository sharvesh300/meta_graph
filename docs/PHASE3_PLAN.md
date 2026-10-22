# MatGraphRAG — Phase 3 Plan: CGCNN Inference and Structure Ingestion

> **Status (2026-10-22): M0–M3 done (results in §8.1–8.4); M4 next.** The reference
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

Things the code and checkpoints imply. M1 has verified all of them; the
outcomes are in §8.2.

- **The band-gap model was trained only on non-metals.** It used the "semi"
  file, which has about 16k entries against 28k for formation energy. So
  `predicted_band_gap` is gated by the classifier:
  - if P(metal) ≥ the threshold, the gap is 0
  - otherwise it is the regressor output, clipped at 0

  M1 confirmed this on the 231 graph metals and set the threshold to
  **0.3**, chosen on the seen split.
- **The log base for the moduli is not documented.** We settle it
  empirically on the 143 graph materials that have elastic data. Their median
  bulk modulus is 106 GPa, which is 2.03 in log₁₀ and 4.66 in ln, so the two
  are easy to tell apart. After that we store values in GPa.
  M1 settled it as **log₁₀**.
- **Data drift.** The 2017 MP formation energies use the legacy
  correction scheme, while the graph uses current MP2020-compatible values.
  Most of the graph is transition-metal oxides, where the corrections
  changed, so expect a **systematic offset**. We measure it and report it,
  and we do not hide it with a fitted correction. M1 found it small
  overall: −0.010 eV/atom on unseen materials. It is larger per cluster,
  e.g. +0.12 for III-V.
- **Contamination.** Many graph materials were probably in CGCNN's training
  data. M1 maps the legacy ids to new ids. No API call is needed, because
  the new id is the legacy number in base 26. It then
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
numpy are already installed. Inference runs on CPU and is fast enough that
MPS is not needed: 62 ms for the largest graph structure (198 sites), as
measured in M0. The weights come from
`scripts/fetch_cgcnn_weights.py`: it downloads the 7 used checkpoints pinned to
commit `f42ab23`, checks their sha256, converts them, and writes them to
`models/cgcnn/txie93-f42ab23/` (gitignored, so the weights are build
artifacts as the PRD asks). Committing the converted 2.8 MB instead is a
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
  meta.json = {property, units, task, args, normalizer, best_validation_score, epoch, source}
  training_ids/mp-ids-*.csv   # the reference repo's dataset id lists (seen/unseen split)
src/matgraph/cgcnn/interpret.py  # raw outputs -> graph values: gating, GPa, expected errors (M1)
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
   - the file is over 1 MB (with the site cap, this bounds parse time, so
     there is no separate timeout)
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
8. **Persist.** Save to `data/raw/user_structures/<material_id>.json`: the
   structure, the predictions and the `prediction_model`. Add an entry to
   `_user_materials.json` with `source: "cgcnn"`. `build_graph.py` replays
   from the stored predictions, so a rebuild is deterministic. A
   `--repredict` option to refresh them after a weights change is not built
   yet; it is only needed once a second model set exists.

**Interfaces**

| Interface | Change |
|---|---|
| `scripts/add_material.py --structure path.cif [--persist] [--explain]` | Structure input |
| `scripts/predict.py path.cif …` | Prints predictions only. Replaces the reference `predict.py`. |
| `POST /predict` | CIF/POSCAR text in the JSON body (`structure`, `format`, `filename`). Returns predictions only, with no graph change (PRD). JSON rather than multipart, like the other endpoints; `python-multipart` is only a transitive dependency here. |
| `POST /materials/ingest` | Also accepts `structure` text, routed to `ingest_structure` |
| `GET /health` | Adds `cgcnn`: available, model set, and a fixed-structure smoke test against the reference implementation's outputs (PRD) |
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

- **`include_predicted: bool = False`.** This is a request option on
  `retrieve()`, `answer_question()` and the `/query` body. It is *not* a
  `QueryConstraints` field, because that model is the LLM parser's output
  schema and the LLM must not switch predictions on.
  - **When it is false,** retrieval sees exactly the Phase 2 graph:
    - no `struct-` materials;
    - no edges marked `provenance: predicted`;
    - no `predicted_*` values;
    - ranking statistics (hub IDF, property spreads) over the same visible
      materials.
  - **When it is true,** a numeric or boolean constraint falls back to
    `predicted_<prop>`, but only where the DFT value is `None`. The check,
    the hop and the candidate are marked `predicted`.
- **`EvidenceFact.predicted`.** A predicted fact renders like this:
  `[E19] MgO predicted band_gap ≈ 3.9897 eV (CGCNN cgcnn-txie93-f42ab23, not a DFT value; expected error ±0.617 eV on unseen graph materials)`.
  - Moduli state their error as a factor.
  - Edges resting on a prediction say so.
  - A caveat fact explains what predicted means.
- **Grounding.** Numbers are matched exactly, as for DFT facts. Matching
  "within the expected error", as first planned, would have let a claim
  state a different number. A claim that rests on a prediction must hedge
  ("predicted", "estimated", "≈", "CGCNN", …) or it is flagged. A claim rests
  on a prediction when either holds:
  - a number in it is supported only by a predicted fact;
  - it cites nothing but predicted facts.
- **LLM prompt.** The rule "call predictions predicted and give their
  error" is added only when the evidence holds predicted facts. Prompts
  (and the LLM cache keys) for DFT-only evidence stay byte-identical to
  Phase 2.

## 8. Milestones and acceptance criteria

| | Milestone | Done when |
|---|---|---|
| **M0** | Add `torch` in an `[ml]` extra. `scripts/fetch_cgcnn_weights.py` handles download, sha256 check and conversion. Vendor `model.py` and write `featurize.py` and `predictor.py`. | **Parity test:** for the repo's 10 `sample-regression` CIFs, our predictor matches the reference `predict.py` (run once against the vendored code) to 1e-5. **Invariance tests:** a supercell, an equivalent setting, permuted sites and a rotated cell all give the same prediction within 1e-4. Everything runs offline once the weights are fetched. |
| **M1** ✓ | Graph evaluation via `scripts/evaluate_cgcnn.py`. Map the legacy ids to find seen versus unseen materials. | `artifacts/eval/cgcnn_graph_eval.md` reports, per model and per cluster, MAE, bias and seen/unseen splits against the 795 DFT values. The moduli log base and the band-gap gating are settled in `cgcnn/interpret.py`. `artifacts/eval/cgcnn_graph_eval.json → expected_errors` (committed) holds the expected errors that evidence will quote. |
| **M2** ✓ | `CGCNNSource` and structure ingestion (§5), with naming, validation, dedup, persistence and replay. CLI `--structure` and `scripts/predict.py`. | A known MP structure in a different setting dedups onto its MP node and shows prediction vs DFT. A new polymorph creates a `struct-…` node with derived edges, predicted values and predicted classes. A rebuild replays it identically. Tests cover the parse limits, naming and hash stability, and dedup. |
| **M3** ✓ | API and dashboard: `/predict`, CIF on `/materials/ingest`, `/health`, the upload panel. Retrieval, evidence and grounding integration (§7). | With `include_predicted=false` the benchmark is identical to Phase 2. With true, struct nodes are retrievable, and every predicted claim is hedged and grounded. API tests stay offline. |
| **M4** | Gap-fill existing nodes (§6). | Every node has `predicted_*` attributes. The README documents coverage of DFT vs predicted per property. |
| **M5** | (PRD Phase 3 continuation) Crystal embedding: the 64-d pooled vector from the formation-energy model, on a `Structure` node. Add `structurally_similar` edges and `/similar-structures`, plus the CrystalNN site-fingerprint baseline. | Tier-4 nDCG for embedding vs fingerprint vs the current 0.517, reported honestly. |
| **M6** | (Optional) Retrain on the current MP snapshot with the reference `main.py` recipe, to remove the 2017 drift. Gate on Matbench `mp_e_form` (published CGCNN 0.0337 eV/atom). Then Hautier substitution, `substitutable_with` and `/similar-elements` (PRD). | Done only if M1 shows the drift matters. |

### 8.1 M0 results (2026-10-22)

- **Code and data added:**
  - `[ml]` extra with `torch>=2.14.1`
  - `src/matgraph/cgcnn/`: `registry.py` (pinned commit, sha256 per
    checkpoint), the vendored `model.py`, `featurize.py`, `predictor.py`, and
    `atom_init.json`
  - `scripts/fetch_cgcnn_weights.py`
  - `scripts/make_cgcnn_reference_fixture.py`
  - `tests/test_cgcnn.py`, with the 10 reference sample CIFs and their
    reference outputs as fixtures
- **Weights.** All 7 checkpoints downloaded at `f42ab23` and matched their
  pinned sha256. They are converted to plain `state_dict` files plus
  `meta.json`, 2.8 MB in total because there is no optimizer state. Six of
  them load with `weights_only=True` as they are. The classifier also needs
  an allowlist for `numpy.core.multiarray.scalar`, the NumPy 1.x name in its
  pickle.
- **Parity.** On the reference repo's 10 sample CIFs, our featuriser and
  predictor match the reference `CIFData` + `collate_pool` +
  `CrystalGraphConvNet` with the original checkpoints. The largest absolute
  difference across all 7 models is **4.8e-7**.
- **Invariance.** Predictions and the 64-d pooled embedding are unchanged
  within 1e-4 for a supercell, permuted sites, a rotated cell, and the
  primitive and conventional settings. We checked this on NaCl and on COD
  1000041. A random-init model already separates structures by about 0.03,
  far above that tolerance, so the tests do discriminate.
- **Speed on the 795 graph structures (CPU).** Featurising takes 8.8 s and
  running all 7 models takes 1.8 s. No structure needed neighbour padding.
- **Test suite.** 59 tests pass, with no network calls. The parity tests
  skip when the weights have not been fetched.

### 8.2 M1 results (2026-10-22)

Full report: [`artifacts/eval/cgcnn_graph_eval.md`](../artifacts/eval/cgcnn_graph_eval.md).

| Attribute | Expected error (unseen graph materials) | MAE in dataset | Checkpoint val. |
|---|---|---|---|
| `predicted_formation_energy_per_atom` | **0.063 eV/atom** (n=389), bias −0.010 | 0.044 | 0.040 |
| `predicted_band_gap` (gated at P(metal) ≥ 0.3) | **0.62 eV** (n=389) | 0.41 | 0.37 (non-metals only) |
| `predicted_is_metal` | **27% error rate** (AUC 0.79, n=389) | AUC 0.91 | AUC 0.96 |
| `predicted_efermi` | **0.69 eV** (n=389) | 0.40 | 0.36 |
| `predicted_bulk_modulus_vrh` | **0.097 log₁₀ GPa**, about ×1.25 (n=55) | 0.046 | 0.055 |
| `predicted_shear_modulus_vrh` | **0.100 log₁₀ GPa**, about ×1.26 (n=55) | 0.060 | 0.091 |
| `predicted_homogeneous_poisson` | **0.030** (n=55) | 0.023 | 0.029 |

- **Seen and unseen.** 406 of the 795 graph materials are in the
  formation-energy/classifier dataset, 286 in the band-gap one and 84 in the
  elastic one. Unseen errors are 1.3–2.1× the in-dataset ones, so the
  checkpoint validation numbers are optimistic for this chemistry.
- **Formation energy works.** Unseen MAE is 0.063 eV/atom with almost no
  bias. That is above the PRD's 0.03–0.04 retraining gate, but usable for
  predicted values with a stated error.
- **Band gap is the weak spot.**
  - Without gating, the regressor gives DFT metals a mean gap of 1.63 eV.
  - Gating cuts the unseen MAE from 0.82 to 0.62 eV.
  - The classifier is the bottleneck: its metal recall on unseen materials
    is only 0.60.
- **Moduli are log₁₀(GPa).** Read as ln, they would be off by 2.2–2.6.
- **Speed (CPU).** 795 structures featurise in 9.3 s and run through all 7
  models in 1.8 s.

### 8.3 M2 results (2026-10-22)

**Code**

| File | Contents |
|---|---|
| `src/matgraph/cgcnn/validate.py` | Parse limits, checks and `struct-` naming |
| `src/matgraph/ingest_structure.py` | The CGCNN path: derived record, StructureMatcher dedup, predicted classes and rules, profile, persistence |
| `src/matgraph/ingest.py` | The relate/report/persist core is factored out as `relate_and_report`. Adds the `predicted` provenance, predicted rule evaluation, provenance on edges, and `cgcnn` replay. |
| `src/matgraph/schema.py` | `property_source`, `predicted_*` and `prediction_model` on `Material` |
| `src/matgraph/graph_store.py` | A formula resolves to the MP node before a `struct-` polymorph |
| `scripts/add_material.py --structure` | Structure input |
| `scripts/predict.py` | Predictions only |

**Naming hash.** It fingerprints site environments, not coordinates, so site
order, origin and supercells give the same id. Over the 795 graph
structures:
- **Uniqueness:** all 795 ids are distinct.
- **Re-described inputs:** shuffled, shifted and supercell versions always
  hash the same.
- **Symmetrised standard cells:** about 12% hash differently, because
  standardisation idealises atom positions.

That last case does not split nodes. Deduplication goes through
`StructureMatcher` against every graph material with the same formula, so a
re-upload resolves to the stored node and keeps its id.

**End-to-end preview on the real graph**

| Input | Outcome |
|---|---|
| LiFePO4 from `mp-aaaabcdl`, as POSCAR | Matches the MP node. Prediction vs DFT: E_form −2.561 vs −2.480 eV/atom, gap 3.97 vs 3.92 eV. |
| The same LiFePO4 as a conventional-cell CIF | Matches the MP node |
| A hypothetical wurtzite MgO | New node `struct-MgO-sg186-6774f5b4`, alongside rock-salt `mp-aaaaabwr`. Gets derived edges (P6_3mc shared with 12 materials, Mg-O, AB, Mg2+/O2-), a predicted `wide_gap` class (3.99 ± 0.62 eV), and rules marked predicted. |

**Persist and replay.** This was checked on a scratch copy of the cognee
store: one copy to persist into, and a pristine one to replay into.
- **Persist:** the node lands with its `predicted_*` attributes, and the
  `classified_as` edge keeps `provenance: predicted` through Kuzu.
- **Formula lookup:** `MgO` still resolves to the MP node.
- **Dedup after persist:** the primitive-cell re-upload resolves to the
  stored node.
- **Replay:** replaying into the pristine copy reproduces identical node
  properties and edges.
- **Caveat:** that run used stubbed embeddings. The local Ollama server
  cannot embed, because its app bundle was moved to the Trash, so
  `llama-server` is missing. This also blocks real persists and query
  embedding until Ollama is reinstalled.

**Tests.** 85 pass. `tests/test_ingest_structure.py` covers validation,
naming, dedup, predicted classes and rules, gating, and persist→replay. It
uses a fake predictor and no stores.

**Open until M3 (closed in M3).** The retriever did not filter predicted
edges yet, so a persisted struct node could show up in class- or
domain-based queries.

### 8.4 M3 results (2026-10-22)

**Benchmark identity.** For each of the 55 benchmark queries' gold
constraints (semantic scores off), we compared two runs:
- the Phase 2 retriever and evidence code (git `HEAD` before M3) on the
  clean graph;
- the new code with predictions off, on the real graph *with* a preview
  `struct-MgO` node added.

Retrieval (candidates, scores, checks, `total_matching`, `pool_size`) and
evidence text match exactly: **0 differences in 55**.

That check found two ranking leaks, both now fixed. A struct node shifted
the hub IDF (anchor queries, tier 4) and the density spread. Before the
fix, 10 tier-4 rankings differed. A regression test covers this.

**API**

| Endpoint | Result |
|---|---|
| `/health` | `cgcnn: available, 7 models, smoke_test ok, max_abs_diff 0.0` against the reference outputs on COD 1000041 |
| `/predict` | Returns the predictions with expected errors and leaves the graph unchanged |
| `/materials/ingest` with a structure | Preview report; templated explanation 35/35 grounded |

**Dashboard (checked in the browser on the real graph)**
- **From structure tab:**
  - Predict shows a table of predictions with their expected errors.
  - Add to graph renders the relation report, with dashed predicted edges
    and predicted provenance badges.
- **Ask, "materials containing Mg with band gap above 3 eV":**
  - Predictions off: MgO and MgO2 (MP).
  - Predictions on: the wurtzite `struct-MgO` is added with a "predicted"
    badge. Its comparison cells are amber "≈" values; the derived density
    stays plain. The explanation is 40/40 grounded.
- **Console:** no errors.

**Graceful degradation (new).** If the question cannot be embedded within
30 s, `/query` answers from the graph constraints alone, with a notice.
Before this, cognee retried for about 4 minutes and returned a 500. That is
the current state of this machine: the Ollama app is in the Trash.

**Also fixed.**
- **Explanations missed the material's own facts.** Relation evidence now
  lists the ingested material first. Before, templated explanations of a
  new material covered only its neighbours, which dropped its own and
  predicted facts.
- **Over-flagged rule verdicts.** Rule verdicts are marked predicted only
  when they rest on the predicted gap. Compositional verdicts are exact.

**Tests.** 97 pass.

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
   sha256 pins. The alternative is to commit the 2.8 MB of converted weights
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
