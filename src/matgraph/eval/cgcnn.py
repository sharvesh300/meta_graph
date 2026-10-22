"""Phase 3 M1: evaluate the pre-trained CGCNN models on the graph's materials.

Every graph material has a cached MP structure and DFT values, so each
model's prediction can be compared against the value the graph already
holds - per model, per cluster, and split into materials that were / were
not in the model's 2017 dataset (by legacy MP id, see registry). The unseen
split gives the expected errors quoted next to predicted values.

The two interpretation choices (moduli log base, band-gap gating threshold)
are evaluated here as alternatives; the chosen ones live in
matgraph.cgcnn.interpret, and the "settled" numbers apply them.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from matgraph.cgcnn.interpret import METAL_THRESHOLD, interpret
from matgraph.cgcnn.registry import MODEL_SET, MODELS_BY_NAME, SOURCE_COMMIT, SOURCE_REPO
from matgraph.enrich import DATA_DIR, elastic_properties, elastic_status

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "artifacts" / "eval"
JSON_PATH = OUT_DIR / "cgcnn_graph_eval.json"
MD_PATH = OUT_DIR / "cgcnn_graph_eval.md"
MIN_UNSEEN_FOR_EXPECTED_ERROR = 30
GATE_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7)


def _log10_or_none(v: float | None) -> float | None:
    return math.log10(v) if v else None


# model name -> (attribute, DFT reference from a raw record, prediction from
# interpreted values, units). Moduli are compared in log10(GPa), the space the
# models predict in, so errors are multiplicative factors.
VIEWS: dict[str, tuple[str, Callable[[dict], float | None], Callable[[dict], float], str]] = {
    "formation-energy-per-atom": ("predicted_formation_energy_per_atom", lambda r: r.get("formation_energy_per_atom"),
                                  lambda v: v["predicted_formation_energy_per_atom"].value, "eV/atom"),
    "band-gap": ("predicted_band_gap", lambda r: r.get("band_gap"), lambda v: v["predicted_band_gap"].value, "eV"),
    "efermi": ("predicted_efermi", lambda r: r.get("efermi"), lambda v: v["predicted_efermi"].value, "eV"),
    "bulk-moduli": ("predicted_bulk_modulus_vrh",
                    lambda r: _log10_or_none((elastic_properties(r) or {}).get("bulk_modulus_vrh")),
                    lambda v: v["predicted_bulk_modulus_vrh"].detail["log10_gpa"], "log10(GPa)"),
    "shear-moduli": ("predicted_shear_modulus_vrh",
                     lambda r: _log10_or_none((elastic_properties(r) or {}).get("shear_modulus_vrh")),
                     lambda v: v["predicted_shear_modulus_vrh"].detail["log10_gpa"], "log10(GPa)"),
    "poisson-ratio": ("predicted_homogeneous_poisson", lambda r: (elastic_properties(r) or {}).get("homogeneous_poisson"),
                      lambda v: v["predicted_homogeneous_poisson"].value, ""),
}
# The gated band gap uses both models; the classifier's dataset (46,744 ids)
# contains the band-gap one (27,430), so "unseen" means outside the former.
SPLIT_MODEL = {"band-gap": "semi-metal-classification"}


def load_graph_records() -> list[dict[str, Any]]:
    """Every cached MP record (the curated build plus persisted user additions)."""
    return [json.loads(p.read_text()) for p in sorted(DATA_DIR.glob("mp-*.json"))]


def cluster_of(record: dict[str, Any]) -> str:
    return record.get("_cluster") or "user_added"


# --- metrics ----------------------------------------------------------------------


def regression_stats(pairs: list[tuple[float, float]]) -> dict[str, float | int | None]:
    """pairs of (prediction, reference)."""
    n = len(pairs)
    if not n:
        return {"n": 0, "mae": None, "rmse": None, "bias": None, "r2": None}
    errors = [p - t for p, t in pairs]
    mean_t = sum(t for _, t in pairs) / n
    ss_tot = sum((t - mean_t) ** 2 for _, t in pairs)
    ss_res = sum(e * e for e in errors)
    return {
        "n": n,
        "mae": sum(abs(e) for e in errors) / n,
        "rmse": math.sqrt(ss_res / n),
        "bias": sum(errors) / n,  # mean(prediction - DFT)
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else None,
    }


def roc_auc(scores: list[float], labels: list[int]) -> float | None:
    """Mann-Whitney AUC with average ranks for ties."""
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, labels) if y)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def classification_stats(scores: list[float], labels: list[int], threshold: float) -> dict[str, Any]:
    pred = [int(s >= threshold) for s in scores]
    tp = sum(1 for p, y in zip(pred, labels) if p and y)
    fp = sum(1 for p, y in zip(pred, labels) if p and not y)
    fn = sum(1 for p, y in zip(pred, labels) if not p and y)
    return {
        "n": len(labels),
        "n_metal": sum(labels),
        "auc": roc_auc(scores, labels),
        "accuracy": sum(1 for p, y in zip(pred, labels) if p == y) / len(labels) if labels else None,
        "precision_metal": tp / (tp + fp) if tp + fp else None,
        "recall_metal": tp / (tp + fn) if tp + fn else None,
    }


# --- evaluation ------------------------------------------------------------------------


def predict_records(records: list[dict[str, Any]]):
    """Predictor outputs per record (model name -> Prediction), plus timings."""
    from pymatgen.core import Structure

    from matgraph.cgcnn.featurize import featurize
    from matgraph.cgcnn.predictor import CGCNNPredictor

    predictor = CGCNNPredictor()
    t0 = time.perf_counter()
    graphs = [featurize(Structure.from_dict(r["structure"])) for r in records]
    t1 = time.perf_counter()
    preds = predictor.predict_graphs(graphs)
    t2 = time.perf_counter()
    timing = {"featurize_s": round(t1 - t0, 2), "inference_s": round(t2 - t1, 2),
              "structures_with_padding": sum(1 for g in graphs if g.n_padded_atoms)}
    return preds, timing


def _splits(records, idx: list[int], ids_in_dataset: set[str]) -> dict[str, list[int]]:
    clusters = sorted({cluster_of(r) for r in records})
    return {
        "all": idx,
        "in_model_dataset": [i for i in idx if records[i]["material_id"] in ids_in_dataset],
        "unseen": [i for i in idx if records[i]["material_id"] not in ids_in_dataset],
        **{f"cluster:{c}": [i for i in idx if cluster_of(records[i]) == c] for c in clusters},
    }


def _expected_error(stats: dict[str, dict], key: str, units: str) -> dict[str, Any]:
    unseen, everything = stats["unseen"], stats["all"]
    if unseen["n"] >= MIN_UNSEEN_FOR_EXPECTED_ERROR:
        return {"value": unseen[key], "units": units, "n": unseen["n"], "basis": "unseen graph materials"}
    return {"value": everything[key], "units": units, "n": everything["n"],
            "basis": f"all graph materials (only {unseen['n']} unseen)"}


def evaluate(records: list[dict[str, Any]], preds: list[dict], training_ids: dict[str, set[str]]) -> dict[str, Any]:
    raw = [{name: p.value for name, p in pr.items()} for pr in preds]
    values = [interpret(pr) for pr in preds]
    report: dict[str, Any] = {
        "model_set": MODEL_SET, "source": f"{SOURCE_REPO}@{SOURCE_COMMIT}", "n_materials": len(records),
        "metal_threshold": METAL_THRESHOLD, "models": {}, "expected_errors": {},
    }

    for name, (attribute, reference, predicted, units) in VIEWS.items():
        idx = [i for i, r in enumerate(records) if reference(r) is not None]
        splits = _splits(records, idx, training_ids[SPLIT_MODEL.get(name, name)])
        stats = {key: regression_stats([(predicted(values[i]), reference(records[i])) for i in sel])
                 for key, sel in splits.items()}
        report["models"][name] = {"attribute": attribute, "units": units, "stats": stats,
                                  "checkpoint_validation_mae": None}
        report["expected_errors"][attribute] = _expected_error(stats, "mae", units)

    name = "semi-metal-classification"
    idx = [i for i, r in enumerate(records) if r.get("is_metal") is not None]
    splits = _splits(records, idx, training_ids[name])
    stats = {key: classification_stats([raw[i][name] for i in sel], [int(records[i]["is_metal"]) for i in sel],
                                       METAL_THRESHOLD) for key, sel in splits.items()}
    report["models"][name] = {"attribute": "predicted_is_metal", "units": "", "stats": stats}
    err = _expected_error(stats, "accuracy", "error rate")
    err["value"] = 1 - err["value"]
    report["expected_errors"]["predicted_is_metal"] = err

    report["decisions"] = {"moduli_log_base": _log_base_evidence(records, raw),
                           "band_gap_gating": _gating_evidence(records, raw, training_ids[name])}
    report["elastic_status_counts"] = dict(sorted(Counter(elastic_status(r) for r in records).items()))
    return report


def _log_base_evidence(records, raw) -> dict[str, Any]:
    out = {}
    for name, field in (("bulk-moduli", "bulk_modulus_vrh"), ("shear-moduli", "shear_modulus_vrh")):
        idx = [i for i, r in enumerate(records) if (elastic_properties(r) or {}).get(field)]
        out[name] = {base: regression_stats([(raw[i][name], log(elastic_properties(records[i])[field])) for i in idx])
                     for base, log in (("log10", math.log10), ("ln", math.log))}
    return out


def _gating_evidence(records, raw, classifier_ids: set[str]) -> dict[str, Any]:
    """The regressor saw non-metals only; gating by P(metal) is chosen on the
    seen split and reported on the unseen one."""
    metals = [i for i, r in enumerate(records) if r.get("is_metal")]
    nonmetals = [i for i, r in enumerate(records) if r.get("is_metal") is False]
    idx = [i for i, r in enumerate(records) if r.get("band_gap") is not None]
    seen = [i for i in idx if records[i]["material_id"] in classifier_ids]
    unseen = [i for i in idx if records[i]["material_id"] not in classifier_ids]

    def gated(sel, threshold):
        return regression_stats([
            (0.0 if threshold is not None and raw[i]["semi-metal-classification"] >= threshold
             else max(raw[i]["band-gap"], 0.0), records[i]["band_gap"]) for i in sel
        ])

    return {
        "regressor_on_dft_metals": {
            "n": len(metals),
            "mean_predicted_gap": sum(raw[i]["band-gap"] for i in metals) / len(metals),
            "share_above_0.5eV": sum(1 for i in metals if raw[i]["band-gap"] > 0.5) / len(metals),
        },
        "regressor_on_dft_nonmetals": regression_stats([(raw[i]["band-gap"], records[i]["band_gap"]) for i in nonmetals]),
        "thresholds": {str(t): {"seen": gated(seen, t), "unseen": gated(unseen, t)}
                       for t in (None, *GATE_THRESHOLDS)},
        "chosen": METAL_THRESHOLD,
    }


# --- report ------------------------------------------------------------------------


def _f(v, digits=3) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def render_markdown(report: dict[str, Any], timing: dict[str, Any]) -> str:
    m, e = report["models"], report["expected_errors"]
    lines = [
        "# CGCNN pre-trained models on the graph (Phase 3 M1)",
        "",
        f"Generated by `scripts/evaluate_cgcnn.py`. Models: `{report['source']}` (MP snapshot 2017-04-19). "
        f"Reference values: the DFT values already on the {report['n_materials']} graph materials (current MP). "
        "**In dataset** = the material's legacy MP id is in the model's published dataset (train + val + test; "
        "the split was random and not published, so ~60% of these were trained on). **Unseen** = not in it; "
        "its MAE is the expected error quoted next to predicted values.",
        "",
        "## Summary (settled interpretation)",
        "",
        "| Model | Attribute | Units | n | MAE in dataset | MAE unseen | Bias unseen | Checkpoint val. MAE | Expected error |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, entry in m.items():
        if name == "semi-metal-classification":
            continue
        s = entry["stats"]
        val = _f(_checkpoint_score(name), 4) + (" (regressor, non-metals)" if name == "band-gap" else "")
        err = e[entry["attribute"]]
        lines.append(
            f"| {name} | `{entry['attribute']}` | {entry['units']} | {s['all']['n']} "
            f"(seen {s['in_model_dataset']['n']}) | {_f(s['in_model_dataset']['mae'])} | {_f(s['unseen']['mae'])} "
            f"| {_f(s['unseen']['bias'], 3)} | {val} | **{_f(err['value'])}** ({err['basis']}, n={err['n']}) |"
        )
    c = m["semi-metal-classification"]["stats"]
    err = e["predicted_is_metal"]
    lines += [
        "",
        f"Band gap is the gated pipeline (0 when P(metal) ≥ {report['metal_threshold']}); its unseen split uses the "
        "classifier's dataset, which contains the band-gap model's. Moduli errors are in log10(GPa): an MAE of 0.07 "
        "is a typical factor of 10^0.07 ≈ 1.17.",
        "",
        f"### Metal classifier (threshold {report['metal_threshold']})",
        "",
        "| Split | n | metals | AUC | accuracy | precision (metal) | recall (metal) |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in ("all", "in_model_dataset", "unseen"):
        s = c[key]
        lines.append(f"| {key} | {s['n']} | {s['n_metal']} | {_f(s['auc'])} | {_f(s['accuracy'])} "
                     f"| {_f(s['precision_metal'])} | {_f(s['recall_metal'])} |")
    lines += [f"", f"Expected error: **{_f(err['value'])}** misclassification rate ({err['basis']}, n={err['n']}). "
              f"Checkpoint validation AUC: {_f(_checkpoint_score('semi-metal-classification'))}.", ""]

    d = report["decisions"]
    lines += ["## Decision 1: moduli log base", "",
              "| Model | MAE vs log10(DFT) | MAE vs ln(DFT) | bias log10 | bias ln |", "|---|---|---|---|---|"]
    for name, cand in d["moduli_log_base"].items():
        lines.append(f"| {name} | {_f(cand['log10']['mae'])} | {_f(cand['ln']['mae'])} "
                     f"| {_f(cand['log10']['bias'])} | {_f(cand['ln']['bias'])} |")
    lines += ["", "The models predict **log10(GPa)**; `interpret` converts to GPa.", ""]

    g = d["band_gap_gating"]
    lines += [
        "## Decision 2: band-gap gating", "",
        f"The band-gap model was trained on non-metals only. On the {g['regressor_on_dft_metals']['n']} DFT metals "
        f"it predicts a mean gap of {_f(g['regressor_on_dft_metals']['mean_predicted_gap'], 2)} eV, "
        f"{g['regressor_on_dft_metals']['share_above_0.5eV']:.0%} of them above 0.5 eV. On DFT non-metals its MAE is "
        f"{_f(g['regressor_on_dft_nonmetals']['mae'])} eV.",
        "",
        "Gap = 0 when P(metal) ≥ threshold, else the regressor (clipped at 0). The threshold is **chosen on the "
        "seen split** and reported on the unseen one, so the expected error is not tuned on the data it is measured on.",
        "",
        "| Threshold | MAE seen | bias seen | MAE unseen | bias unseen |", "|---|---|---|---|---|",
    ]
    for t, s in g["thresholds"].items():
        label = "no gating" if t == "None" else t
        mark = " ← chosen" if t == str(g["chosen"]) else ""
        lines.append(f"| {label}{mark} | {_f(s['seen']['mae'])} | {_f(s['seen']['bias'])} "
                     f"| {_f(s['unseen']['mae'])} | {_f(s['unseen']['bias'])} |")

    lines += ["", "## By cluster (all graph materials, settled)", ""]
    clusters = [k.split(":", 1)[1] for k in m["formation-energy-per-atom"]["stats"] if k.startswith("cluster:")]
    names = ["formation-energy-per-atom", "band-gap", "efermi"]
    lines.append("| Cluster | n | " + " | ".join(f"{n} MAE / bias" for n in names) + " | metal acc. |")
    lines.append("|---|---|" + "---|" * (len(names) + 1))
    for cl in clusters:
        key = f"cluster:{cl}"
        cells = [f"{_f(m[n]['stats'][key]['mae'])} / {_f(m[n]['stats'][key]['bias'])}" for n in names]
        lines.append(f"| {cl} | {m[names[0]]['stats'][key]['n']} | " + " | ".join(cells)
                     + f" | {_f(c[key]['accuracy'])} |")

    counts = report["elastic_status_counts"]
    lines += [
        "",
        "## Notes",
        "",
        f"- Elastic comparisons use only physical DFT tensors: {counts.get('ok', 0)} ok, "
        f"{counts.get('unphysical', 0)} unphysical (excluded), {counts.get('missing', 0)} missing. So the elastic "
        "unseen split is small; see n in the summary.",
        "- The reference DFT values use current MP corrections and the models were trained on the 2017 snapshot; "
        "the bias columns show how much of the error is a systematic offset.",
        "- Values quoted to users come from `matgraph.cgcnn.interpret` with these expected errors attached.",
        f"- Timing (CPU): featurise {timing['featurize_s']} s, all 7 models {timing['inference_s']} s for "
        f"{report['n_materials']} structures; {timing['structures_with_padding']} needed neighbour padding.",
        "",
    ]
    return "\n".join(lines)


def _checkpoint_score(name: str) -> float | None:
    from matgraph.cgcnn.registry import WEIGHTS_DIR

    meta = WEIGHTS_DIR / MODELS_BY_NAME[name].name / "meta.json"
    return json.loads(meta.read_text())["best_validation_score"] if meta.exists() else None


def run() -> dict[str, Any]:
    from matgraph.cgcnn.registry import load_training_ids

    records = load_graph_records()
    preds, timing = predict_records(records)
    report = evaluate(records, preds, load_training_ids())
    report["timing"] = timing
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(report, indent=2))
    MD_PATH.write_text(render_markdown(report, timing))
    return report
