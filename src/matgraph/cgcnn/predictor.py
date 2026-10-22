"""Run the pinned pre-trained CGCNN models on pymatgen structures.

All models are loaded once; each structure is featurised once and shared by
every model. Values are the models' raw outputs (denormalised regression
values in the units of registry.ModelSpec, or P(metal) for the classifier);
interpreting them for the graph - band-gap gating by the classifier, the
moduli log base - is calibrated against the graph in M1, not here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from pymatgen.core import Structure

from matgraph.cgcnn.featurize import GAUSSIAN_FILTER, CrystalGraph, featurize, gaussian_expand
from matgraph.cgcnn.model import CrystalGraphConvNet
from matgraph.cgcnn.registry import MODELS, MODELS_BY_NAME, WEIGHTS_DIR, ModelSpec


class WeightsNotFound(FileNotFoundError):
    pass


@dataclass(frozen=True)
class Prediction:
    name: str  # registry model name, e.g. "band-gap"
    attribute: str  # Material attribute it maps to
    value: float
    units: str
    model_id: str
    validation_score: float  # checkpoint's best validation MAE (AUC for the classifier)
    flags: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class LoadedModel:
    spec: ModelSpec
    net: CrystalGraphConvNet
    meta: dict[str, Any]


def weights_available(weights_dir: Path = WEIGHTS_DIR) -> bool:
    return all((weights_dir / spec.name / "state_dict.pt").exists() for spec in MODELS)


def build_model(meta: dict[str, Any]) -> CrystalGraphConvNet:
    """Architecture from the checkpoint's own args - the pre-trained models use
    n_conv=4, h_fea_len=32, not the reference code's defaults."""
    args = meta["args"]
    return CrystalGraphConvNet(
        orig_atom_fea_len=meta["orig_atom_fea_len"],
        nbr_fea_len=meta["nbr_fea_len"],
        atom_fea_len=args["atom_fea_len"],
        n_conv=args["n_conv"],
        h_fea_len=args["h_fea_len"],
        n_h=args["n_h"],
        classification=args["task"] == "classification",
    )


def load_model(spec: ModelSpec, weights_dir: Path = WEIGHTS_DIR) -> LoadedModel:
    model_dir = weights_dir / spec.name
    if not (model_dir / "state_dict.pt").exists():
        raise WeightsNotFound(
            f"CGCNN weights for {spec.name!r} not found in {weights_dir} "
            "(run: uv run python scripts/fetch_cgcnn_weights.py)"
        )
    meta = json.loads((model_dir / "meta.json").read_text())
    net = build_model(meta)
    net.load_state_dict(torch.load(model_dir / "state_dict.pt", map_location="cpu", weights_only=True))
    net.eval()
    return LoadedModel(spec, net, meta)


def collate(graphs: list[CrystalGraph]) -> tuple[torch.Tensor, ...]:
    """Batch crystal graphs into one disconnected graph (as collate_pool does)."""
    atom_fea, nbr_fea, nbr_idx, crystal_idx = [], [], [], []
    base = 0
    for i, g in enumerate(graphs):
        atom_fea.append(g.atom_fea)
        nbr_fea.append(gaussian_expand(g.nbr_dist))
        nbr_idx.append(g.nbr_idx + base)
        crystal_idx.append(np.full(g.num_atoms, i, dtype=np.int64))
        base += g.num_atoms
    return (
        torch.tensor(np.concatenate(atom_fea), dtype=torch.float32),
        torch.tensor(np.concatenate(nbr_fea), dtype=torch.float32),
        torch.from_numpy(np.concatenate(nbr_idx)),
        torch.from_numpy(np.concatenate(crystal_idx)),
    )


def _flags(graph: CrystalGraph) -> tuple[str, ...]:
    return ("few_neighbours",) if graph.n_padded_atoms else ()


class CGCNNPredictor:
    def __init__(self, weights_dir: Path = WEIGHTS_DIR, names: Iterable[str] | None = None):
        specs = [MODELS_BY_NAME[n] for n in names] if names is not None else list(MODELS)
        self.models = [load_model(spec, weights_dir) for spec in specs]
        for m in self.models:
            if m.meta["nbr_fea_len"] != len(GAUSSIAN_FILTER):
                raise ValueError(f"{m.spec.name}: featuriser/model mismatch in edge feature length")

    @torch.no_grad()
    def predict_graphs(self, graphs: list[CrystalGraph], batch_size: int = 64) -> list[dict[str, Prediction]]:
        results: list[dict[str, Prediction]] = [{} for _ in graphs]
        for start in range(0, len(graphs), batch_size):
            chunk = graphs[start:start + batch_size]
            inputs = collate(chunk)
            for m in self.models:
                out = m.net(*inputs)
                if m.spec.task == "classification":
                    values = torch.exp(out)[:, 1]
                else:
                    norm = m.meta["normalizer"]
                    values = out.view(-1) * norm["std"] + norm["mean"]
                for offset, (graph, value) in enumerate(zip(chunk, values.tolist())):
                    results[start + offset][m.spec.name] = Prediction(
                        name=m.spec.name, attribute=m.spec.attribute, value=value, units=m.spec.units,
                        model_id=m.spec.model_id, validation_score=m.meta["best_validation_score"],
                        flags=_flags(graph),
                    )
        return results

    def predict_many(self, structures: list[Structure], batch_size: int = 64) -> list[dict[str, Prediction]]:
        return self.predict_graphs([featurize(s) for s in structures], batch_size)

    def predict(self, structure: Structure) -> dict[str, Prediction]:
        return self.predict_many([structure])[0]
