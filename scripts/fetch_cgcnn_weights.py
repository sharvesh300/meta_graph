"""Download the pinned pre-trained CGCNN checkpoints and convert them.

Usage:
    uv run python scripts/fetch_cgcnn_weights.py [--from-dir PATH] [--force]

Each checkpoint is fetched from txie-93/cgcnn at a pinned commit (or read
from a local clone's pre-trained/ with --from-dir), checked against its
pinned sha256, and re-saved as models/cgcnn/<set>/<name>/state_dict.pt (plain
tensors, loadable with weights_only=True) plus meta.json (architecture args,
normalizer, validation score, provenance). Runtime code never unpickles the
original checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import requests
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from matgraph.cgcnn.registry import (  # noqa: E402
    MODELS, SOURCE_COMMIT, SOURCE_REPO, WEIGHTS_DIR, ModelSpec, checkpoint_url,
)

logger = logging.getLogger(__name__)

# The classifier stores best_mae_error (its AUC) as a numpy float64 scalar,
# which weights_only loading rejects unless these are allowlisted. The pickle
# names numpy 1.x's module path, so the scalar is registered under that name.
NUMPY_SAFE_GLOBALS = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype,
                      np.dtypes.Float64DType]


def _to_float(value) -> float:
    return float(value.item()) if hasattr(value, "item") else float(value)


def read_checkpoint(spec: ModelSpec, from_dir: Path | None) -> bytes:
    if from_dir is not None:
        return (from_dir / f"{spec.name}.pth.tar").read_bytes()
    resp = requests.get(checkpoint_url(spec.name), timeout=60)
    resp.raise_for_status()
    return resp.content


def convert(spec: ModelSpec, blob: bytes, out_dir: Path) -> dict:
    digest = hashlib.sha256(blob).hexdigest()
    if digest != spec.sha256:
        raise RuntimeError(f"{spec.name}: sha256 {digest} does not match pinned {spec.sha256}")
    tmp = out_dir / "original.pth.tar"
    tmp.write_bytes(blob)
    try:
        with torch.serialization.safe_globals(NUMPY_SAFE_GLOBALS):
            ckpt = torch.load(tmp, map_location="cpu", weights_only=True)
    finally:
        tmp.unlink()

    state_dict = {k: v.clone() for k, v in ckpt["state_dict"].items()}
    args = dict(ckpt["args"])
    # Machine-specific training paths are irrelevant here.
    args.pop("data_options", None)
    args.pop("atom_encoder_param", None)
    meta = {
        "name": spec.name,
        "attribute": spec.attribute,
        "units": spec.units,
        "task": spec.task,
        "args": args,
        "orig_atom_fea_len": state_dict["embedding.weight"].shape[1],
        "nbr_fea_len": state_dict["convs.0.fc_full.weight"].shape[1] - 2 * args["atom_fea_len"],
        "normalizer": {k: _to_float(v) for k, v in ckpt["normalizer"].items()},
        "best_validation_score": _to_float(ckpt["best_mae_error"]),
        "validation_metric": "AUC" if spec.task == "classification" else "MAE",
        "epoch": int(ckpt["epoch"]),
        "source": {"repo": SOURCE_REPO, "commit": SOURCE_COMMIT, "url": checkpoint_url(spec.name),
                   "sha256": spec.sha256},
    }
    torch.save(state_dict, out_dir / "state_dict.pt")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(from_dir: Path | None, force: bool) -> None:
    for spec in MODELS:
        out_dir = WEIGHTS_DIR / spec.name
        if (out_dir / "state_dict.pt").exists() and not force:
            logger.info("%-28s already converted", spec.name)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = convert(spec, read_checkpoint(spec, from_dir), out_dir)
        logger.info("%-28s n_conv=%d h_fea_len=%d  validation %s %.4f  -> %s", spec.name, meta["args"]["n_conv"],
                    meta["args"]["h_fea_len"], meta["validation_metric"], meta["best_validation_score"], out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--from-dir", type=Path, help="local txie-93/cgcnn pre-trained/ directory instead of GitHub")
    parser.add_argument("--force", action="store_true", help="re-convert even if already present")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    main(args.from_dir, args.force)
