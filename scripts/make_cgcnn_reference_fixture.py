"""Regenerate tests/fixtures/cgcnn/reference_predictions.json from the reference code.

Usage:
    git clone https://github.com/txie-93/cgcnn /path/to/cgcnn   # then check out the pinned commit
    uv run python scripts/make_cgcnn_reference_fixture.py /path/to/cgcnn

Runs txie-93/cgcnn's own CIFData + collate_pool + CrystalGraphConvNet (the
path its predict.py takes, minus the CLI/CSV wrapper) with the original
.pth.tar checkpoints on the repo's 10 sample-regression CIFs, for every model
in matgraph.cgcnn.registry. tests/test_cgcnn.py checks our featuriser and
predictor reproduce these outputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from matgraph.cgcnn.registry import MODELS, SOURCE_COMMIT  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "cgcnn" / "reference_predictions.json"
SAFE = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype, np.dtypes.Float64DType]


def main(ref_dir: Path) -> None:
    commit = subprocess.run(["git", "-C", str(ref_dir), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    if commit != SOURCE_COMMIT:
        raise SystemExit(f"{ref_dir} is at {commit}, expected pinned {SOURCE_COMMIT}")
    sys.path.insert(0, str(ref_dir))
    from cgcnn.data import CIFData, collate_pool
    from cgcnn.model import CrystalGraphConvNet

    dataset = CIFData(str(ref_dir / "data" / "sample-regression"))
    (inputs, _target, cif_ids) = collate_pool([dataset[i] for i in range(len(dataset))])
    out: dict[str, dict[str, float]] = {cid: {} for cid in cif_ids}
    for spec in MODELS:
        with torch.serialization.safe_globals(SAFE):
            ckpt = torch.load(ref_dir / "pre-trained" / f"{spec.name}.pth.tar", map_location="cpu", weights_only=True)
        a = ckpt["args"]
        model = CrystalGraphConvNet(inputs[0].shape[-1], inputs[1].shape[-1], atom_fea_len=a["atom_fea_len"],
                                    n_conv=a["n_conv"], h_fea_len=a["h_fea_len"], n_h=a["n_h"],
                                    classification=a["task"] == "classification")
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        with torch.no_grad():
            pred = model(*inputs)
        if a["task"] == "classification":
            values = torch.exp(pred)[:, 1]
        else:
            values = pred.view(-1) * float(ckpt["normalizer"]["std"]) + float(ckpt["normalizer"]["mean"])
        for cid, v in zip(cif_ids, values.tolist()):
            out[cid][spec.name] = v
    FIXTURE.write_text(json.dumps({"source_commit": commit, "predictions": out}, indent=2, sort_keys=True))
    print(f"wrote {FIXTURE} ({len(out)} structures x {len(MODELS)} models)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ref_dir", type=Path, help="local clone of txie-93/cgcnn at the pinned commit")
    main(parser.parse_args().ref_dir)
