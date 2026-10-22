"""Pinned pre-trained CGCNN models from txie-93/cgcnn and where they live.

The checkpoints are downloaded and converted by scripts/fetch_cgcnn_weights.py
into WEIGHTS_DIR/<name>/{state_dict.pt, meta.json}; models/ is gitignored
(weights are build artifacts, not source).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent.parent.parent

SOURCE_REPO = "txie-93/cgcnn"
SOURCE_COMMIT = "f42ab233c4ee0c416879d6bc2d22a264418413ad"
MODEL_SET = "txie93-f42ab23"
WEIGHTS_DIR = ROOT / "models" / "cgcnn" / MODEL_SET


@dataclass(frozen=True)
class ModelSpec:
    name: str  # checkpoint file stem in pre-trained/
    attribute: str  # Material attribute the (interpreted) prediction will fill
    units: str  # units of the raw model output
    task: Literal["regression", "classification"]
    sha256: str  # of the original .pth.tar at SOURCE_COMMIT

    @property
    def model_id(self) -> str:
        return f"cgcnn-{MODEL_SET}/{self.name}"


# final-energy-per-atom is not fetched: an uncorrected total energy that has
# no counterpart in the graph schema.
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("formation-energy-per-atom", "predicted_formation_energy_per_atom", "eV/atom", "regression",
              "e2f8bb910744f2c2f21d968285b6fd4316d795d0f3e7465ed0e3b4f3c92651a1"),
    ModelSpec("band-gap", "predicted_band_gap", "eV", "regression",
              "d269775b9be0995c046abed3dd6556c281a3eff7b2e1d73cf81b99e2a1d4aa6c"),
    # Output is P(metal); positive class = metal, negative = semiconductor.
    ModelSpec("semi-metal-classification", "predicted_is_metal", "probability", "classification",
              "3791d48885afafb28c0797e22637b63ea591cf7fb27497ce537911261b92edc8"),
    ModelSpec("efermi", "predicted_efermi", "eV", "regression",
              "5ed4ba15b5bd3e04ce6b0eca75f15bb6465557aada4097f11291ea1e7cd48afb"),
    # Moduli are predicted in log(GPa); the log base is settled in M1.
    ModelSpec("bulk-moduli", "predicted_bulk_modulus_vrh", "log(GPa)", "regression",
              "0a3f66b9a0ee6acef11bb852717424a6c3560073239c17bdbea568cc43816dd3"),
    ModelSpec("shear-moduli", "predicted_shear_modulus_vrh", "log(GPa)", "regression",
              "34173bc59a0a8f9f74021155b304b74d9e709b1ff241d3f02e83031c9deb216c"),
    ModelSpec("poisson-ratio", "predicted_homogeneous_poisson", "", "regression",
              "8d152338550b49ff6c3790cccdb892ee8bafe513bc49ab21955cdf7807e89513"),
)

MODELS_BY_NAME = {spec.name: spec for spec in MODELS}


def checkpoint_url(name: str) -> str:
    return f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_COMMIT}/pre-trained/{name}.pth.tar"
