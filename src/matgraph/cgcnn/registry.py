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
    # Moduli are predicted in log10(GPa) (settled in M1, see cgcnn.interpret).
    ModelSpec("bulk-moduli", "predicted_bulk_modulus_vrh", "log10(GPa)", "regression",
              "0a3f66b9a0ee6acef11bb852717424a6c3560073239c17bdbea568cc43816dd3"),
    ModelSpec("shear-moduli", "predicted_shear_modulus_vrh", "log10(GPa)", "regression",
              "34173bc59a0a8f9f74021155b304b74d9e709b1ff241d3f02e83031c9deb216c"),
    ModelSpec("poisson-ratio", "predicted_homogeneous_poisson", "", "regression",
              "8d152338550b49ff6c3790cccdb892ee8bafe513bc49ab21955cdf7807e89513"),
)

MODELS_BY_NAME = {spec.name: spec for spec in MODELS}

# Legacy MP ids of each pre-trained model's dataset (train + val + test; the
# split itself was random and not published). Matched to models by size: the
# checkpoints' train/val/test counts sum to 46,742 / 27,429 / 3,399.
TRAINING_ID_LISTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "mp-ids-46744": ("b7519a45457a992d174408ce7fb5809633563360f4535b2bfedb0389b130fb38",
                     ("formation-energy-per-atom", "semi-metal-classification", "efermi")),
    "mp-ids-27430": ("b7eb48ae697546393ff9d7571442a909ce1253bc7240b4f2a8b504d3e3f54d82", ("band-gap",)),
    "mp-ids-3402": ("f9e8e726c92b2542fd7b88ad1240afd847605decbabd50f3325959be0c94eb50",
                    ("bulk-moduli", "shear-moduli", "poisson-ratio")),
}
TRAINING_IDS_DIR = WEIGHTS_DIR / "training_ids"


def checkpoint_url(name: str) -> str:
    return f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_COMMIT}/pre-trained/{name}.pth.tar"


def training_ids_url(list_name: str) -> str:
    return f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_COMMIT}/data/material-data/{list_name}.csv"


def legacy_to_new_id(legacy_id: str) -> str:
    """MP's new-format ids are the legacy number in base 26 over a-z, padded
    to 8 letters: mp-149 -> mp-aaaaaaft, mp-754118 -> mp-aaabqxoo."""
    prefix, number = legacy_id.rsplit("-", 1)
    n, letters = int(number), []
    while n:
        n, digit = divmod(n, 26)
        letters.append(chr(ord("a") + digit))
    return f"{prefix}-{''.join(reversed(letters)).rjust(8, 'a')}"


def load_training_ids() -> dict[str, set[str]]:
    """model name -> new-format ids of the materials in its dataset."""
    out: dict[str, set[str]] = {}
    for list_name, (_sha, model_names) in TRAINING_ID_LISTS.items():
        path = TRAINING_IDS_DIR / f"{list_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing (run: uv run python scripts/fetch_cgcnn_weights.py)")
        ids = {legacy_to_new_id(line.strip()) for line in path.read_text().splitlines() if line.strip()}
        for name in model_names:
            out[name] = ids
    return out
