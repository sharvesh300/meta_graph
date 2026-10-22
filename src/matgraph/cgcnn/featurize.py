"""Structure -> CGCNN crystal graph, matching txie-93/cgcnn's CIFData exactly.

Same settings as the pre-trained models: 92-d atom_init.json element vectors,
the 12 nearest neighbours within 8 A (periodic images allowed, so a multigraph),
padding with index 0 / distance r + 1 when fewer are found, and a Gaussian
distance expansion over arange(0, 8 + 0.2, 0.2) with var = 0.2 (41 dims).

Torch-free: the expansion is applied at batch time (predictor.collate) so a
featurised crystal stays small (distances, not 41-d edge vectors).
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

ATOM_INIT_PATH = Path(__file__).resolve().parent / "atom_init.json"
RADIUS = 8.0
MAX_NUM_NBR = 12
DMIN = 0.0
STEP = 0.2
GAUSSIAN_FILTER = np.arange(DMIN, RADIUS + STEP, STEP)
GAUSSIAN_VAR = STEP


class FeaturizationError(ValueError):
    pass


@dataclass(frozen=True)
class CrystalGraph:
    atom_fea: np.ndarray  # (n, 92) element vectors
    nbr_dist: np.ndarray  # (n, MAX_NUM_NBR) neighbour distances, A
    nbr_idx: np.ndarray  # (n, MAX_NUM_NBR) neighbour site indices
    n_padded_atoms: int  # atoms with fewer than MAX_NUM_NBR neighbours inside RADIUS

    @property
    def num_atoms(self) -> int:
        return self.atom_fea.shape[0]


@functools.cache
def atom_init() -> dict[int, np.ndarray]:
    raw = json.loads(ATOM_INIT_PATH.read_text())
    return {int(z): np.array(vec, dtype=float) for z, vec in raw.items()}


def gaussian_expand(distances: np.ndarray) -> np.ndarray:
    """(..., ) distances -> (..., len(GAUSSIAN_FILTER)) as in GaussianDistance.expand."""
    return np.exp(-((distances[..., np.newaxis] - GAUSSIAN_FILTER) ** 2) / GAUSSIAN_VAR**2)


def featurize(structure: Structure) -> CrystalGraph:
    table = atom_init()
    try:
        atom_fea = np.vstack([table[site.specie.Z] for site in structure])
    except KeyError as error:
        raise FeaturizationError(f"no CGCNN element features for Z={error.args[0]}") from error
    except AttributeError as error:  # disordered site: .specie is undefined
        raise FeaturizationError("CGCNN needs an ordered structure (no partial occupancies)") from error

    nbr_idx, nbr_dist, padded = [], [], 0
    for nbrs in structure.get_all_neighbors(RADIUS):
        nbrs = sorted(nbrs, key=lambda n: n.nn_distance)[:MAX_NUM_NBR]
        missing = MAX_NUM_NBR - len(nbrs)
        if missing:
            padded += 1
        nbr_idx.append([n.index for n in nbrs] + [0] * missing)
        nbr_dist.append([n.nn_distance for n in nbrs] + [RADIUS + 1.0] * missing)
    return CrystalGraph(
        atom_fea=atom_fea,
        nbr_dist=np.array(nbr_dist, dtype=float),
        nbr_idx=np.array(nbr_idx, dtype=np.int64),
        n_padded_atoms=padded,
    )
