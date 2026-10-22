"""Parse and check user-supplied crystal structures, and name them.

Limits (docs/PHASE3_PLAN.md §5): at most MAX_INPUT_BYTES of CIF/POSCAR text,
at most MAX_SITES sites, ordered sites only (CGCNN needs one element per
site), elements with CGCNN features that the graph does not exclude, and no
two atoms closer than MIN_DISTANCE. The size and site caps bound parse and
featurisation time, so no separate parse timeout is applied.

Naming (§4.1): struct-<reduced_formula>-sg<number>-<hash8>. The space group
comes from SpacegroupAnalyzer (symprec 0.1, as MP uses). hash8 fingerprints
the geometry independently of the cell description: volume per atom plus the
multiset of site environments (species and sorted distances to the nearest
neighbours, rounded to 0.01 A), reduced by their gcd. Site order, origin and
supercells give the same hash; a symmetrised standard cell can differ (its
atoms are idealised by up to symprec). The id only names a *new* crystal:
deduplication is by StructureMatcher (matgraph.ingest_structure), so a
re-upload in any setting resolves to the stored node and keeps its id.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from functools import reduce
from math import gcd
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

MAX_INPUT_BYTES = 1_000_000
MAX_SITES = 200
MIN_DISTANCE = 0.5  # A
SYMPREC = 0.1
HASH_RADIUS = 6.0  # A
HASH_NEIGHBOURS = 12
FORMATS = {".cif": "cif", ".mcif": "cif", ".vasp": "poscar", ".poscar": "poscar"}


class StructureError(ValueError):
    pass


def parse_structure(text: str, fmt: str | None = None) -> Structure:
    """CIF or POSCAR text -> Structure (first structure of a multi-block CIF)."""
    if len(text.encode()) > MAX_INPUT_BYTES:
        raise StructureError(f"structure input is larger than {MAX_INPUT_BYTES // 1000} kB")
    formats = [fmt] if fmt else (["cif", "poscar"] if "data_" in text else ["poscar", "cif"])
    errors = []
    for f in formats:
        try:
            return Structure.from_str(text, fmt=f)
        except Exception as error:  # noqa: BLE001 - pymatgen parsers raise many types
            errors.append(f"{f}: {error}")
    raise StructureError("could not parse the structure as " + " or ".join(formats) + " (" + "; ".join(errors) + ")")


def read_structure_file(path: Path) -> Structure:
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise StructureError(f"{path.name} is larger than {MAX_INPUT_BYTES // 1000} kB")
    name = path.name.lower()
    fmt = FORMATS.get(path.suffix.lower()) or ("poscar" if name.startswith(("poscar", "contcar")) else None)
    return parse_structure(path.read_text(errors="replace"), fmt)


def allowed_elements(elements_table: dict) -> set[str]:
    """Elements the graph's element table knows that also have CGCNN features."""
    from pymatgen.core import Element

    from matgraph.cgcnn.featurize import atom_init

    zs = set(atom_init())
    return {sym for sym in elements_table if Element(sym).Z in zs}


def check_structure(structure: Structure, allowed_elements: set[str], excluded_elements: set[str]) -> None:
    if len(structure) == 0:
        raise StructureError("structure has no sites")
    if len(structure) > MAX_SITES:
        raise StructureError(f"structure has {len(structure)} sites; the limit is {MAX_SITES}")
    if not structure.is_ordered:
        raise StructureError("structure has partial occupancies; CGCNN needs an ordered structure")
    symbols = {el.symbol for el in structure.composition.elements}
    if excluded := sorted(symbols & excluded_elements):
        raise StructureError(f"structure contains excluded elements: {', '.join(excluded)}")
    if unknown := sorted(symbols - allowed_elements):
        raise StructureError(f"no element data for: {', '.join(unknown)}")
    if structure.volume <= 0:
        raise StructureError("structure has a degenerate lattice")
    close = [n for nbrs in structure.get_all_neighbors(MIN_DISTANCE) for n in nbrs]
    if close:
        raise StructureError(f"atoms closer than {MIN_DISTANCE} A ({min(n.nn_distance for n in close):.3f} A)")


def canonical_hash(structure: Structure) -> str:
    envs: Counter = Counter()
    for site, nbrs in zip(structure, structure.get_all_neighbors(HASH_RADIUS)):
        distances = sorted(round(n.nn_distance, 2) for n in nbrs)[:HASH_NEIGHBOURS]
        envs[(site.specie.symbol, tuple(distances))] += 1
    divisor = reduce(gcd, envs.values())
    environments = sorted((env, count // divisor) for env, count in envs.items())
    volume_per_atom = round(structure.volume / len(structure), 1)
    return hashlib.sha256(repr((volume_per_atom, environments)).encode()).hexdigest()[:8]


def struct_id(structure: Structure) -> str:
    number = SpacegroupAnalyzer(structure, symprec=SYMPREC).get_space_group_number()
    return f"struct-{structure.composition.reduced_formula}-sg{number}-{canonical_hash(structure)}"
