"""CGCNN inference: featuriser, invariances, and parity with txie-93/cgcnn.

Featuriser and invariance tests run offline with no weights (randomly
initialised models in eval mode). Parity tests need the converted
pre-trained weights (scripts/fetch_cgcnn_weights.py) and are skipped
without them; their expected values come from the reference code itself
(scripts/make_cgcnn_reference_fixture.py).
"""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pymatgen.core import Lattice, Structure  # noqa: E402
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # noqa: E402

from matgraph.cgcnn.featurize import GAUSSIAN_FILTER, MAX_NUM_NBR, RADIUS, FeaturizationError, featurize  # noqa: E402
from matgraph.cgcnn.model import CrystalGraphConvNet  # noqa: E402
from matgraph.cgcnn.predictor import CGCNNPredictor, collate, weights_available  # noqa: E402
from matgraph.cgcnn.registry import MODELS  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "cgcnn"
REFERENCE = json.loads((FIXTURES / "reference_predictions.json").read_text())["predictions"]
needs_weights = pytest.mark.skipif(not weights_available(), reason="run scripts/fetch_cgcnn_weights.py")


def rocksalt() -> Structure:
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def fixture_structure(cif_id: str) -> Structure:
    return Structure.from_file(FIXTURES / f"{cif_id}.cif")


def random_model(seed: int = 0) -> CrystalGraphConvNet:
    torch.manual_seed(seed)
    net = CrystalGraphConvNet(92, len(GAUSSIAN_FILTER), atom_fea_len=64, n_conv=4, h_fea_len=32)
    net.eval()
    return net


@torch.no_grad()
def embed_and_predict(net, structure):
    out, embedding = net(*collate([featurize(structure)]), return_embedding=True)
    return out.item(), embedding[0].numpy()


# --- featuriser ------------------------------------------------------------------


def test_featurize_shapes_and_neighbour_order():
    graph = featurize(rocksalt())
    assert graph.atom_fea.shape == (8, 92)
    assert graph.nbr_idx.shape == graph.nbr_dist.shape == (8, MAX_NUM_NBR)
    assert graph.n_padded_atoms == 0
    # 6 Cl at a/2, then Na at a/sqrt(2) - sorted ascending, as the reference does.
    assert np.allclose(graph.nbr_dist[0, :6], 5.64 / 2)
    assert np.allclose(graph.nbr_dist[0, 6:], 5.64 / np.sqrt(2))
    assert np.all(np.diff(graph.nbr_dist, axis=1) >= -1e-9)


def test_featurize_pads_sparse_structures_like_reference():
    # One atom in a huge cell: no neighbours inside 8 A, so every slot is padding.
    graph = featurize(Structure(Lattice.cubic(20.0), ["Si"], [[0, 0, 0]]))
    assert graph.n_padded_atoms == 1
    assert np.all(graph.nbr_idx == 0)
    assert np.all(graph.nbr_dist == RADIUS + 1.0)


def test_featurize_rejects_disordered_sites():
    disordered = Structure(Lattice.cubic(4.0), [{"Fe": 0.5, "Ni": 0.5}], [[0, 0, 0]])
    with pytest.raises(FeaturizationError, match="ordered"):
        featurize(disordered)


def test_pooling_matches_reference_per_crystal_mean():
    atom_fea = torch.randn(7, 5)
    crystal_idx = torch.tensor([0, 0, 0, 1, 1, 2, 2])
    expected = torch.cat([atom_fea[idx].mean(0, keepdim=True)
                          for idx in (torch.arange(0, 3), torch.arange(3, 5), torch.arange(5, 7))])
    assert torch.allclose(CrystalGraphConvNet.pooling(atom_fea, crystal_idx), expected, atol=1e-6)


# --- invariances (PRD: verified before the encoder is trusted) ----------------------


@pytest.mark.parametrize("make", [rocksalt, lambda: fixture_structure("1000041")], ids=["NaCl", "cod-1000041"])
def test_invariances(make):
    net = random_model()
    base = make()
    value, embedding = embed_and_predict(net, base)

    rng = np.random.default_rng(0)
    rotation = Structure(
        Lattice(base.lattice.matrix @ _random_rotation(rng).T), base.species, base.frac_coords
    )
    permuted = Structure.from_sites([base[i] for i in rng.permutation(len(base))])
    variants = {
        "supercell": base * (2, 1, 1),
        "permuted sites": permuted,
        "rotated cell": rotation,
        "primitive setting": SpacegroupAnalyzer(base).get_primitive_standard_structure(),
        "conventional setting": SpacegroupAnalyzer(base).get_conventional_standard_structure(),
    }
    for label, variant in variants.items():
        v, e = embed_and_predict(net, variant)
        assert v == pytest.approx(value, abs=1e-4), label
        assert np.allclose(e, embedding, atol=1e-4), label


def _random_rotation(rng) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_batching_does_not_mix_crystals():
    net = random_model()
    structures = [fixture_structure(cid) for cid in sorted(REFERENCE)[:4]]
    with torch.no_grad():
        batched = net(*collate([featurize(s) for s in structures])).view(-1).tolist()
    singles = [embed_and_predict(net, s)[0] for s in structures]
    assert batched == pytest.approx(singles, abs=1e-5)


# --- parity with the reference implementation and pre-trained weights ------------


@pytest.fixture(scope="module")
def predictor():
    return CGCNNPredictor()


@needs_weights
def test_pretrained_architecture_comes_from_checkpoint(predictor):
    for m in predictor.models:
        assert m.meta["args"]["n_conv"] == 4 and m.meta["args"]["h_fea_len"] == 32
        assert len(m.net.convs) == 4 and m.net.conv_to_fc.out_features == 32


@needs_weights
def test_parity_with_reference_predictions(predictor):
    cif_ids = sorted(REFERENCE)
    results = predictor.predict_many([fixture_structure(cid) for cid in cif_ids])
    for cid, preds in zip(cif_ids, results):
        assert set(preds) == {spec.name for spec in MODELS}
        for name, expected in REFERENCE[cid].items():
            assert preds[name].value == pytest.approx(expected, abs=1e-5, rel=1e-5), (cid, name)


@needs_weights
def test_prediction_metadata(predictor):
    preds = predictor.predict(rocksalt())
    gap = preds["band-gap"]
    assert gap.attribute == "predicted_band_gap" and gap.units == "eV"
    assert gap.model_id == "cgcnn-txie93-f42ab23/band-gap"
    assert gap.validation_score == pytest.approx(0.369, abs=1e-3)
    assert 0.0 <= preds["semi-metal-classification"].value <= 1.0
    assert gap.flags == ()
