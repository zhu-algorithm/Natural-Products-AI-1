import pytest
from npfm.chemistry import canonicalize, descriptors, similarity, conformers


def test_chirality_and_identity():
    a, b = "C[C@H](O)C(=O)O", "C[C@@H](O)C(=O)O"
    assert descriptors(a)["full_inchikey"] != descriptors(b)["full_inchikey"]
    assert similarity(a, b) < 1.0
    assert similarity(a, a) == 1.0


def test_invalid_and_computed_scope():
    with pytest.raises(ValueError): canonicalize("not smiles")
    with pytest.raises(ValueError): canonicalize("")
    assert descriptors("CCO")["value_type"] == "calculated_descriptor_not_experimental"


def test_actual_conformer_output(tmp_path):
    r = conformers("CCO", tmp_path/"ethanol.sdf", n_conformers=2)
    assert r["conformers"] >= 1
    assert "force_field" in (tmp_path/"ethanol.sdf").read_text()
