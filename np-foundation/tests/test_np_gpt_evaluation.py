"""Small, explicit chemistry/reference fixtures; not generation-performance claims."""
from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

pytest.importorskip("rdkit")
from rdkit import Chem
from rdkit.Chem import inchi

from npfm.np_gpt.evaluation import evaluate_generated


def identity(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True), inchi.MolToInchiKey(molecule)


def corpus(tmp_path, entries, canonical=True, wrong_key=False):
    directory = tmp_path / "corpus"
    directory.mkdir()
    database = directory / "metadata.sqlite"
    connection = sqlite3.connect(database)
    canonical_field = ", canonical_isomeric_smiles TEXT" if canonical else ""
    connection.execute(
        "CREATE TABLE molecules (id INTEGER PRIMARY KEY, smiles TEXT, full_inchikey TEXT, "
        "split TEXT, included INTEGER" + canonical_field + ")"
    )
    for index, (smiles, split, included) in enumerate(entries, 1):
        normalized, key = identity(smiles)
        if wrong_key:
            key = identity("CCN")[1]
        values = [index, smiles, key, split, included]
        if canonical:
            values.append(normalized)
        placeholders = ",".join("?" for _ in values)
        connection.execute("INSERT INTO molecules VALUES (" + placeholders + ")", values)
    connection.execute("CREATE INDEX idx_full_key ON molecules(full_inchikey)")
    if canonical:
        connection.execute("CREATE INDEX idx_canonical ON molecules(canonical_isomeric_smiles)")
    connection.commit()
    connection.close()
    (directory / "manifest.json").write_text(json.dumps({
        "metadata": {"path": "metadata.sqlite", "sha256": hashlib.sha256(database.read_bytes()).hexdigest()},
        "training_executed": False,
    }))
    return directory


def test_denominators_duplicates_reference_and_computed_properties(tmp_path):
    directory = corpus(tmp_path, [("CCO", "train", 1), ("CCN", "val", 1), ("CCCl", "train", 0)])
    generated = [
        {"smiles": "CCO", "terminated": True},
        {"smiles": "OCC", "terminated": False},
        {"smiles": "CCN", "terminated": True},
        {"smiles": "CCC", "terminated": True},
        {"smiles": "C1CC", "terminated": True},
        {"smiles": "CCCl", "terminated": True},
    ]
    output = tmp_path / "evaluation.json"
    report = evaluate_generated(generated, directory, output)
    metrics = report["metrics"]
    assert metrics["attempts"] == 6 and metrics["valid_samples"] == 5
    assert metrics["validity_rate"] == pytest.approx(5 / 6)
    assert metrics["termination_rate"] == pytest.approx(5 / 6)
    assert metrics["valid_and_terminated_rate"] == pytest.approx(4 / 6)
    assert metrics["unique_valid_structures"] == 4
    assert metrics["uniqueness_rate"] == pytest.approx(4 / 5)
    assert metrics["training_novelty"]["rate"] == pytest.approx(3 / 4)
    assert metrics["full_corpus_novelty"]["rate"] == pytest.approx(1 / 4)
    assert metrics["training_novelty"]["denominator_unique_valid"] == 4
    assert report["samples"][1]["duplicate_of_sample_index"] == 0
    assert report["samples"][1]["valid"] and not report["samples"][1]["terminated"]
    assert report["samples"][5]["in_full_corpus"] is True
    assert report["samples"][5]["in_training_corpus"] is False  # excluded despite split=train
    assert report["samples"][0]["property_kind"] == "computed_not_experimental"
    assert report["samples"][0]["computed_properties"]["molecular_weight"] > 0
    assert report["computed_property_summary_unique_valid"]["qed"]["n"] == 4
    assert report["corpus_reference"]["manifest_hash_verified"]
    assert report["corpus_reference"]["training_reference_rows"] == 1
    assert json.loads(output.read_text())["metrics"] == metrics
    assert not output.with_name(output.name + ".part").exists()


def test_stereoisomers_and_isotopes_are_not_deduplicated(tmp_path):
    known = "N[C@@H](C)C(=O)O"
    directory = corpus(tmp_path, [(known, "train", 1), ("CCO", "train", 1)])
    report = evaluate_generated([
        {"smiles": known, "terminated": True},
        {"smiles": "N[C@H](C)C(=O)O", "terminated": True},
        {"smiles": "CCO", "terminated": True},
        {"smiles": "[13CH3]CO", "terminated": True},
    ], directory)
    metrics = report["metrics"]
    assert metrics["unique_valid_structures"] == metrics["unique_full_inchikeys"] == 4
    assert metrics["full_corpus_novelty"]["novel_unique_valid"] == 2
    assert report["samples"][0]["full_inchikey"] != report["samples"][1]["full_inchikey"]
    assert report["samples"][2]["full_inchikey"] != report["samples"][3]["full_inchikey"]


def test_unknown_reference_and_termination_are_not_invented():
    report = evaluate_generated(["CCO", "OCC"], None)
    metrics = report["metrics"]
    assert metrics["termination_rate"] is None
    assert metrics["valid_and_terminated_rate"] is None
    assert metrics["training_novelty"]["rate"] is None
    assert metrics["training_novelty"]["unknown_unique_valid"] == 1
    assert metrics["training_novelty"]["coverage"] == 0
    assert report["samples"][0]["in_full_corpus"] is None


def test_invalid_inputs_are_counted_not_repaired_and_declared_validity_is_ignored():
    report = evaluate_generated([
        {"smiles": "CCO ethanol", "terminated": True},
        {"smiles": "CCO ", "terminated": True},
        {"smiles": "", "terminated": True},
        {"smiles": "C(C)(C)(C)(C)C", "terminated": True, "valid": True},
    ], None)
    assert report["metrics"]["attempts"] == 4
    assert report["metrics"]["validity_rate"] == 0
    assert report["metrics"]["uniqueness_rate"] is None
    assert all(row["invalid_reason"] for row in report["samples"])
    assert all(row["in_full_corpus"] is None for row in report["samples"])


def test_canonical_index_can_match_when_source_key_disagrees(tmp_path):
    directory = corpus(tmp_path, [("CCO", "train", 1)], wrong_key=True)
    sample = evaluate_generated([{"smiles": "OCC", "terminated": True}], directory)["samples"][0]
    assert sample["reference_matches"]["full_inchikey"]["full"] is False
    assert sample["reference_matches"]["canonical_isomeric_smiles"]["full"] is True
    assert sample["in_full_corpus"] is True
    assert sample["in_training_corpus"] is True


def test_raw_smiles_column_is_not_misrepresented_as_canonical_index(tmp_path):
    directory = corpus(tmp_path, [("CCO", "train", 1)], canonical=False)
    report = evaluate_generated([{"smiles": "OCC", "terminated": True}], directory)
    assert not report["corpus_reference"]["canonical_matching_available"]
    assert set(report["samples"][0]["reference_matches"]) == {"full_inchikey"}
    assert report["samples"][0]["in_full_corpus"] is True


def test_empty_canonical_column_is_not_called_a_canonical_index(tmp_path):
    directory = corpus(tmp_path, [("CCO", "train", 1)])
    database = directory / "metadata.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("UPDATE molecules SET canonical_isomeric_smiles=NULL")
    connection.commit()
    connection.close()
    manifest = {"metadata": {"path": "metadata.sqlite", "sha256": hashlib.sha256(database.read_bytes()).hexdigest()}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    report = evaluate_generated([{"smiles": "OCC", "terminated": True}], directory)
    assert report["corpus_reference"]["canonical_column_present"]
    assert not report["corpus_reference"]["canonical_matching_available"]
    assert report["corpus_reference"]["canonical_reference_coverage"] == 0
    assert set(report["samples"][0]["reference_matches"]) == {"full_inchikey"}


def test_changed_reference_hash_is_rejected(tmp_path):
    directory = corpus(tmp_path, [("CCO", "train", 1)])
    connection = sqlite3.connect(directory / "metadata.sqlite")
    connection.execute("UPDATE molecules SET included=0")
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="SHA-256"):
        evaluate_generated([], directory)


def test_termination_alias_and_conflicting_or_string_flags():
    report = evaluate_generated([{"smiles": "CCO", "terminated_with_eos": True}], None)
    assert report["metrics"]["termination_rate"] == 1
    with pytest.raises(ValueError, match="Conflicting"):
        evaluate_generated([{"smiles": "CCO", "terminated": True, "terminated_with_eos": False}], None)
    with pytest.raises(ValueError, match="bool"):
        evaluate_generated([{"smiles": "CCO", "terminated": "false"}], None)


def test_empty_attempts_have_undefined_rates_and_json_safe_output(tmp_path):
    report = evaluate_generated([], None, tmp_path / "empty.json")
    assert report["metrics"]["attempts"] == 0
    assert report["metrics"]["validity_rate"] is None
    assert report["metrics"]["full_corpus_novelty"]["rate"] is None
    assert report["samples"] == []
    json.dumps(report, allow_nan=False)


def test_valid_wildcards_and_fragments_are_flagged_not_clinically_endorsed():
    report = evaluate_generated([
        {"smiles": "*", "terminated": True},
        {"smiles": "CCO.O", "terminated": True},
    ], None)
    assert report["metrics"]["valid_samples"] == 2
    assert "wildcard_atom" in report["samples"][0]["quality_flags"]
    assert "disconnected_fragments" in report["samples"][1]["quality_flags"]
    assert report["samples"][0]["identity_method"] == "canonical_isomeric_smiles"
    assert any("clinical" in limitation for limitation in report["limitations"])


def test_missing_or_incompatible_reference_is_not_silently_ignored(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate_generated(["CCO"], tmp_path / "absent")
    directory = tmp_path / "incompatible"
    directory.mkdir()
    connection = sqlite3.connect(directory / "metadata.sqlite")
    connection.execute("CREATE TABLE molecules (smiles TEXT)")
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="requires"):
        evaluate_generated(["CCO"], directory)
