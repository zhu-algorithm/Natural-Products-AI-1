import csv
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from npfm.np_gpt.corpus import prepare_corpus, split_for_connectivity, _token_dtype
from npfm.tokenizer import SmilesTokenizer


def source_key(index):
    letters = []
    for _ in range(14):
        letters.append(chr(65 + index % 26))
        index //= 26
    return "".join(reversed(letters)) + "-UHFFFAOYSA-N"


def keys_for_split(part, count=1):
    keys = []
    for i in range(1, 10000):
        key = source_key(i)
        if split_for_connectivity(key[:14]) == part:
            keys.append(key)
            if len(keys) == count:
                return keys
    raise AssertionError("Could not find deterministic fixture keys")


def write_csv(path, rows):
    fields = ["compound_id", "smiles", "full_inchikey", "source_ids", "license", "custom_note"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, (smiles, key) in enumerate(rows, 1):
            writer.writerow({"compound_id": f"fixture:{i}", "smiles": smiles, "full_inchikey": key,
                             "source_ids": "coconut|lotus", "license": "CC0-1.0|CC-BY-4.0",
                             "custom_note": "source field must survive"})


def test_hash_partition_is_stable_and_groups_stereochemistry():
    for i in range(200):
        key = source_key(i)
        first = split_for_connectivity(key[:14], seed="fixed")
        stereo_variant = key[:14] + "-REOHCLBHSA-N"
        assert first == split_for_connectivity(stereo_variant[:14], seed="fixed")
    assert {split_for_connectivity(source_key(i)[:14]) for i in range(1000)} == {"train", "val", "test"}
    with pytest.raises(ValueError):
        split_for_connectivity("short")
    with pytest.raises(ValueError):
        split_for_connectivity(source_key(1)[:14], val_fraction=.8, test_fraction=.3)


def test_end_to_end_audit_vocabulary_no_truncation_and_mmap_round_trip(tmp_path):
    pytest.importorskip("rdkit")
    train = keys_for_split("train", 4)
    val, test = keys_for_split("val")[0], keys_for_split("test")[0]
    original = "C[C@H](O)C(=O)O"
    rows = [(original, train[0]), (original, train[0]),
            ("[13CH3]" + "C" * 40, train[1]), ("CC O", train[2]),
            ("C1CC", train[3]), ("[Xe]", val), ("N", test), ("CO", "invalid-key")]
    source, output = tmp_path / "source.csv", tmp_path / "corpus"
    write_csv(source, rows)
    manifest = prepare_corpus(source, output, max_seq_length=32, workers=1)
    assert manifest["source"]["records"] == len(rows)
    assert manifest["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["unique_full_inchikey_records"] == 6
    assert manifest["unassigned_records"] == 1
    assert manifest["split_policy"]["scaffold_split"] is False
    assert manifest["split_policy"]["source_connectivity_cross_split_groups"] == 0
    assert manifest["tokenizer"]["fitted_on"] == "eligible_train_only"
    tokenizer = SmilesTokenizer.load(output / manifest["tokenizer"]["path"])
    assert "[Xe]" not in tokenizer.tokens
    assert "[13CH3]" not in tokenizer.tokens
    assert manifest["splits"]["train"]["unk_tokens"] == 0
    assert manifest["splits"]["val"]["unk_tokens"] == 1
    with sqlite3.connect(output / "metadata.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM molecules").fetchone()[0] == len(rows)
        assert db.execute("SELECT duplicate_of FROM molecules WHERE id=2").fetchone()[0] == 1
        long = db.execute("SELECT included,token_length,smiles,raw_row_json FROM molecules WHERE id=3").fetchone()
        assert long[0] == 0 and long[1] > 32 and long[2] == rows[2][0]
        assert json.loads(long[3])["custom_note"] == "source field must survive"
        assert db.execute("SELECT included FROM molecules WHERE id=4").fetchone()[0] == 0
        assert db.execute("SELECT chemistry_status FROM molecules WHERE id=5").fetchone()[0] == "invalid"
        assert db.execute("SELECT source_ids,license FROM molecules WHERE id=1").fetchone() == (
            "coconut|lotus", "CC0-1.0|CC-BY-4.0")
        for part in ["train", "val", "test"]:
            info = manifest["splits"][part]
            offsets = np.load(output / info["offsets_path"], mmap_mode="r")
            row_ids = np.load(output / info["row_ids_path"], mmap_mode="r")
            tokens = np.memmap(output / info["tokens_path"], dtype=manifest["token_dtype_str"], mode="r")
            assert offsets.shape == (info["records"] + 1,)
            assert row_ids.shape == (info["records"],)
            assert int(offsets[-1]) == info["tokens"]
            for i, row_id in enumerate(row_ids):
                ids = tokens[offsets[i]:offsets[i + 1]]
                assert ids[0] == tokenizer.bos_id and ids[-1] == tokenizer.eos_id
                record = db.execute("SELECT smiles,split,included FROM molecules WHERE id=?", (int(row_id),)).fetchone()
                assert record[1:] == (part, 1)
                if part == "train":
                    assert tokenizer.decode(ids) == record[0] == original
    for name, info in manifest["files"].items():
        assert (output / name).stat().st_size == info["bytes"]
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == info["sha256"]


def test_atom_budget_and_empty_splits_preserve_audit(tmp_path):
    pytest.importorskip("rdkit")
    source, output = tmp_path / "input.csv", tmp_path / "out"
    write_csv(source, [("CC", source_key(1)), ("CCO", source_key(2))])
    manifest = prepare_corpus(source, output, max_atoms=2, val_fraction=0, test_fraction=0, workers=1)
    assert manifest["splits"]["train"]["records"] == 1
    assert manifest["exclusion_counts"]["over_max_atoms"] == 1
    for part in ["val", "test"]:
        assert (output / f"{part}.tokens.bin").stat().st_size == 0
        assert np.load(output / f"{part}.offsets.npy").tolist() == [0]
        assert np.load(output / f"{part}.row_ids.npy").shape == (0,)
    with sqlite3.connect(output / "metadata.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM molecules").fetchone()[0] == 2
        assert db.execute("SELECT canonical_isomeric_smiles FROM molecules WHERE id=2").fetchone()[0] == "CCO"


def test_unvalidated_mode_does_not_claim_chemistry_validation(tmp_path):
    source, output = tmp_path / "input.csv", tmp_path / "out"
    write_csv(source, [("C1CC", source_key(1))])
    manifest = prepare_corpus(source, output, validate_chemistry=False, val_fraction=0, test_fraction=0, workers=1)
    assert manifest["chemistry"]["validated"] is False
    assert manifest["chemistry"]["canonical_cross_split_audit"]["status"] == "not_evaluated"
    assert manifest["splits"]["train"]["records"] == 1
    with sqlite3.connect(output / "metadata.sqlite") as db:
        assert db.execute("SELECT canonical_isomeric_smiles FROM molecules").fetchone()[0] is None


def test_source_key_anomaly_canonical_overlap_is_disclosed(tmp_path):
    pytest.importorskip("rdkit")
    source, output = tmp_path / "input.csv", tmp_path / "out"
    train, val = keys_for_split("train")[0], keys_for_split("val")[0]
    write_csv(source, [("CCO", train), ("OCC", val)])
    manifest = prepare_corpus(source, output, workers=1)
    audit = manifest["chemistry"]["canonical_cross_split_audit"]
    assert audit["cross_split_groups_all_records"] == 1
    assert audit["cross_split_groups_eligible_records"] == 1
    assert "source_reported" in manifest["identifiers"]["full_inchikey"]
    with sqlite3.connect(output / "metadata.sqlite") as db:
        assert db.execute("SELECT full_inchikey FROM molecules ORDER BY id").fetchall() == [(train,), (val,)]


def test_token_width_boundary_and_existing_output_guard(tmp_path):
    assert _token_dtype(65536) == np.dtype("<u2")
    assert _token_dtype(65537) == np.dtype("<u4")
    with pytest.raises(ValueError):
        _token_dtype(2**32 + 1)
    source, output = tmp_path / "input.csv", tmp_path / "out"
    write_csv(source, [("CC", source_key(1))])
    output.mkdir()
    (output / "keep.txt").write_text("previous user artifact")
    with pytest.raises(FileExistsError):
        prepare_corpus(source, output, validate_chemistry=False, workers=1)
    assert (output / "keep.txt").read_text() == "previous user artifact"
