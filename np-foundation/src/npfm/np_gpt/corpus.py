"""Build an auditable, bounded-memory SMILES language-model corpus.

Source SMILES are tokenized unchanged. RDKit canonical isomeric SMILES are stored
separately for reference membership. Splits group source-reported InChIKey
connectivity blocks; this is explicitly not a Bemis--Murcko scaffold split.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, nullcontext
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import sqlite3
import sys
import tempfile
import time

import numpy as np

from ..tokenizer import SmilesTokenizer

SPLITS = ("train", "val", "test")
FULL_KEY = re.compile(r"[A-Z]{14}-[A-Z]{10}-[A-Z]\Z")
CONNECTIVITY_KEY = re.compile(r"[A-Z]{14}\Z")
BATCH_SIZE = 4096


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_info(path):
    return {"bytes": Path(path).stat().st_size, "sha256": _sha256(path)}


def _validate_fractions(val_fraction, test_fraction):
    if not (math.isfinite(val_fraction) and math.isfinite(test_fraction)):
        raise ValueError("Split fractions must be finite")
    if min(val_fraction, test_fraction) < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("Nonnegative val/test fractions must leave a positive training fraction")


def split_for_connectivity(connectivity_id, *, seed="np-gpt-v1", val_fraction=.05, test_fraction=.05):
    """Stable assignment from SHA256(seed + NUL + 14-letter connectivity block)."""
    _validate_fractions(val_fraction, test_fraction)
    if not CONNECTIVITY_KEY.fullmatch(connectivity_id):
        raise ValueError("A complete 14-letter connectivity block is required")
    digest = hashlib.sha256((str(seed) + "\0" + connectivity_id).encode("utf-8")).digest()
    number = int.from_bytes(digest, "big")
    scale = 1 << 256
    train_threshold = int((1 - val_fraction - test_fraction) * scale)
    val_threshold = int((1 - test_fraction) * scale)
    return "train" if number < train_threshold else "val" if number < val_threshold else "test"


def _token_dtype(vocabulary_size):
    if vocabulary_size <= 0 or vocabulary_size > 2**32:
        raise ValueError("Vocabulary size must fit unsigned 32-bit token IDs")
    return np.dtype("<u2") if vocabulary_size <= 2**16 else np.dtype("<u4")


def _length_summary(histogram):
    count = sum(histogram.values())
    if not count:
        return {"records": 0, "min": None, "max": None, "mean": None,
                "p50": None, "p90": None, "p95": None, "p99": None, "p99_9": None}
    ordered = sorted(histogram.items())
    result = {"records": count, "min": ordered[0][0], "max": ordered[-1][0],
              "mean": sum(length * n for length, n in ordered) / count}
    for label, quantile in [("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99), ("p99_9", .999)]:
        rank, cumulative = max(1, math.ceil(quantile * count)), 0
        for length, n in ordered:
            cumulative += n
            if cumulative >= rank:
                result[label] = length
                break
    return result


def _initialize_chemistry_worker():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def _analyze_smiles(arguments):
    smiles, validate_chemistry = arguments
    result = {"token_length": None, "lexical_error": None, "canonical_isomeric_smiles": None,
              "atom_count": None, "chemistry_error": None,
              "chemistry_status": "not_checked" if not validate_chemistry else "invalid"}
    try:
        result["token_length"] = len(SmilesTokenizer.tokenize(smiles)) + 2
    except (TypeError, ValueError) as exc:
        result["lexical_error"] = str(exc)[:300]
    if validate_chemistry:
        from rdkit import Chem, rdBase
        try:
            with rdBase.BlockLogs():
                molecule = Chem.MolFromSmiles(smiles)
                if molecule is None:
                    result["chemistry_error"] = "RDKit rejected source SMILES"
                else:
                    result["canonical_isomeric_smiles"] = Chem.MolToSmiles(
                        molecule, canonical=True, isomericSmiles=True)
                    result["atom_count"] = molecule.GetNumAtoms()
                    result["chemistry_status"] = "valid"
        except Exception as exc:
            result["chemistry_error"] = type(exc).__name__ + ": " + str(exc)[:250]
    return result


def _batches(iterator, size=BATCH_SIZE):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


SCHEMA = """
CREATE TABLE molecules (
    id INTEGER PRIMARY KEY,
    input_row INTEGER NOT NULL,
    compound_id TEXT,
    smiles TEXT NOT NULL,
    canonical_isomeric_smiles TEXT,
    full_inchikey TEXT,
    full_key_valid INTEGER NOT NULL,
    connectivity_id TEXT,
    source_ids TEXT,
    preferred_source TEXT,
    source_record_id TEXT,
    name TEXT,
    license TEXT,
    split TEXT,
    included INTEGER NOT NULL,
    duplicate_of INTEGER,
    status TEXT NOT NULL,
    exclusion_reason TEXT,
    exclusion_reasons_json TEXT NOT NULL,
    token_length INTEGER,
    atom_count INTEGER,
    lexical_error TEXT,
    chemistry_status TEXT NOT NULL,
    chemistry_error TEXT,
    unk_count INTEGER,
    sequence_index INTEGER,
    raw_row_json TEXT NOT NULL
);
"""


def _audit_count(connection, where, field):
    query = ("SELECT COUNT(*) FROM (SELECT " + field + " FROM molecules WHERE " + where
             + " GROUP BY " + field + " HAVING COUNT(DISTINCT split)>1)")
    return connection.execute(query).fetchone()[0]


def prepare_corpus(input_csv, output_dir, *, seed="np-gpt-v1", max_seq_length=512,
                   max_atoms=None, val_fraction=.05, test_fraction=.05,
                   workers=4, validate_chemistry=True):
    """Prepare full sequences without silent truncation or vocabulary leakage.

    All input rows remain in metadata.sqlite, including duplicate identifiers,
    over-budget molecules and parse failures. Only eligible first occurrences of
    complete source-reported InChIKeys enter token streams. Vocabulary is fitted
    exclusively on those eligible training rows. Unknown validation/test tokens
    are retained as UNK and explicitly counted.

    Binary token files are little-endian uint16/uint32. NPY offsets are int64 of
    length N+1; NPY row IDs are int64 of length N and reference molecules.id.
    The max sequence length includes BOS and EOS. No sequence is truncated.

    workers is capped at four. Chemistry may be disabled explicitly, in which
    case canonical membership is unavailable and max_atoms is prohibited.
    """
    _validate_fractions(val_fraction, test_fraction)
    if not isinstance(max_seq_length, int) or max_seq_length < 3:
        raise ValueError("max_seq_length must permit BOS, a molecular token, and EOS")
    if max_atoms is not None and (not isinstance(max_atoms, int) or max_atoms < 1):
        raise ValueError("max_atoms must be a positive integer or None")
    if max_atoms is not None and not validate_chemistry:
        raise ValueError("max_atoms requires chemistry validation")
    if not isinstance(workers, int) or not 1 <= workers <= 4:
        raise ValueError("Use between one and four worker processes")
    rdkit_version = None
    if validate_chemistry:
        try:
            from rdkit import rdBase
            rdkit_version = rdBase.rdkitVersion
        except ImportError as exc:
            raise RuntimeError("Chemistry validation requires RDKit; disable it explicitly if appropriate") from exc
    input_csv, output_dir = Path(input_csv).resolve(), Path(output_dir).resolve()
    if not input_csv.is_file():
        raise FileNotFoundError(input_csv)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Output directory must be absent or empty; existing corpora are never overwritten")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + output_dir.name + ".preparing-", dir=output_dir.parent))
    started = time.monotonic()
    source_stat = input_csv.stat()
    source_sha = _sha256(input_csv)
    connection = None
    split_stats = {part: {"input_records": 0, "unique_records": 0, "duplicate_records": 0,
                          "eligible_records": 0, "excluded_records": 0, "lexical_failure_records": 0,
                          "chemistry_failure_records": 0, "over_seq_length_records": 0,
                          "over_atom_budget_records": 0, "unk_tokens": 0, "unk_records": 0,
                          "unk_tokens_all_lexically_valid": 0, "unk_records_all_lexically_valid": 0}
                   for part in SPLITS}
    all_lengths = {part: Counter() for part in SPLITS}
    eligible_lengths = {part: Counter() for part in SPLITS}
    exclusions, source_ids, licenses = Counter(), Counter(), Counter()
    input_records, unassigned_records = 0, 0
    seen = {}
    try:
        connection = sqlite3.connect(staging / "metadata.sqlite")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA cache_size=-65536")
        connection.execute("PRAGMA temp_store=FILE")
        connection.executescript(SCHEMA)
        placeholders = ",".join("?" for _ in range(27))
        insertion = "INSERT INTO molecules VALUES (" + placeholders + ")"
        pool_context = (ProcessPoolExecutor(max_workers=workers, initializer=_initialize_chemistry_worker)
                        if workers > 1 and validate_chemistry else nullcontext(None))
        with input_csv.open(encoding="utf-8-sig", newline="") as handle, pool_context as pool:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"smiles", "full_inchikey"} <= set(reader.fieldnames):
                raise ValueError("Input CSV requires smiles and full_inchikey columns")
            columns = list(reader.fieldnames)
            for batch in _batches(enumerate(reader, 1)):
                arguments = [(row.get("smiles") or "", validate_chemistry) for _, row in batch]
                analyzed = pool.map(_analyze_smiles, arguments, chunksize=64) if pool else map(_analyze_smiles, arguments)
                records = []
                for (input_row, row), analysis in zip(batch, analyzed):
                    input_records += 1
                    smiles = row.get("smiles") or ""
                    key = (row.get("full_inchikey") or "").strip()
                    valid_key = bool(FULL_KEY.fullmatch(key))
                    connectivity = key[:14] if valid_key else None
                    split = split_for_connectivity(connectivity, seed=seed, val_fraction=val_fraction,
                                                   test_fraction=test_fraction) if valid_key else None
                    duplicate_of = seen.get(key) if valid_key else None
                    if valid_key and duplicate_of is None:
                        seen[key] = input_row
                    reasons = []
                    if not valid_key:
                        reasons.append("invalid_full_inchikey")
                    if duplicate_of is not None:
                        reasons.append("duplicate_full_inchikey")
                    if analysis["lexical_error"]:
                        reasons.append("lexical_failure")
                    if validate_chemistry and analysis["chemistry_status"] != "valid":
                        reasons.append("chemistry_failure")
                    length = analysis["token_length"]
                    if length is not None and length > max_seq_length:
                        reasons.append("over_max_seq_length")
                    if max_atoms is not None and analysis["atom_count"] is not None and analysis["atom_count"] > max_atoms:
                        reasons.append("over_max_atoms")
                    included = not reasons
                    exclusions.update(reasons)
                    source_ids[row.get("source_ids", row.get("source_id", "")) or ""] += 1
                    licenses[row.get("license") or ""] += 1
                    if split is None:
                        unassigned_records += 1
                    else:
                        stats = split_stats[split]
                        stats["input_records"] += 1
                        stats["duplicate_records"] += int(duplicate_of is not None)
                        stats["unique_records"] += int(duplicate_of is None)
                        stats["eligible_records"] += int(included)
                        stats["excluded_records"] += int(not included)
                        stats["lexical_failure_records"] += int(bool(analysis["lexical_error"]))
                        stats["chemistry_failure_records"] += int(bool(analysis["chemistry_error"]))
                        stats["over_seq_length_records"] += int(length is not None and length > max_seq_length)
                        stats["over_atom_budget_records"] += int("over_max_atoms" in reasons)
                        if length is not None and duplicate_of is None:
                            all_lengths[split][length] += 1
                        if included:
                            eligible_lengths[split][length] += 1
                    records.append((input_row, input_row, row.get("compound_id") or "NPFM:" + key,
                                    smiles, analysis["canonical_isomeric_smiles"], key, int(valid_key), connectivity,
                                    row.get("source_ids", row.get("source_id", "")), row.get("preferred_source"),
                                    row.get("source_record_id"), row.get("name"), row.get("license"), split,
                                    int(included), duplicate_of, "included" if included else "excluded",
                                    reasons[0] if reasons else None, json.dumps(reasons), length,
                                    analysis["atom_count"], analysis["lexical_error"], analysis["chemistry_status"],
                                    analysis["chemistry_error"], None, None,
                                    json.dumps(row, ensure_ascii=False, separators=(",", ":"))))
                connection.executemany(insertion, records)
                connection.commit()
                if input_records % (BATCH_SIZE * 25) == 0:
                    print(f"corpus: audited {input_records:,} input rows", file=sys.stderr, flush=True)
        unique_records = len(seen)
        del seen
        final_stat = input_csv.stat()
        if (source_stat.st_size, source_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
            raise RuntimeError("Input CSV changed while preparing the corpus")
        if not split_stats["train"]["eligible_records"]:
            raise ValueError("No eligible training records; vocabulary cannot be fitted without leakage")
        connection.executescript("""
            CREATE INDEX idx_molecules_full_key ON molecules(full_inchikey);
            CREATE INDEX idx_molecules_canonical ON molecules(canonical_isomeric_smiles);
            CREATE INDEX idx_molecules_split_included ON molecules(split,included,id);
            CREATE INDEX idx_molecules_connectivity ON molecules(connectivity_id);
        """)
        connection.commit()
        print("corpus: fitting vocabulary on eligible training molecules only", file=sys.stderr, flush=True)
        vocabulary_input = (r[0] for r in connection.execute(
            "SELECT smiles FROM molecules WHERE split='train' AND included=1 ORDER BY id"))
        tokenizer = SmilesTokenizer.fit(vocabulary_input)
        tokenizer.save(staging / "tokenizer.json")
        dtype = _token_dtype(len(tokenizer))
        positions = dict.fromkeys(SPLITS, 0)
        token_positions = dict.fromkeys(SPLITS, 0)
        offsets, row_ids = {}, {}
        with ExitStack() as stack:
            streams = {part: stack.enter_context((staging / (part + ".tokens.bin")).open("wb", buffering=1024 * 1024))
                       for part in SPLITS}
            for part in SPLITS:
                count = split_stats[part]["eligible_records"]
                if count:
                    offsets[part] = np.lib.format.open_memmap(staging / (part + ".offsets.npy"), mode="w+",
                                                             dtype="<i8", shape=(count + 1,))
                    row_ids[part] = np.lib.format.open_memmap(staging / (part + ".row_ids.npy"), mode="w+",
                                                             dtype="<i8", shape=(count,))
                    offsets[part][0] = 0
                else:
                    np.save(staging / (part + ".offsets.npy"), np.asarray([0], dtype="<i8"))
                    np.save(staging / (part + ".row_ids.npy"), np.asarray([], dtype="<i8"))
            cursor = connection.execute("SELECT id,smiles,split,included,token_length FROM molecules ORDER BY id")
            while True:
                batch = cursor.fetchmany(BATCH_SIZE)
                if not batch:
                    break
                updates = []
                for row_id, smiles, split, included, token_length in batch:
                    if token_length is None:
                        continue
                    ids = tokenizer.encode(smiles, max_length=max_seq_length if included else None)
                    unknown = ids.count(tokenizer.unk_id)
                    if split is not None:
                        split_stats[split]["unk_tokens_all_lexically_valid"] += unknown
                        split_stats[split]["unk_records_all_lexically_valid"] += int(unknown > 0)
                    sequence_index = None
                    if included:
                        if len(ids) != token_length or len(ids) > max_seq_length:
                            raise RuntimeError("Sequence length mismatch; no truncated records are permitted")
                        sequence_index = positions[split]
                        streams[split].write(np.asarray(ids, dtype=dtype).tobytes())
                        token_positions[split] += len(ids)
                        offsets[split][sequence_index + 1] = token_positions[split]
                        row_ids[split][sequence_index] = row_id
                        positions[split] += 1
                        split_stats[split]["unk_tokens"] += unknown
                        split_stats[split]["unk_records"] += int(unknown > 0)
                    updates.append((unknown, sequence_index, row_id))
                connection.executemany("UPDATE molecules SET unk_count=?,sequence_index=? WHERE id=?", updates)
                connection.commit()
            for part in SPLITS:
                streams[part].flush()
                os.fsync(streams[part].fileno())
                if positions[part] != split_stats[part]["eligible_records"]:
                    raise RuntimeError("Written record count differs from audit selection")
                if part in offsets:
                    offsets[part].flush()
                    row_ids[part].flush()
        offsets.clear()
        row_ids.clear()
        if split_stats["train"]["unk_tokens"]:
            raise RuntimeError("Training UNK tokens indicate inconsistent vocabulary fitting")
        connectivity_overlap = _audit_count(connection, "full_key_valid=1", "connectivity_id")
        if connectivity_overlap:
            raise RuntimeError("Source connectivity groups leaked across splits")
        canonical_audit = {"status": "not_evaluated"}
        if validate_chemistry:
            where = "canonical_isomeric_smiles IS NOT NULL AND canonical_isomeric_smiles!=''"
            canonical_audit = {"status": "evaluated",
                               "cross_split_groups_all_records": _audit_count(connection, where, "canonical_isomeric_smiles"),
                               "cross_split_groups_eligible_records": _audit_count(connection, where + " AND included=1", "canonical_isomeric_smiles"),
                               "note": "Source keys are not recomputed; any canonical overlap is disclosed, not silently reassigned"}
        connection.commit()
        connection.close()
        connection = None
        files = {path.name: _file_info(path) for path in sorted(staging.iterdir()) if path.is_file()}
        split_manifest = {}
        for part in SPLITS:
            stats = split_stats[part]
            n, total = stats["eligible_records"], token_positions[part]
            split_manifest[part] = {**stats, "records": n, "tokens": total, "token_count": total,
                                    "prediction_tokens": total - n, "lexical_tokens": total - 2 * n,
                                    "tokens_path": part + ".tokens.bin", "offsets_path": part + ".offsets.npy",
                                    "row_ids_path": part + ".row_ids.npy",
                                    "unk_fraction_lexical_tokens": stats["unk_tokens"] / max(1, total - 2 * n),
                                    "lengths": {"eligible": _length_summary(eligible_lengths[part]),
                                                "all_unique_lexically_valid": _length_summary(all_lengths[part])},
                                    "eligible_length_histogram": dict(sorted(eligible_lengths[part].items()))}
        self_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
        manifest = {
            "schema_version": "1.0", "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {"path": str(input_csv), "name": input_csv.name, "bytes": source_stat.st_size,
                       "sha256": source_sha, "records": input_records, "columns": columns},
            "unique_full_inchikey_records": unique_records, "unassigned_records": unassigned_records,
            "max_seq_length": max_seq_length, "max_atoms": max_atoms,
            "token_dtype": dtype.name, "token_dtype_str": dtype.str, "offset_dtype": "int64",
            "sequence_format": "BOS + all original source-SMILES tokens + EOS; no truncation",
            "split_policy": {"method": "SHA256 connectivity-block grouping", "seed": str(seed),
                             "hash_input": "UTF-8(seed + NUL + first 14 source InChIKey letters)",
                             "hash_integer": "unsigned big-endian full SHA256 digest",
                             "fractions": {"train": 1 - val_fraction - test_fraction,
                                           "val": val_fraction, "test": test_fraction},
                             "exact_fraction_guarantee": False, "scaffold_split": False,
                             "source_connectivity_cross_split_groups": connectivity_overlap},
            "splits": split_manifest,
            "tokenizer": {"path": "tokenizer.json", **files["tokenizer.json"], "vocab_size": len(tokenizer),
                          "fitted_on": "eligible_train_only", "implementation": "npfm.tokenizer.SmilesTokenizer",
                          "special_ids": {"pad": tokenizer.pad_id, "bos": tokenizer.bos_id, "eos": tokenizer.eos_id,
                                          "unk": tokenizer.unk_id, "mask": tokenizer.mask_id}},
            "metadata": {"path": "metadata.sqlite", **files["metadata.sqlite"], "table": "molecules",
                         "row_id_column": "id", "row_id_meaning": "one-based input CSV data row",
                         "train_membership_filter": "split='train' AND included=1",
                         "all_corpus_membership_filter": "all rows, including length exclusions and duplicates",
                         "raw_audit": "raw_row_json preserves every input field; smiles preserves the original string"},
            "identifiers": {"full_inchikey": "source_reported; syntax checked; not recomputed or cross-validated",
                            "canonical_isomeric_smiles": "RDKit canonical isomeric SMILES, stored separately" if validate_chemistry else "not computed",
                            "deduplication": "first input row per exact valid complete source InChIKey; all other rows audited",
                            "caution": "Full standard InChIKeys do not distinguish all tautomeric or unspecified stereo states"},
            "chemistry": {"validated": validate_chemistry, "rdkit_version": rdkit_version,
                          "failure_policy": "exclude from token streams and retain raw audit" if validate_chemistry else "not evaluated",
                          "canonical_cross_split_audit": canonical_audit},
            "exclusion_counts": dict(exclusions), "source_ids_record_counts": dict(source_ids),
            "license_record_counts": dict(licenses), "files": files,
            "performance": {"seconds": time.monotonic() - started, "workers": workers,
                            "batch_rows": BATCH_SIZE, "self_peak_rss_bytes": self_rss,
                            "largest_child_peak_rss_bytes": child_rss,
                            "conservative_process_peak_sum_upper_bound_bytes": self_rss + workers * child_rss,
                            "rss_note": "Linux per-process high-water marks; shared pages may be counted more than once"},
            "limitations": ["Connectivity grouping is not a scaffold or publication-time split",
                            "A reference to the training partition does not prove a checkpoint saw every row",
                            "All molecular records are audit-preserved; excluded rows do not contribute training tokens",
                            "Source licenses remain attached; model commercialization is not granted by preprocessing"],
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        os.replace(staging, output_dir)
        return manifest
    except Exception:
        if connection is not None:
            connection.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", default="np-gpt-v1")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-atoms", type=int)
    parser.add_argument("--val-fraction", type=float, default=.05)
    parser.add_argument("--test-fraction", type=float, default=.05)
    parser.add_argument("--workers", type=int, default=4)
    chemistry = parser.add_mutually_exclusive_group()
    chemistry.add_argument("--validate-chemistry", dest="validate_chemistry", action="store_true")
    chemistry.add_argument("--no-validate-chemistry", dest="validate_chemistry", action="store_false")
    parser.set_defaults(validate_chemistry=True)
    args = parser.parse_args(argv)
    manifest = prepare_corpus(args.input_csv, args.output, seed=args.seed, max_seq_length=args.max_seq_length,
                              max_atoms=args.max_atoms, val_fraction=args.val_fraction, test_fraction=args.test_fraction,
                              workers=args.workers, validate_chemistry=args.validate_chemistry)
    print(json.dumps({"output": str(Path(args.output).resolve()), "source_records": manifest["source"]["records"],
                      "token_dtype": manifest["token_dtype"], "vocab_size": manifest["tokenizer"]["vocab_size"],
                      "splits": {part: {k: manifest["splits"][part][k] for k in ["records", "tokens", "unk_tokens", "lengths"]}
                                 for part in SPLITS}, "exclusions": manifest["exclusion_counts"],
                      "canonical_cross_split_audit": manifest["chemistry"]["canonical_cross_split_audit"],
                      "performance": manifest["performance"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
