"""Provenance-preserving chemical table readers and normalization.

Missing activities remain missing. Full stereochemical keys are retained;
connectivity-only keys are never promoted to full molecular identifiers.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import zipfile

csv.field_size_limit(2**31 - 1)
CORE_FIELDS = ["compound_id", "smiles", "full_inchikey", "connectivity_id", "source_id",
               "source_record_id", "name", "entity_type", "evidence_type", "source_url",
               "source_version", "license", "properties_json"]
ALIASES = {
    "smiles": ["smiles", "canonical_smiles", "isomeric_smiles", "canonicalsmiles", "canonical smiles",
               "isomericsmiles", "structure_smiles", "compound_smiles", "rdkit_smiles", "np_smiles", "canonical_smile"],
    "full_inchikey": ["full_inchikey", "inchikey", "inchi_key", "standard_inchi_key", "standardinchikey",
                      "structure_inchikey", "compound_inchikey", "inchikey_2d", "np_inchikey"],
    "source_record_id": ["source_record_id", "identifier", "coconut_id", "npaid", "npa_id", "npc_id", "chebi id", "chebi_id",
                         "np_id", "lotus_id", "molecule_id", "structure_wikidata", "wikidata_id", "compound_id", "cid", "id"],
    "name": ["name", "iupac_name", "compound_name", "traditional_name", "pref_name", "np_name", "structure_nametraditional", "structure_nameiupac"],
}


def _first(row, candidates, default=""):
    for key in candidates:
        value = row.get(key)
        if value is not None and str(value).strip() not in {"", "nan", "NaN", "None", "null"}:
            return str(value).strip()
    return default


def normalize_record(row, source_id="unknown", source_version="", license_id="", source_url=""):
    lowered = {str(k).strip().lower().replace("-", "_"): v for k, v in row.items() if k is not None}
    smiles = _first(lowered, ALIASES["smiles"])
    record_id = _first(lowered, ALIASES["source_record_id"])
    if not record_id:
        material = smiles or json.dumps(row, sort_keys=True, ensure_ascii=False)
        record_id = "sha256_" + hashlib.sha256(material.encode()).hexdigest()[:20]
    key = _first(lowered, ALIASES["full_inchikey"])
    if not re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", key):
        key = ""
    current_source = _first(lowered, ["source_id"], source_id)
    compound_id = record_id if record_id.startswith(current_source + ":") else current_source + ":" + record_id
    if "source_record_id" in lowered and "source_id" in lowered:
        compound_id = _first(lowered, ["compound_id"], compound_id)
    return {"compound_id": compound_id, "smiles": smiles, "full_inchikey": key,
            "connectivity_id": key[:14] if key else "", "source_id": current_source,
            "source_record_id": record_id, "name": _first(lowered, ALIASES["name"]),
            "entity_type": _first(lowered, ["entity_type"], "unspecified_compound_record"),
            "evidence_type": _first(lowered, ["evidence_type"], "source_database_record"),
            "source_url": _first(lowered, ["source_url"], source_url),
            "source_version": _first(lowered, ["source_version"], source_version),
            "license": _first(lowered, ["license", "license_id"], license_id),
            "properties_json": json.dumps(row, ensure_ascii=False, separators=(",", ":"))}


@contextmanager
def _open_text(path, member=None):
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if not n.endswith("/") and
                     Path(n).suffix.lower() in {".csv", ".tsv", ".txt", ".jsonl", ".ndjson", ".json", ".smi"}]
            selected = member or (names[0] if len(names) == 1 else None)
            if not selected:
                raise ValueError(f"ZIP contains {len(names)} tabular candidates; specify member: {names[:10]}")
            if selected not in archive.namelist():
                raise ValueError("Requested ZIP member missing")
            with archive.open(selected) as binary:
                with io.TextIOWrapper(binary, encoding="utf-8-sig", errors="strict", newline="") as handle:
                    yield handle, selected
    elif path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8-sig", errors="strict", newline="") as handle:
            yield handle, path.stem
    else:
        with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
            yield handle, path.name


def _json_records(handle):
    """Stream a top-level JSON array without loading large compound annotations."""
    decoder = json.JSONDecoder()
    buffer, position, eof, started = "", 0, False, False
    while True:
        if position >= len(buffer) and not eof:
            buffer = handle.read(1024 * 1024)
            position, eof = 0, not bool(buffer)
        while position < len(buffer) and buffer[position].isspace():
            position += 1
        if not started:
            if position >= len(buffer):
                if eof:
                    return
                continue
            if buffer[position] != "[":
                obj = json.loads(buffer[position:] + handle.read())
                if isinstance(obj, dict):
                    yield obj
                    return
                raise ValueError("JSON must contain objects or an array of objects")
            position += 1
            started = True
            continue
        if position >= len(buffer):
            if eof:
                raise ValueError("Truncated JSON array")
            continue
        if buffer[position] == "]":
            return
        if buffer[position] == ",":
            position += 1
            continue
        try:
            obj, end = decoder.raw_decode(buffer, position)
        except json.JSONDecodeError:
            if eof:
                raise
            block = handle.read(1024 * 1024)
            eof = not bool(block)
            buffer, position = buffer[position:] + block, 0
            continue
        if not isinstance(obj, dict):
            raise ValueError("JSON array elements must be objects")
        position = end
        yield obj


def _iter_sdf(path):
    try:
        from rdkit import Chem, RDLogger, rdBase
    except ImportError as exc:
        raise RuntimeError("SDF input requires optional RDKit") from exc
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    with opener(path, "rb") as handle:
        # Avoid flooding logs; per-record failures are preserved in counts/records.
        with rdBase.BlockLogs():
            supplier = Chem.ForwardSDMolSupplier(handle, sanitize=True, removeHs=False)
            for index, molecule in enumerate(supplier, 1):
                if molecule is None:
                    yield {"source_record_id": f"sdf_row_{index}", "_parse_error": "RDKit could not parse SDF record"}
                    continue
                record = molecule.GetPropsAsDict(includePrivate=False, includeComputed=False)
                record.setdefault("source_record_id", molecule.GetProp("_Name") if molecule.HasProp("_Name") else f"sdf_row_{index}")
                try:
                    record["smiles"] = Chem.MolToSmiles(molecule, isomericSmiles=True)
                except Exception as exc:
                    record["_parse_error"] = "SMILES conversion failed: " + type(exc).__name__
                    yield record
                    continue
                try:
                    record["full_inchikey"] = Chem.MolToInchiKey(molecule)
                except Exception as exc:
                    record["full_inchikey"] = ""
                    record["_identifier_error"] = "InChIKey conversion failed: " + type(exc).__name__
                record["structure_transform"] = f"RDKit {rdBase.rdkitVersion}: SDF to isomeric SMILES and full InChIKey"
                yield record


def iter_records(path, format=None, member=None):
    path = Path(path)
    bare_name = path.stem if path.suffix.lower() == ".gz" else path.name
    if (format or Path(bare_name).suffix.lstrip(".")).lower() in {"sdf", "sd"}:
        yield from _iter_sdf(path)
        return
    with _open_text(path, member) as (handle, name):
        kind = (format or Path(name).suffix.lstrip(".")).lower()
        if kind in {"jsonl", "ndjson"}:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        elif kind == "json":
            yield from _json_records(handle)
        elif kind in {"smi", "smiles"}:
            for index, line in enumerate(handle, 1):
                if line.strip() and not line.lstrip().startswith("#"):
                    parts = line.strip().split(maxsplit=1)
                    yield {"smiles": parts[0], "identifier": parts[1] if len(parts) > 1 else str(index)}
        else:
            first = handle.readline()
            if not first:
                return
            delimiter = "\t" if "\t" in first else ","
            # chain avoids seeking on decompression streams and permits large files.
            from itertools import chain
            yield from csv.DictReader(chain([first], handle), delimiter=delimiter)


def export_records(inputs, output, format=None, *, source_id="unknown", source_version="",
                   license_id="", source_url="", limit=None, member=None,
                   property_columns=None, deduplicate=False, include_properties=True):
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    format = (format or output.suffix.lstrip(".")).lower()
    if format not in {"csv", "jsonl"}:
        raise ValueError("Output format must be CSV or JSONL")
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    property_columns = list(property_columns or [])
    if set(property_columns) & set(CORE_FIELDS):
        raise ValueError("Property columns may not replace core fields")
    temporary = output.with_name(output.name + ".part")
    counts = {"read": 0, "written": 0, "missing_smiles": 0, "exact_duplicates": 0}
    seen = set()
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CORE_FIELDS + property_columns) if format == "csv" else None
            if writer:
                writer.writeheader()
            for path in inputs:
                for row in iter_records(path, member=member):
                    if limit is not None and counts["written"] >= limit:
                        break
                    counts["read"] += 1
                    record = normalize_record(row, source_id, source_version, license_id, source_url)
                    if not include_properties:
                        record["properties_json"] = ""
                    if not record["smiles"]:
                        counts["missing_smiles"] += 1
                        continue
                    identity = record["full_inchikey"] or record["smiles"]
                    if deduplicate and identity in seen:
                        counts["exact_duplicates"] += 1
                        continue
                    if deduplicate:
                        seen.add(identity)
                    for column in property_columns:
                        record[column] = row.get(column, "")
                    if writer:
                        writer.writerow(record)
                    else:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts["written"] += 1
                if limit is not None and counts["written"] >= limit:
                    break
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {**counts, "output": str(output.resolve()), "format": format,
            "deduplication": "exact full InChIKey, otherwise exact SMILES" if deduplicate else "none; provenance rows retained",
            "structure_validation": "Text structures preserved; SDF parsed by RDKit; no cross-source chemical standardization"}


def merge_normalized_csvs(inputs, output):
    """Full-key structure union. Raw occurrence tables remain separate evidence."""
    inputs = [Path(p) for p in inputs]
    index, input_rows = {}, 0
    for path in inputs:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = row.get("full_inchikey", "")
                if not re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", key) or not row.get("smiles"):
                    raise ValueError(f"Merge requires full InChIKey and SMILES: {path}")
                source, license_id = row.get("source_id", path.stem), row.get("license", "")
                if key not in index:
                    index[key] = (source, license_id)
                else:
                    sources, licenses = index[key]
                    if source not in sources.split("|"):
                        sources += "|" + source
                    if license_id and license_id not in licenses.split("|"):
                        licenses = licenses + "|" + license_id if licenses else license_id
                    index[key] = (sources, licenses)
                input_rows += 1
    unique = len(index)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    fields = ["compound_id", "smiles", "full_inchikey", "connectivity_id", "source_ids",
              "preferred_source", "source_record_id", "name", "license"]
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for path in inputs:
                with path.open(encoding="utf-8", newline="") as source_handle:
                    for row in csv.DictReader(source_handle):
                        key = row["full_inchikey"]
                        labels = index.pop(key, None)
                        if labels is None:
                            continue
                        writer.writerow({"compound_id": "NPFM:" + key, "smiles": row["smiles"],
                                         "full_inchikey": key, "connectivity_id": key[:14],
                                         "source_ids": labels[0], "preferred_source": row.get("source_id", path.stem),
                                         "source_record_id": row.get("source_record_id", ""),
                                         "name": row.get("name", ""), "license": labels[1]})
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"input_rows": input_rows, "written": unique, "duplicate_rows": input_rows - unique,
            "output": str(output.resolve()), "identity": "complete 27-character InChIKey",
            "representative_smiles": "first input record", "provenance": "all source IDs retained; raw occurrences remain separate"}


def export_occurrence_map(sources, output):
    """Compact lossless row locators plus explicit LOTUS occurrence evidence.

    Each source specifies path, source_id, key_column, id_column, relative_path.
    Original files must be retained; the map is not a unique experiment count.
    """
    from .download import sha256_file
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    provenance, total = [], 0
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=5) as handle:
            for source in sources:
                path, count = Path(source["path"]), 0
                for count, row in enumerate(iter_records(path), 1):
                    evidence = {k: v for k, v in row.items() if k and
                                (k.startswith("organism_") or k.startswith("reference_") or k == "manual_validation") and v != ""}
                    record = {"full_inchikey": row.get(source["key_column"], ""),
                              "source_id": source["source_id"], "source_record_id": row.get(source["id_column"], ""),
                              "raw_file": source["relative_path"], "data_row": count}
                    if evidence:
                        record["occurrence_evidence"] = evidence
                    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                total += count
                provenance.append({"file": source["relative_path"], "sha256": sha256_file(path),
                                   "license": source.get("license"), "record_count": count})
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"output": str(output.resolve()), "bytes": output.stat().st_size, "sha256": sha256_file(output),
            "map_records": total, "sources": provenance, "row_index": "one-based data row excluding header",
            "semantics": "Complete row map for acquired exports; overlapping source tables remain distinct, not unique experiments",
            "coconut_scope": "COCONUT lite lacks original organism/publication occurrences; no missing evidence is invented",
            "all_original_fields": "Retained raw file + data_row locates fields omitted from compact map"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("input")
    inspect.add_argument("--member")
    inspect.add_argument("--count", action="store_true")
    normalize = sub.add_parser("normalize")
    normalize.add_argument("inputs", nargs="+")
    normalize.add_argument("--output", required=True)
    normalize.add_argument("--format", choices=["csv", "jsonl"])
    normalize.add_argument("--source-id", default="unknown")
    normalize.add_argument("--source-version", default="")
    normalize.add_argument("--license", default="")
    normalize.add_argument("--source-url", default="")
    normalize.add_argument("--member")
    normalize.add_argument("--limit", type=int)
    normalize.add_argument("--properties", default="")
    normalize.add_argument("--deduplicate", action="store_true")
    normalize.add_argument("--compact", action="store_true")
    merge = sub.add_parser("merge")
    merge.add_argument("inputs", nargs="+")
    merge.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        iterator = iter_records(args.input, member=args.member)
        first = next(iterator, None)
        count = (int(first is not None) + sum(1 for _ in iterator)) if args.count else None
        print(json.dumps({"columns": list(first) if first else [], "first_row": first, "record_count": count}, ensure_ascii=False, indent=2))
    elif args.command == "merge":
        print(json.dumps(merge_normalized_csvs(args.inputs, args.output), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(export_records(args.inputs, args.output, args.format, source_id=args.source_id,
                                        source_version=args.source_version, license_id=args.license,
                                        source_url=args.source_url, limit=args.limit, member=args.member,
                                        property_columns=[x for x in args.properties.split(",") if x],
                                        deduplicate=args.deduplicate, include_properties=not args.compact), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
