"""Generated-structure metrics with explicit denominators and reference scope.

RDKit validity, exact novelty and calculated descriptors are engineering/chemical
checks. They do not establish natural occurrence, synthesis, drug suitability or
clinical benefit. No model weights or experimental labels are used here.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from statistics import mean, median
from typing import Any


FULL_KEY = re.compile(r"[A-Z]{14}-[A-Z]{10}-[A-Z]")
PROPERTY_UNITS = {
    "molecular_weight": "g/mol; RDKit average atomic masses",
    "crippen_logp": "dimensionless calculated logP",
    "tpsa": "angstrom^2; calculated topological polar surface area",
    "hbond_donors": "count",
    "hbond_acceptors": "count",
    "rotatable_bonds": "count",
    "rings": "count",
    "heavy_atoms": "count",
    "qed": "dimensionless calculated QED descriptor; not a probability",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


class _CorpusReference:
    """Read-only exact lookups; never infer canonicality from a raw SMILES column."""

    def __init__(self, corpus_dir: str | Path):
        root = Path(corpus_dir).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Corpus reference does not exist: {root}")
        manifest_path = root / "manifest.json" if root.is_dir() else root.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        metadata = manifest.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Corpus manifest metadata must be an object")
        if root.is_file():
            database = root
        else:
            name = metadata.get("path", "metadata.sqlite")
            database = Path(name)
            if not database.is_absolute():
                database = root / database
        database = database.resolve()
        if not database.is_file():
            raise FileNotFoundError(f"Corpus SQLite index not found: {database}")
        wal = database.with_name(database.name + "-wal")
        if wal.exists() and wal.stat().st_size:
            raise ValueError("Finalize/checkpoint the SQLite WAL before evaluating a frozen corpus")
        actual_hash = _sha256(database)
        expected_hash = metadata.get("sha256")
        if expected_hash is not None and actual_hash != expected_hash:
            raise ValueError("Corpus SQLite SHA-256 differs from its manifest")
        self.connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
        try:
            self.connection.execute("PRAGMA query_only=ON")
            columns = {row[1] for row in self.connection.execute("PRAGMA table_info(molecules)")}
            required = {"full_inchikey", "split", "included"}
            if not required.issubset(columns):
                raise ValueError("Corpus molecules table requires full_inchikey, split and included")
            canonical_column_present = "canonical_isomeric_smiles" in columns
            self.connection.execute("BEGIN")
            counts = self.connection.execute(
                "SELECT COUNT(*), SUM(CASE WHEN split='train' AND included=1 THEN 1 ELSE 0 END) "
                "FROM molecules"
            ).fetchone()
            if canonical_column_present:
                canonical_counts = self.connection.execute(
                    "SELECT COUNT(*), SUM(CASE WHEN split='train' AND included=1 THEN 1 ELSE 0 END) "
                    "FROM molecules WHERE canonical_isomeric_smiles IS NOT NULL AND canonical_isomeric_smiles<>''"
                ).fetchone()
            else:
                canonical_counts = (0, 0)
            self.canonical_available = bool(canonical_counts[0])
            self.canonical_full_complete = canonical_counts[0] == counts[0]
            self.canonical_train_complete = (canonical_counts[1] or 0) == (counts[1] or 0)
            self.metadata = {
                "database_path": str(database), "database_sha256": actual_hash,
                "manifest_path": str(manifest_path) if manifest_path.is_file() else None,
                "manifest_sha256": _sha256(manifest_path) if manifest_path.is_file() else None,
                "manifest_hash_verified": expected_hash is not None,
                "reference_rows": int(counts[0]), "training_reference_rows": int(counts[1] or 0),
                "training_reference_filter": "split='train' AND included=1",
                "full_reference_filter": "all molecules rows, including excluded/duplicate rows",
                "canonical_column_present": canonical_column_present,
                "canonical_reference_rows": int(canonical_counts[0]),
                "canonical_training_reference_rows": int(canonical_counts[1] or 0),
                "canonical_reference_coverage": _rate(int(canonical_counts[0]), int(counts[0])),
                "canonical_matching_available": self.canonical_available,
                "canonical_matching": "exact canonical_isomeric_smiles column equality" if self.canonical_available else "not available; raw smiles is never substituted",
                "inchikey_matching": "full 27-character key equality; source keys may be source-reported, not recomputed",
                "training_exposure_note": "Prepared included train partition, not proof that every row was consumed by a particular checkpoint",
                "source_manifest_metadata": metadata,
            }
            self.cache: dict[tuple[str | None, str], dict[str, Any]] = {}
        except Exception:
            self.connection.close()
            raise

    def close(self) -> None:
        self.connection.close()

    def _lookup_column(self, column: str, value: str) -> dict[str, bool]:
        # Column names are constants selected by this module; values are bound.
        row = self.connection.execute(
            f'SELECT COUNT(*), MAX(CASE WHEN split=\'train\' AND included=1 THEN 1 ELSE 0 END) '
            f'FROM molecules WHERE "{column}"=?', (value,)
        ).fetchone()
        return {"full": bool(row[0]), "training": bool(row[1])}

    def lookup(self, key: str | None, canonical: str) -> dict[str, Any]:
        identity = key, canonical
        if identity not in self.cache:
            matches = {}
            if key:
                matches["full_inchikey"] = self._lookup_column("full_inchikey", key)
            if self.canonical_available:
                matches["canonical_isomeric_smiles"] = self._lookup_column("canonical_isomeric_smiles", canonical)
            in_training = any(m["training"] for m in matches.values()) if matches else None
            in_full = any(m["full"] for m in matches.values()) if matches else None
            # A failed InChI conversion plus a partially absent canonical index
            # cannot establish absence among the unindexed reference rows.
            if not key and in_training is False and not self.canonical_train_complete:
                in_training = None
            if not key and in_full is False and not self.canonical_full_complete:
                in_full = None
            self.cache[identity] = {
                "in_training_corpus": in_training,
                "in_full_corpus": in_full,
                "reference_matches": matches,
            }
        return self.cache[identity]


def _structure(smiles: Any) -> tuple[dict[str, Any], Any]:
    from rdkit import Chem, rdBase
    from rdkit.Chem import inchi

    result = {
        "valid": False, "invalid_reason": None, "canonical_isomeric_smiles": None,
        "full_inchikey": None, "identity_error": None, "quality_flags": [],
    }
    if not isinstance(smiles, str) or not smiles:
        result["invalid_reason"] = "missing_or_nonstring_smiles"
        return result, None
    if any(char.isspace() for char in smiles):
        result["invalid_reason"] = "whitespace_or_molecule_name_not_allowed"
        return result, None
    params = Chem.SmilesParserParams()
    params.parseName = False
    params.allowCXSMILES = False
    params.sanitize = True
    with rdBase.BlockLogs():
        try:
            molecule = Chem.MolFromSmiles(smiles, params)
        except Exception:
            molecule = None
    if molecule is None or not molecule.GetNumAtoms():
        result["invalid_reason"] = "rdkit_parse_or_sanitization_failed"
        return result, None
    try:
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    except Exception:
        result["invalid_reason"] = "rdkit_canonicalization_failed"
        return result, None
    result.update(valid=True, canonical_isomeric_smiles=canonical)
    if len(Chem.GetMolFrags(molecule)) > 1:
        result["quality_flags"].append("disconnected_fragments")
    if any(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()):
        result["quality_flags"].append("wildcard_atom")
    if any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()):
        result["quality_flags"].append("radical_electrons")
    with rdBase.BlockLogs():
        try:
            text = inchi.MolToInchi(molecule, logLevel=None)
            key = inchi.InchiToInchiKey(text) if text else None
            if isinstance(key, str) and FULL_KEY.fullmatch(key):
                result["full_inchikey"] = key
            else:
                result["identity_error"] = "standard_inchi_key_unavailable"
        except Exception:
            result["identity_error"] = "standard_inchi_key_unavailable"
    return result, molecule


def _computed_properties(molecule: Any) -> tuple[dict[str, Any], dict[str, str]]:
    from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

    calculators = {
        "molecular_weight": Descriptors.MolWt, "crippen_logp": Crippen.MolLogP,
        "tpsa": rdMolDescriptors.CalcTPSA, "hbond_donors": rdMolDescriptors.CalcNumHBD,
        "hbond_acceptors": rdMolDescriptors.CalcNumHBA,
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds,
        "rings": rdMolDescriptors.CalcNumRings, "heavy_atoms": lambda m: m.GetNumHeavyAtoms(),
        "qed": QED.qed,
    }
    values, errors = {}, {}
    for name, calculate in calculators.items():
        try:
            value = calculate(molecule)
            if not math.isfinite(value):
                raise ValueError("nonfinite result")
            values[name] = value
        except Exception as exc:
            values[name] = None
            errors[name] = type(exc).__name__
    return values, errors


def _termination(record: Mapping[str, Any]) -> bool | None:
    primary, alias = record.get("terminated"), record.get("terminated_with_eos")
    for value in (primary, alias):
        if value is not None and not isinstance(value, bool):
            raise ValueError("terminated/terminated_with_eos must be bool or null")
    if primary is not None and alias is not None and primary != alias:
        raise ValueError("Conflicting termination fields")
    return primary if primary is not None else alias


def _novelty_summary(groups: list[list[dict[str, Any]]], field: str) -> dict[str, Any]:
    known, novel, unknown = 0, 0, 0
    for samples in groups:
        values = [sample[field] for sample in samples]
        if any(value is True for value in values):
            known += 1
        elif all(value is False for value in values):
            novel += 1
        else:
            unknown += 1
    total = len(groups)
    return {
        "known_unique_valid": known, "novel_unique_valid": novel,
        "unknown_unique_valid": unknown, "denominator_unique_valid": total,
        "evaluable_unique_valid": known + novel,
        "coverage": _rate(known + novel, total),
        "rate": _rate(novel, total) if unknown == 0 else None,
    }


def evaluate_generated(
    records: Iterable[Mapping[str, Any] | str], corpus_dir: str | Path | None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate every generation attempt without repairing or discarding failures.

    Records normally contain ``smiles`` and Boolean ``terminated``. The alias
    ``terminated_with_eos`` is accepted; missing termination remains unknown.
    Unique structures use full standard InChIKey, falling back to canonical
    isomeric SMILES when InChI is unavailable. Separate uniqueness counts for
    both representations are also returned. An exact match by either available
    reference representation makes a candidate known, a conservative policy.

    The optional corpus directory contains manifest.json + metadata.sqlite.
    No reference means novelty is unknown, never automatically novel. Reference
    training membership means the prepared included train split, not observed
    checkpoint exposure. Properties are RDKit computations, not measured labels.
    """
    from rdkit import rdBase
    from rdkit.Chem import inchi

    reference = _CorpusReference(corpus_dir) if corpus_dir is not None else None
    samples: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    try:
        for index, item in enumerate(records):
            if isinstance(item, str):
                original = {"smiles": item}
            elif isinstance(item, Mapping):
                original = dict(item)
            else:
                raise TypeError("Each generated record must be a mapping or SMILES string")
            terminated = _termination(original)
            chemistry, molecule = _structure(original.get("smiles"))
            sample = {
                "sample_index": index, "smiles": original.get("smiles"), "terminated": terminated,
                "generation_record": original, **chemistry, "unique_valid_structure": False,
                "duplicate_of_sample_index": None, "in_training_corpus": None,
                "in_full_corpus": None, "reference_matches": {},
                "computed_properties": None, "computed_property_errors": {},
                "property_kind": "computed_not_experimental",
            }
            if sample["valid"]:
                canonical, key = sample["canonical_isomeric_smiles"], sample["full_inchikey"]
                identity = ("full_inchikey", key) if key else ("canonical_isomeric_smiles", canonical)
                sample["identity_method"] = identity[0]
                if reference is not None:
                    sample.update(reference.lookup(key, canonical))
                values, errors = _computed_properties(molecule)
                sample.update(computed_properties=values, computed_property_errors=errors)
                if identity not in grouped:
                    grouped[identity] = []
                    sample["unique_valid_structure"] = True
                else:
                    sample["duplicate_of_sample_index"] = grouped[identity][0]["sample_index"]
                grouped[identity].append(sample)
            samples.append(sample)
        reference_metadata = reference.metadata if reference is not None else None
    finally:
        if reference is not None:
            reference.close()
    total = len(samples)
    valid = [sample for sample in samples if sample["valid"]]
    groups = list(grouped.values())
    reported = sum(sample["terminated"] is not None for sample in samples)
    terminated_count = sum(sample["terminated"] is True for sample in samples)
    valid_terminated = sum(sample["terminated"] is True for sample in valid)
    metrics = {
        "attempts": total, "valid_samples": len(valid), "validity_rate": _rate(len(valid), total),
        "termination_reported_samples": reported, "terminated_samples": terminated_count,
        "termination_reporting_coverage": _rate(reported, total),
        "termination_rate": _rate(terminated_count, total) if reported == total else None,
        "termination_rate_among_reported": _rate(terminated_count, reported),
        "valid_and_terminated_samples": valid_terminated,
        "valid_and_terminated_rate": _rate(valid_terminated, total) if reported == total else None,
        "unique_valid_structures": len(groups), "uniqueness_rate": _rate(len(groups), len(valid)),
        "unique_canonical_isomeric_smiles": len({sample["canonical_isomeric_smiles"] for sample in valid}),
        "samples_with_full_inchikey": sum(sample["full_inchikey"] is not None for sample in valid),
        "unique_full_inchikeys": len({sample["full_inchikey"] for sample in valid if sample["full_inchikey"]}),
        "training_novelty": _novelty_summary(groups, "in_training_corpus"),
        "full_corpus_novelty": _novelty_summary(groups, "in_full_corpus"),
    }
    property_summary = {}
    for name in PROPERTY_UNITS:
        values = [group[0]["computed_properties"].get(name) for group in groups]
        values = [value for value in values if value is not None]
        property_summary[name] = {
            "n": len(values), "mean": mean(values) if values else None,
            "median": median(values) if values else None, "min": min(values) if values else None,
            "max": max(values) if values else None, "unit": PROPERTY_UNITS[name],
        }
    report = {
        "schema_version": "1.0", "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "status": "generation_evaluation_completed", "rdkit_version": rdBase.rdkitVersion,
        "inchi_version": inchi.GetInchiVersion() if hasattr(inchi, "GetInchiVersion") else None,
        "metrics": metrics, "samples": samples, "corpus_reference": reference_metadata,
        "computed_property_summary_unique_valid": property_summary,
        "definitions": {
            "validity": "RDKit parsed/sanitized nonempty SMILES; names, whitespace and CXSMILES are not accepted; no repair",
            "termination": "Explicit EOS flag supplied by generator; not inferred from validity",
            "unique_structure": "Full standard InChIKey; canonical isomeric SMILES fallback when unavailable",
            "uniqueness_denominator": "All valid samples, including nonterminated samples",
            "novelty_denominator": "Unique valid structures, including nonterminated samples; no denominator switching",
            "novelty_match": "Known if full InChIKey OR available canonical_isomeric_smiles column matches",
            "unknown_novelty": "Rate is null if any unique valid identity cannot be checked or no reference supplied",
            "identity_processing": "No salt stripping, neutralization or manual tautomer standardization; standard InChI applies its own normalization",
            "computed_properties": "RDKit calculations summarized over one representative per unique valid structure; not experimental measurements",
        },
        "limitations": [
            "Valid or exactly novel SMILES do not establish natural occurrence, synthesis, drug suitability or clinical benefit.",
            "Novelty is limited to this frozen reference and identity policy, not all known molecules or scientific novelty.",
            "Prepared train membership is not a per-checkpoint memorization/exposure audit.",
            "Standard InChI may equate representations that have different canonical SMILES; both unique counts are reported.",
            "No similarity, scaffold novelty, biological activity, disease indication or calibrated clinical probability is evaluated.",
            "Wildcard atoms, radicals and disconnected fragments are flagged but RDKit syntactic validity alone does not exclude them.",
        ],
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(destination)
    return report
