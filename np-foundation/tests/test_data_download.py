"""Six substantive acquisition/provenance boundary tests; no external network."""
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from npfm.data import export_records, iter_records, merge_normalized_csvs, normalize_record
from npfm.download import build_manifest, download_file, plan_downloads


class Response(io.BytesIO):
    def __init__(self, body, content_type="text/csv"):
        super().__init__(body)
        self.headers = {"Content-Type": content_type, "Content-Length": str(len(body))}

    def geturl(self):
        return "https://example.org/data.csv"


RESOURCE = {"id": "fixture", "name": "fixture", "version": "test", "license": {"spdx": "CC0-1.0"}}
SPEC = {"url": "https://example.org/data.csv", "filename": "data.csv"}


class TestDataDownload(unittest.TestCase):
    def test_html_error_is_not_committed_as_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("npfm.download.urlopen", return_value=Response(b"<!doctype html><html>login</html>", "application/octet-stream")):
                result = download_file(RESOURCE, SPEC, directory, attempts=1)
            self.assertEqual(result["status"], "failed")
            self.assertIn("HTML", result["error"])
            self.assertFalse((Path(directory) / "fixture/data.csv").exists())
            self.assertFalse(list(Path(directory).rglob("*.part")))

    def test_cache_is_hashed_and_tampering_forces_fresh_download(self):
        body = b"compound_id,smiles\n1,CCO\n"
        with tempfile.TemporaryDirectory() as directory:
            with patch("npfm.download.urlopen", return_value=Response(body)):
                first = download_file(RESOURCE, SPEC, directory, attempts=1)
            with patch("npfm.download.urlopen", side_effect=AssertionError("Cache must avoid network")):
                cached = download_file(RESOURCE, SPEC, directory, attempts=1)
            self.assertEqual(cached["status"], "cached")
            self.assertEqual(first["sha256"], hashlib.sha256(body).hexdigest())
            Path(first["path"]).write_bytes(b"tampered")
            with patch("npfm.download.urlopen", return_value=Response(body)) as network:
                repaired = download_file(RESOURCE, SPEC, directory, attempts=1)
            self.assertEqual(repaired["status"], "downloaded")
            network.assert_called_once()
            self.assertEqual(Path(first["path"]).read_bytes(), body)

    def test_full_stereo_keys_survive_and_partial_key_is_not_promoted(self):
        rows = [{"identifier": "a", "canonical_smiles": "C[C@H](O)C(=O)O", "standard_inchi_key": "JVTAAEKCZFNVCJ-REOHCLBHSA-N"},
                {"identifier": "b", "canonical_smiles": "C[C@@H](O)C(=O)O", "standard_inchi_key": "JVTAAEKCZFNVCJ-UWTATZPHSA-N"}]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, one, two, union = [directory / name for name in ["input.csv", "one.csv", "two.csv", "union.csv"]]
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            report = export_records(source, one, source_id="one", deduplicate=True)
            export_records(source, two, source_id="two", deduplicate=True)
            self.assertEqual(report["written"], 2)
            merged = merge_normalized_csvs([one, two], union)
            self.assertEqual(merged["written"], 2)
            with union.open() as handle:
                output = list(csv.DictReader(handle))
            self.assertNotEqual(output[0]["full_inchikey"], output[1]["full_inchikey"])
            self.assertEqual(output[0]["connectivity_id"], output[1]["connectivity_id"])
            self.assertEqual(output[0]["source_ids"], "one|two")
        partial = normalize_record({"smiles": "CCO", "inchikey": "LFQSCWFLJHTTHZ"}, "test")
        self.assertEqual(partial["full_inchikey"], "")
        self.assertIn("LFQSCWFLJHTTHZ", partial["properties_json"])

    def test_missing_activity_is_not_fabricated_as_negative(self):
        raw = {"np_id": "NPC1", "SMILES": "CCO", "InChIKey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N", "IC50": ""}
        normalized = normalize_record(raw, "npass")
        self.assertEqual(normalized["compound_id"], "npass:NPC1")
        self.assertEqual(json.loads(normalized["properties_json"])["IC50"], "")
        self.assertNotIn("active", normalized)
        # Preserve long JSON records across streaming buffer boundaries.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.json"
            long_record = {**raw, "notes": "a" * (1024 * 1024 + 100)}
            path.write_text(json.dumps([long_record, raw]))
            self.assertEqual(list(iter_records(path)), [long_record, raw])

    def test_license_plan_keeps_manual_and_blocked_resources_visible(self):
        def resource(id, gate, files):
            return {"id": id, "name": id, "access": {"license_gate": gate, "files": files}}
        resources = [resource("open", "open", [SPEC]), resource("nc", "noncommercial", [SPEC]),
                     resource("mixed", "mixed", [SPEC]), resource("manual", "open", [])]
        plan = {r["id"]: r["status"] for r in plan_downloads(resources)}
        self.assertEqual(plan, {"open": "ready", "nc": "blocked", "mixed": "blocked", "manual": "manual"})
        approved = {r["id"]: r["status"] for r in plan_downloads(resources, True, True)}
        self.assertEqual(approved["nc"], "ready")
        self.assertEqual(approved["mixed"], "ready")
        self.assertEqual(approved["manual"], "manual")

    def test_manifest_reconciles_later_success_with_historical_failure(self):
        resource = {**RESOURCE, "access": {"files": [SPEC], "license_gate": "open"}}
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "download_log.jsonl"
            log.write_text(json.dumps({"resource_id": "fixture", "status": "failed", "error": "earlier timeout"}) + "\n")
            with patch("npfm.download.urlopen", return_value=Response(b"compound_id,smiles\n1,CCO\n")):
                download_file(resource, SPEC, directory, attempts=1)
            manifest = build_manifest({"resources": [resource]}, directory, verify_hashes=True)
            self.assertEqual(manifest["total_files"], 1)
            self.assertEqual(manifest["resources"][0]["status"], "downloaded")
            self.assertEqual(len(manifest["resources"][0]["historical_failures"]), 1)
            self.assertEqual(manifest["integrity_problems"], [])


if __name__ == "__main__":
    unittest.main()
