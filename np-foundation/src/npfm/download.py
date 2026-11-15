"""Auditable acquisition of explicitly registered public research datasets.

Downloads are not permission grants. Mixed, restricted and unverified resources
remain outside public code repositories. This module uses the standard library.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def default_registry():
    if os.environ.get("NPFM_REGISTRY"):
        return Path(os.environ["NPFM_REGISTRY"])
    for path in [Path(__file__).resolve().parents[2] / "registry/databases.json",
                 Path.cwd() / "registry/databases.json"]:
        if path.is_file():
            return path
    raise FileNotFoundError("Pass --registry or set NPFM_REGISTRY")


def load_registry(path=None):
    with Path(path or default_registry()).open(encoding="utf-8") as handle:
        data = json.load(handle)
    ids = [row["id"] for row in data.get("resources", [])]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate registry IDs")
    return data


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _check_url(url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Only explicit public HTTPS URLs are supported")
    if parsed.username or parsed.password:
        raise ValueError("Do not embed credentials in URLs")


def _safe_component(value):
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"Unsafe path component: {value!r}")
    return value


def plan_downloads(resources, allow_noncommercial=False, allow_unknown_license=False):
    plan = []
    for resource in resources:
        access = resource.get("access", {})
        gate = access.get("license_gate", "unknown")
        status, reason = "ready", "Explicit public bulk endpoint registered"
        if gate == "paid":
            status, reason = "blocked", "Subscription or separate permission required"
        elif access.get("requires_manual") or not access.get("files"):
            status, reason = "manual", access.get("notes", "No verified direct bulk file")
        elif gate == "noncommercial" and not allow_noncommercial:
            status, reason = "blocked", "Pass --allow-noncommercial only for an appropriate permitted use"
        elif gate in {"unknown", "mixed", "restricted"} and not allow_unknown_license:
            status, reason = "blocked", "Review terms before --allow-unknown-license; not cleared for public redistribution"
        plan.append({"id": resource["id"], "name": resource["name"], "status": status,
                     "reason": reason, "license": resource.get("license"),
                     "access": access.get("method"), "files": access.get("files", [])})
    return plan


def download_file(resource, spec, output_dir, timeout=60, attempts=3, max_bytes=8 * 1024**3):
    """Download atomically, verify cache and write a full provenance sidecar.

    A locally computed SHA256 is an integrity fingerprint, not publisher attestation.
    Optional registry SHA256/MD5 and byte counts are checked against actual bytes.
    """
    url = spec["url"]
    _check_url(url)
    filename = _safe_component(spec.get("filename") or Path(urlparse(url).path).name)
    directory = Path(output_dir).resolve() / _safe_component(resource["id"])
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    sidecar = target.with_name(filename + ".provenance.json")
    if target.exists() and sidecar.exists():
        try:
            previous = json.loads(sidecar.read_text(encoding="utf-8"))
            actual = sha256_file(target)
            valid = (previous.get("url") == url and previous.get("sha256") == actual
                     and previous.get("bytes") == target.stat().st_size
                     and (not spec.get("sha256") or spec["sha256"] == actual)
                     and (not spec.get("bytes") or spec["bytes"] == target.stat().st_size))
            if valid:
                if spec.get("md5"):
                    digest = hashlib.md5()
                    with target.open("rb") as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(block)
                    if digest.hexdigest() != spec["md5"]:
                        raise ValueError("Cached publisher MD5 mismatch")
                previous.update({"cache_verified_at": utc_now(),
                                 "publisher_checksum_verified": bool(spec.get("sha256") or spec.get("md5")),
                                 "publisher_checksum_type": "sha256" if spec.get("sha256") else "md5" if spec.get("md5") else None})
                _atomic_json(sidecar, previous)
                return {**previous, "status": "cached"}
        except (OSError, ValueError):
            pass
    temporary = target.with_name(filename + ".part")
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            started = utc_now()
            request = Request(url, headers={"User-Agent": "NPFM/0.1 (public research data; provenance retained)",
                                           "Accept": "*/*", "Accept-Encoding": "identity"})
            digest, source_md5, count, first = hashlib.sha256(), hashlib.md5(), 0, True
            with urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                expected = response.headers.get("Content-Length")
                if "text/html" in content_type.lower():
                    raise ValueError("Server returned HTML instead of a data file")
                if expected and int(expected) > max_bytes:
                    raise ValueError("File exceeds maximum allowed bytes")
                resolved = response.geturl()
                _check_url(resolved)
                with temporary.open("wb") as handle:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        if first:
                            prefix = block[:512].lstrip().lower()
                            if prefix.startswith((b"<!doctype html", b"<html")):
                                raise ValueError("HTML error page masquerading as data")
                            first = False
                        count += len(block)
                        if count > max_bytes:
                            raise ValueError("Download exceeded maximum allowed bytes")
                        digest.update(block)
                        source_md5.update(block)
                        handle.write(block)
                    handle.flush()
                    os.fsync(handle.fileno())
                if not count:
                    raise ValueError("Empty download")
                if expected and count != int(expected):
                    raise ValueError(f"Content-Length mismatch: expected {expected}, got {count}")
                actual = digest.hexdigest()
                if spec.get("sha256") and actual != spec["sha256"]:
                    raise ValueError("Publisher/registry SHA256 mismatch")
                if spec.get("md5") and source_md5.hexdigest() != spec["md5"]:
                    raise ValueError("Publisher MD5 mismatch")
                if spec.get("bytes") and count != spec["bytes"]:
                    raise ValueError("Pinned byte count mismatch")
                with temporary.open("rb") as handle:
                    signature = handle.read(4)
                if spec.get("content_format") == "tar":
                    if not spec.get("md5") and not spec.get("sha256"):
                        raise ValueError("TAR suffix exception requires publisher checksum")
                    with tarfile.open(temporary, "r:") as archive:
                        if archive.next() is None:
                            raise ValueError("Empty TAR")
                elif filename.endswith(".gz") and not signature.startswith(b"\x1f\x8b"):
                    raise ValueError("Invalid gzip signature")
                if filename.endswith(".zip") and not signature.startswith(b"PK"):
                    raise ValueError("Invalid ZIP signature")
                metadata = {"resource_id": resource["id"], "resource_version": resource.get("version"),
                            "filename": filename, "path": str(target), "url": url,
                            "resolved_url": resolved, "bytes": count, "sha256": actual,
                            "publisher_checksum_verified": bool(spec.get("sha256") or spec.get("md5")),
                            "publisher_checksum_type": "sha256" if spec.get("sha256") else "md5" if spec.get("md5") else None,
                            "content_format": spec.get("content_format"),
                            "started_at": started, "completed_at": utc_now(),
                            "content_type": content_type, "etag": response.headers.get("ETag"),
                            "last_modified": response.headers.get("Last-Modified"),
                            "license": resource.get("license"),
                            "file_license": spec.get("license", resource.get("license", {}).get("spdx")),
                            "record_count": None, "record_count_note": "Not yet counted; not a unique compound count",
                            "status": "downloaded"}
            os.replace(temporary, target)
            _atomic_json(sidecar, metadata)
            return metadata
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            temporary.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(min(attempt * 2, 6))
    return {"resource_id": resource["id"], "filename": filename, "url": url,
            "status": "failed", "error": last_error, "completed_at": utc_now()}


def fetch_resources(resources, output_dir, *, allow_noncommercial=False, allow_unknown_license=False,
                    timeout=60, attempts=3, max_bytes=8 * 1024**3, progress=True):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    by_id = {r["id"]: r for r in resources}
    results = []
    for item in plan_downloads(resources, allow_noncommercial, allow_unknown_license):
        if item["status"] != "ready":
            batch = [{"resource_id": item["id"], "status": item["status"],
                      "reason": item["reason"], "completed_at": utc_now()}]
        else:
            batch = []
            for spec in item["files"]:
                if progress:
                    print(f"Downloading {item['id']}: {spec.get('filename')}", file=sys.stderr, flush=True)
                result = download_file(by_id[item["id"]], spec, output_dir, timeout, attempts, max_bytes)
                results.append(result)
                with (output_dir / "download_log.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(result, ensure_ascii=False) + "\n")
                if progress:
                    print(json.dumps({k: result.get(k) for k in ["resource_id", "filename", "status", "bytes", "sha256", "error"]}), flush=True)
        for result in batch:
            results.append(result)
            with (output_dir / "download_log.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(result, ensure_ascii=False) + "\n")
            if progress:
                print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = {"generated_at": utc_now(), "results": results,
               "counts": {s: sum(r["status"] == s for r in results)
                          for s in ["downloaded", "cached", "failed", "manual", "blocked"]}}
    # Concurrent fetches may each write this report: the cumulative manifest is authoritative.
    _atomic_json(output_dir / ("download_report_" + str(os.getpid()) + ".json"), summary)
    return summary


def build_manifest(registry, output_dir, *, verify_hashes=False):
    output_dir = Path(output_dir).resolve()
    files, problems, history = [], [], []
    log_path = output_dir / "download_log.jsonl"
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                history.append(json.loads(line))
            except ValueError:
                problems.append({"log": str(log_path), "error": "Malformed log row"})
    for sidecar in sorted(output_dir.rglob("*.provenance.json")):
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            path = sidecar.with_name(sidecar.name.removesuffix(".provenance.json"))
            if not path.is_file():
                problems.append({"sidecar": str(sidecar), "error": "Data file missing"})
                continue
            if path.stat().st_size != data.get("bytes"):
                problems.append({"path": str(path), "error": "File-size mismatch"})
                continue
            if verify_hashes and sha256_file(path) != data.get("sha256"):
                problems.append({"path": str(path), "error": "SHA256 mismatch"})
                continue
            data.update({"relative_path": str(path.relative_to(output_dir)),
                         "integrity_checked_at": utc_now() if verify_hashes else None,
                         "current_status": "verified" if verify_hashes else "present_size_checked"})
            files.append(data)
        except (OSError, ValueError, KeyError) as exc:
            problems.append({"sidecar": str(sidecar), "error": str(exc)})
    resources = []
    for resource in registry["resources"]:
        matched = [f for f in files if f.get("resource_id") == resource["id"]]
        expected = resource.get("access", {}).get("files", [])
        expected_names = {s.get("filename") or Path(urlparse(s["url"]).path).name for s in expected}
        present = {f["filename"] for f in matched}
        attempted = [h for h in history if h.get("resource_id") == resource["id"]]
        if expected_names and expected_names <= present:
            status = "downloaded"
        elif matched:
            status = "partial"
        elif not expected_names:
            status = "manual"
        elif any(h.get("status") == "failed" for h in attempted):
            status = "failed"
        elif any(h.get("status") == "blocked" for h in attempted):
            status = "blocked"
        else:
            status = "not_attempted"
        resources.append({"id": resource["id"], "name": resource["name"], "status": status,
                          "expected_files": len(expected_names), "successful_files": len(matched),
                          "missing_filenames": sorted(expected_names - present),
                          "bytes": sum(f["bytes"] for f in matched), "license": resource.get("license"),
                          "access_notes": resource.get("access", {}).get("notes"),
                          "last_access_check": resource.get("last_access_check"),
                          "historical_failures": [h for h in attempted if h.get("status") == "failed"]})
    result = {"schema_version": "1.1", "generated_at": utc_now(), "root": str(output_dir),
              "hashes_verified": verify_hashes, "files": files, "resources": resources,
              "total_files": len(files), "total_bytes": sum(f["bytes"] for f in files),
              "resource_status_counts": {s: sum(r["status"] == s for r in resources)
                                         for s in ["downloaded", "partial", "failed", "manual", "blocked", "not_attempted"]},
              "integrity_problems": problems,
              "incomplete_files": [str(p.relative_to(output_dir)) for p in output_dir.rglob("*.part")],
              "status_scope": "Selected explicitly registered export files, not all database tables or upstream collections",
              "warning": "Heterogeneous overlapping record counts cannot be summed as unique natural compounds"}
    _atomic_json(output_dir / "cumulative_manifest.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "plan", "fetch", "manifest"])
    parser.add_argument("--registry")
    parser.add_argument("--ids")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--allow-noncommercial", action="store_true")
    parser.add_argument("--allow-unknown-license", action="store_true")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-gb", type=float, default=8)
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args(argv)
    registry = load_registry(args.registry)
    resources = registry["resources"]
    if args.ids:
        ids = {x.strip() for x in args.ids.split(",") if x.strip()}
        missing = ids - {r["id"] for r in resources}
        if missing:
            parser.error("Unknown IDs: " + ", ".join(sorted(missing)))
        resources = [r for r in resources if r["id"] in ids]
    elif args.command == "fetch" and not args.all:
        parser.error("fetch requires --ids or --all")
    if args.timeout <= 0 or args.attempts < 1 or args.max_gb <= 0:
        parser.error("timeout, attempts and max-gb must be positive")
    if args.command == "list":
        for r in resources:
            print(json.dumps({"id": r["id"], "name": r["name"], "license": r["license"]["spdx"],
                              "bulk_files": len(r["access"].get("files", []))}, ensure_ascii=False))
    elif args.command == "plan":
        print(json.dumps(plan_downloads(resources, args.allow_noncommercial, args.allow_unknown_license), ensure_ascii=False, indent=2))
    elif args.command == "manifest":
        result = build_manifest({**registry, "resources": resources}, args.output, verify_hashes=args.verify_hashes)
        print(json.dumps({k: result[k] for k in ["total_files", "total_bytes", "resource_status_counts", "integrity_problems", "incomplete_files"]}, indent=2))
        return int(bool(result["integrity_problems"] or result["incomplete_files"]))
    else:
        result = fetch_resources(resources, args.output, allow_noncommercial=args.allow_noncommercial,
                                 allow_unknown_license=args.allow_unknown_license, timeout=args.timeout,
                                 attempts=args.attempts, max_bytes=int(args.max_gb * 1024**3))
        return int(any(result["counts"][s] for s in ["failed", "blocked", "manual"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
