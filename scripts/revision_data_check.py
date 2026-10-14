"""Replay the frozen revision audit's dataset identity checks (CPU and read-only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, content: bool) -> dict:
    stat = path.stat()
    row = {"path": str(path), "resolved_path": str(path.resolve()),
           "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if content:
        row["sha256"] = sha256(path)
    return row


def inventory(root: Path, pattern: str, content: bool) -> dict:
    rows = [file_identity(path, content) for path in sorted(set(root.glob(pattern)))]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {"n_files": len(rows), "total_bytes": sum(row["size"] for row in rows),
            "manifest_sha256": hashlib.sha256(payload).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    checks = []
    start = time.monotonic()
    try:
        for recipe in audit["data_manifest_recipes"]:
            dataset = audit["data_identity"][recipe["dataset"]]
            expected = dataset[recipe["field"]]
            actual = inventory(Path(dataset["root"]), recipe["glob"], recipe["content"])
            checks.append({"name": recipe["dataset"] + "." + recipe["field"],
                           "passed": all(actual[key] == expected[key] for key in actual),
                           "expected": {key: expected[key] for key in actual},
                           "actual": actual, "recipe": recipe})

        identities = []
        identities.extend(audit["data_identity"]["S11"]["split_files"])
        identities.append(audit["data_identity"]["STURM"]["metadata"])
        gdacs = audit["data_identity"]["GDACS"]
        identities.extend(gdacs[key] for key in
                          ("shipped_index", "shipped_manifest", "regenerated_manifest",
                           "existing_mapped_index"))
        for expected in identities:
            actual = file_identity(Path(expected["path"]), "sha256" in expected)
            checks.append({"name": expected["path"], "passed": actual == expected,
                           "expected": expected, "actual": actual})
    except (OSError, ValueError, KeyError) as error:
        checks.append({"name": "audit_replay", "passed": False,
                       "error": type(error).__name__ + ": " + str(error)})

    result = {"schema_version": 1, "audit": str(args.audit.resolve()),
              "audit_sha256": sha256(args.audit), "passed": all(c["passed"] for c in checks),
              "checks": checks, "elapsed_seconds": time.monotonic() - start,
              "limits": "Optical image inventories check path/size/mtime, not content; "
                        "label inventories hash content. Exact original GDACS pool membership "
                        "remains blocked even when these existing dataset identities match."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + "." + str(os.getpid()) + ".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"passed": result["passed"], "checks": len(checks),
                      "elapsed_seconds": result["elapsed_seconds"], "out": str(args.out)}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
