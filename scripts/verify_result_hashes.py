"""Verify regenerated scientific results against committed expected hashes.

CI runs ``make verify``: reproduce all results from scratch, then compare
SHA-256 hashes of the deterministic JSON reports against
``expected_hashes.json``. A code change that silently alters a scientific
number now fails CI exactly like a broken unit test would — regression
detection on the *results*, not just the code.

Bootstrapping / intentional updates:
    python scripts/verify_result_hashes.py results/ expected_hashes.json --update
then commit the updated expected_hashes.json alongside the change that
legitimately moved the numbers, with the diff explaining *why*.

Exit codes: 0 = all match, 1 = mismatch/missing, 2 = usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("expected_file", type=Path)
    ap.add_argument("--update", action="store_true",
                    help="Rewrite expected_file from current results (deliberate use only).")
    args = ap.parse_args(argv)

    if not args.results_dir.is_dir():
        print(f"error: {args.results_dir} is not a directory", file=sys.stderr)
        return 2

    current = {
        p.name: sha256_of(p) for p in sorted(args.results_dir.glob("*.json"))
    }
    if not current:
        print(f"error: no *.json results in {args.results_dir}", file=sys.stderr)
        return 2

    if args.update:
        args.expected_file.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"updated {args.expected_file} with {len(current)} hashes")
        return 0

    if not args.expected_file.exists():
        print(
            f"error: {args.expected_file} missing — bootstrap with --update",
            file=sys.stderr,
        )
        return 1

    try:
        expected = json.loads(args.expected_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(
            f"error: {args.expected_file} is not valid JSON: {exc}",
            file=sys.stderr,
        )
        return 2
    failures: list[str] = []
    for name, want in sorted(expected.items()):
        got = current.get(name)
        if got is None:
            failures.append(f"MISSING  {name}")
        elif got != want:
            failures.append(f"CHANGED  {name}\n  expected {want}\n  got      {got}")
    for name in sorted(set(current) - set(expected)):
        failures.append(f"UNTRACKED {name} (add via --update if intentional)")

    if failures:
        print("scientific-result regression detected:\n" + "\n".join(failures),
              file=sys.stderr)
        return 1
    print(f"all {len(expected)} result hashes match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
