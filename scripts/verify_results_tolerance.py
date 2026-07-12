"""Tolerance-based semantic verification of scientific results.

Compares numeric values in results/summary.json against expected_results.json
using ``math.isclose(rel_tol, abs_tol)``.  Unlike the byte-exact hash check
(``verify_result_hashes.py``), this passes across platforms where different
compiled BLAS backends produce slightly different float rounding — provided the
scientific conclusions are unchanged.

Each entry in expected_results.json is either a bare number (uses global
defaults) or an object ``{"value": <num>, "atol": <num>, "rtol": <num>}``
with per-key tolerance overrides.

Usage:
    python scripts/verify_results_tolerance.py results/ expected_results.json
    python scripts/verify_results_tolerance.py results/ expected_results.json --rtol 1e-2

Exit codes: 0 = all values within tolerance, 1 = regression detected, 2 = usage error.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import NamedTuple


class _Expected(NamedTuple):
    value: float
    rtol: float
    atol: float


def _flatten_actual(d: dict, prefix: str = "") -> dict[str, float]:
    """Flatten a nested dict into dot-separated keys, keeping only numeric leaves."""
    out: dict[str, float] = {}
    for k, v in sorted(d.items()):
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_actual(v, key))
        elif isinstance(v, (int, float)):
            out[key] = float(v)
    return out


def _flatten_expected(
    d: dict,
    default_rtol: float,
    default_atol: float,
    prefix: str = "",
) -> dict[str, _Expected]:
    """Flatten expected_results.json, resolving per-key tolerance overrides."""
    out: dict[str, _Expected] = {}
    for k, v in sorted(d.items()):
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, (int, float)):
            out[key] = _Expected(float(v), default_rtol, default_atol)
        elif isinstance(v, dict):
            if "value" in v:

                out[key] = _Expected(
                    float(v["value"]),
                    float(v.get("rtol", default_rtol)),
                    float(v.get("atol", default_atol)),
                )
            else:

                out.update(_flatten_expected(v, default_rtol, default_atol, key))
    return out


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"error: {path} is not valid JSON: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("expected_file", type=Path)
    ap.add_argument("--rtol", type=float, default=1e-4,
                    help="Relative tolerance (default: 1e-4).")
    ap.add_argument("--atol", type=float, default=1e-6,
                    help="Absolute tolerance (default: 1e-6).")
    args = ap.parse_args(argv)

    summary_path = args.results_dir / "summary.json"
    if not summary_path.exists():
        print(f"error: {summary_path} not found", file=sys.stderr)
        return 2

    if not args.expected_file.exists():
        print(f"error: {args.expected_file} not found", file=sys.stderr)
        return 2

    actual = _load_json(summary_path)
    expected = _load_json(args.expected_file)

    actual_flat = _flatten_actual(actual)
    expected_flat = _flatten_expected(expected, args.rtol, args.atol)

    failures: list[str] = []

    for key in sorted(expected_flat):
        if key not in actual_flat:
            exp = expected_flat[key]
            failures.append(f"MISSING  {key}  (expected {exp.value})")
            continue
        exp = expected_flat[key]
        act_val = actual_flat[key]
        if not math.isclose(act_val, exp.value, rel_tol=exp.rtol, abs_tol=exp.atol):
            failures.append(
                f"DRIFT    {key}  expected={exp.value}  got={act_val}  "
                f"(rtol={exp.rtol}, atol={exp.atol})"
            )

    expected_groups = set(expected.keys())
    for key in sorted(set(actual_flat) - set(expected_flat)):
        group = key.split(".")[0]
        if group not in expected_groups:
            failures.append(
                f"UNTRACKED  {key}={actual_flat[key]}  (not in expected)"
            )

    if failures:
        print(
            "semantic-result regression detected:\n" + "\n".join(failures),
            file=sys.stderr,
        )
        return 1

    print(
        f"all {len(expected_flat)} numeric values within tolerance "
        f"(global rtol={args.rtol}, atol={args.atol})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
