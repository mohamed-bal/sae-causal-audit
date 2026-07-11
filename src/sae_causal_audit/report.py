"""Report persistence and rendering.

Two output formats, two audiences:

* ``save_json`` — machine-readable, **deterministically serialized**
  (sorted keys, fixed float formatting via ``repr``), so CI can hash the
  file and detect scientific-result regressions, not just code changes.
* ``render_markdown`` — a human-readable audit summary suitable for a
  PR comment, a report artifact, or pasting into a write-up.

Non-finite floats (``inf`` specificities are legitimate) are encoded as
strings ``"inf"``/``"-inf"`` in JSON — standard-compliant JSON has no
Infinity literal, and silently emitting the Python extension breaks
downstream parsers. ``load_json`` reverses the encoding.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .audit import AuditReport


def _encode(obj: Any) -> Any:
    """Recursively make a report dict strict-JSON-safe and deterministic."""
    if isinstance(obj, float):
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        if math.isnan(obj):
            raise ValueError("NaN reached serialization; upstream invariant broken")
        return obj
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    return obj


def _decode(obj: Any) -> Any:
    if obj == "inf":
        return float("inf")
    if obj == "-inf":
        return float("-inf")
    if isinstance(obj, dict):
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


VOLATILE_FIELDS = frozenset({"runtime_seconds"})


def save_json(report: AuditReport, path: str | Path) -> Path:
    """Write the report as deterministic, strict JSON. Returns the path.

    Volatile fields (see ``VOLATILE_FIELDS``) are excluded: two runs that
    produce identical science must produce byte-identical files.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    d = {k: v for k, v in report.to_dict().items() if k not in VOLATILE_FIELDS}
    payload = _encode(d)
    p.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return p


def load_json(path: str | Path) -> dict:
    """Load a saved report back into a plain dict (inf strings restored)."""
    return _decode(json.loads(Path(path).read_text(encoding="utf-8")))


def _fmt(x: float) -> str:
    if math.isinf(x):
        return "∞" if x > 0 else "-inf"
    return f"{x:.3g}"


def render_markdown(report: AuditReport, title: str = "SAE Causal Audit") -> str:
    """Render a human-readable Markdown summary of an audit report."""
    c = report.census
    lines: list[str] = [
        f"# {title}",
        "",
        f"Schema `{report.schema_version}` · seed {report.config.seed} · "
        f"{report.runtime_seconds:.2f}s",
        "",
        "## Headline: inert census",
        "",
        f"- Matched pairs: **{c.n_matched}**",
        f"- Correlationally recovered (cos ≥ {report.config.cosine_threshold}): "
        f"**{c.n_recovered}**",
        f"- Recovered but **causally inert** (atom never fires): "
        f"**{c.n_recovered_inert}** "
        f"(**{c.inert_rate_among_recovered:.0%}** of recovered)",
        "",
    ]
    if report.ablation_specificity_ci and report.steering_specificity_ci:
        lines += [
            "## Specificity (recovered subset, median + bootstrap CI)",
            "",
            f"- Ablation: {report.ablation_specificity_ci}",
            f"- Steering: {report.steering_specificity_ci}",
            "",
        ]
    lines += [
        "## Per-pair results",
        "",
        "| feat | atom | cos | sign | fired | abl. spec | steer spec | inert |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in sorted(report.results, key=lambda r: r.feature_idx):
        lines.append(
            f"| {r.feature_idx} | {r.atom_idx} | {r.cosine:.3f} | "
            f"{'+' if r.sign > 0 else '-'} | {r.fired_frac:.2f} | "
            f"{_fmt(r.ablation_specificity)} | {_fmt(r.steering_specificity)} | "
            f"{'INERT' if r.causally_inert else '—'} |"
        )
    if report.metadata:
        lines += ["", "## Metadata", ""]
        lines += [f"- **{k}**: {v}" for k, v in sorted(report.metadata.items())]
    return "\n".join(lines) + "\n"
