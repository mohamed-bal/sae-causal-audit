"""Regenerate every toy-regime scientific result from scratch.

No arguments needed, fully deterministic (fixed seeds, single-threaded,
CPU float32).

Outputs:
    results/audit_good_k4.json    - causal audit, TopK k=4 SAE
    results/audit_bad_k13.json    - causal audit, TopK k=13 SAE
    results/audit_good_k4.md      - human-readable render
    results/audit_bad_k13.md
    results/summary.json          - headline numbers

Runtime: ~2-4 minutes on a laptop CPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path



os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import torch

from sae_causal_audit import AuditConfig, render_markdown, run_audit, save_json
from sae_causal_audit.stats import bootstrap_ci
from sae_causal_audit.toy import (
    ToyConfig,
    ToyProbe,
    train_topk_sae,
    train_toy_model,
    true_directions,
    well_represented_mask,
)


def _print_torch_diagnostics() -> None:
    """Dump torch backend info to stderr, for CI debugging."""
    import sys as _sys

    lines = [
        "=== torch reproducibility diagnostics ===",
        f"torch version: {torch.__version__}",
        f"MKL available: {torch.backends.mkl.is_available()}",
        f"MKL-DNN/oneDNN available: {torch.backends.mkldnn.is_available()}",
        f"num_threads (intra-op): {torch.get_num_threads()}",
        f"num_interop_threads (inter-op): {torch.get_num_interop_threads()}",
        "--- torch.__config__.show() ---",
        torch.__config__.show(),
        "--- torch.__config__.parallel_info() ---",
        torch.__config__.parallel_info(),
        "=== end diagnostics ===",
    ]
    print("\n".join(lines), file=_sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--steps", type=int, default=4000)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    _print_torch_diagnostics()
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError as e:
       
        print(
            f"warning: deterministic algorithms unavailable in this torch "
            f"build ({e}); continuing with seeded-but-not-algorithm-strict "
            f"reproducibility",
            file=sys.stderr,
        )

    print("training toy model (32 features -> 8 dims, sparsity 0.95)...")
    model = train_toy_model(ToyConfig(seed=0), steps=args.steps)
    mask = well_represented_mask(model)
    n_wr = int(mask.sum())
    print(f"  well-represented: {n_wr}/{model.cfg.n_features}")

    summary: dict[str, dict] = {
        "toy_model": {
            "n_features": model.cfg.n_features,
            "n_hidden": model.cfg.n_hidden,
            "sparsity": model.cfg.sparsity,
            "well_represented": n_wr,
        }
    }

    for name, k in [("good_k4", 4), ("bad_k13", 13)]:
        print(f"training + auditing TopK SAE k={k} ({name})...")
        sae = train_topk_sae(model, d_sae=128, k=k, steps=args.steps, seed=0)
        report = run_audit(
            sae=sae,
            downstream=model.output,
            probe=ToyProbe(model, seed=0),
            true_directions=true_directions(model),
            config=AuditConfig(n_samples_on=500, n_samples_off=500, seed=0),
            metadata={"sae": name, "k": str(k), "regime": "toy"},
        )

        wr = [r for r in report.results if mask[r.feature_idx]]
        rec = [r for r in wr if r.cosine >= 0.90]

        if rec:
            abl_ci = bootstrap_ci(
                [r.ablation_specificity for r in rec],
                statistic="median",
                confidence=report.config.bootstrap_confidence,
                n_resamples=report.config.bootstrap_resamples,
                seed=report.config.seed,
            )
            st_ci = bootstrap_ci(
                [r.steering_specificity for r in rec],
                statistic="median",
                confidence=report.config.bootstrap_confidence,
                n_resamples=report.config.bootstrap_resamples,
                seed=report.config.seed,
            )
        else:
            abl_ci = None
            st_ci = None

        report = dataclasses.replace(
            report, ablation_specificity_ci=abl_ci, steering_specificity_ci=st_ci
        )

        save_json(report, args.out / f"audit_{name}.json")
        (args.out / f"audit_{name}.md").write_text(
            render_markdown(report, title=f"Toy-regime causal audit: TopK k={k}"),
            encoding="utf-8",
        )

        inert = [r for r in rec if r.causally_inert]
        summary[name] = {
            "recovered": len(rec),
            "of_well_represented": len(wr),
            "recovered_inert": len(inert),
            "inert_rate": round(len(inert) / max(len(rec), 1), 4),
            "ablation_specificity_ci": str(report.ablation_specificity_ci),
        }
        print(
            f"  recovered {len(rec)}/{len(wr)}, inert {len(inert)}/{len(rec)} "
            f"({len(inert) / max(len(rec), 1):.0%})"
        )

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"done -> {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
