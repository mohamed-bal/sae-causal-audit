"""Regenerate every toy-regime scientific result from scratch.

One command, zero arguments needed, fully deterministic (fixed seeds,
fixed thread count, CPU float32): the executable form of the claim
"every number in the write-up is written by code to results/*.json".

Outputs (all deterministic given the pinned environment):
    results/audit_good_k4.json    - causal audit, TopK k=4 SAE
    results/audit_bad_k13.json    - causal audit, TopK k=13 SAE
    results/audit_good_k4.md      - human-readable render
    results/audit_bad_k13.md
    results/summary.json          - the headline numbers, one small file

Runtime: ~2-4 minutes on a laptop CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Force MKL's Conditional Bitwise Reproducibility mode BEFORE torch (and its
# MKL backend) initializes. Different GitHub-hosted runner instances can sit
# on different underlying CPU microarchitectures (AVX2 vs AVX512, etc.); MKL
# auto-detects this at runtime and picks a different vectorized code path,
# which changes floating-point rounding even with identical seeds and an
# identical pinned torch version. COMPATIBLE mode fixes the code path so the
# result no longer depends on which physical CPU the job happened to land on.
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Defensive: cover every BLAS/threading backend that could be active,
# depending on which library the torch wheel is actually linked against.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Fix Python's hash seed so any dict/set iteration order is deterministic.
os.environ.setdefault("PYTHONHASHSEED", "0")

import torch

# Pin inter-op parallelism IMMEDIATELY after import, before any work is
# scheduled.  torch.set_num_threads (intra-op) is set later in main(), but
# inter-op must be locked here — it cannot be changed once parallel work
# has started.
torch.set_num_interop_threads(1)

from sae_causal_audit import AuditConfig, render_markdown, run_audit, save_json
from sae_causal_audit.toy import (
    ToyConfig,
    ToyProbe,
    train_topk_sae,
    train_toy_model,
    true_directions,
    well_represented_mask,
)


def _print_torch_diagnostics() -> None:
    """Print torch backend info to stderr for CI debugging."""
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

    # Determinism guards: single-threaded reductions avoid nondeterministic
    # float accumulation order across machines with different core counts.
    torch.set_num_threads(1)

    # Print backend diagnostics so CI logs reveal what BLAS is actually active.
    _print_torch_diagnostics()
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError as e:
        # Some torch builds lack a deterministic CPU kernel for a specific op
        # this pipeline happens to use. Seeds still make every result
        # reproducible; only the stricter algorithm-level guarantee is lost.
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
        save_json(report, args.out / f"audit_{name}.json")
        (args.out / f"audit_{name}.md").write_text(
            render_markdown(report, title=f"Toy-regime causal audit: TopK k={k}"),
            encoding="utf-8",
        )

        wr = [r for r in report.results if mask[r.feature_idx]]
        rec = [r for r in wr if r.cosine >= 0.90]
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
