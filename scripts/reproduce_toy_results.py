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
import sys
from pathlib import Path

import torch

from sae_causal_audit import AuditConfig, render_markdown, run_audit, save_json
from sae_causal_audit.toy import (
    ToyConfig,
    ToyProbe,
    train_topk_sae,
    train_toy_model,
    true_directions,
    well_represented_mask,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--steps", type=int, default=4000)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    # Determinism guards: single-threaded reductions avoid nondeterministic
    # float accumulation order across machines with different core counts.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

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
