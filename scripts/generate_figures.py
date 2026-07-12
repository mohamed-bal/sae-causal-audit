"""Regenerate every figure in figures/ from the committed results/*.json.

Run after `make reproduce` (or as part of it). Deterministic given the same
results files: no randomness here except the bootstrap resampling, which is
itself seeded. This closes the gap where figures were previously produced
ad hoc — every visual asset in this repo is now regenerable by one script,
the same standard applied to every numeric result.

Usage:
    python scripts/generate_figures.py --results results/ --out figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sae_causal_audit.stats import bootstrap_ci
from sae_causal_audit.toy import ToyConfig, train_toy_model, well_represented_mask

GOOD_C, BAD_C, INERT_C = "#2563eb", "#dc2626", "#f59e0b"
CAP = 3000.0

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.facecolor": "white",
    }
)


def _fx(v):
    """Decode the strict-JSON inf encoding ('inf' / '-inf' strings) to float."""
    if v == "inf":
        return float("inf")
    if v == "-inf":
        return float("-inf")
    return v


def _load_recovered(path: Path, well_represented: set[int] | None, threshold: float = 0.90):
    if not path.exists():
        raise SystemExit(
            f"error: {path} not found — run `make reproduce` first "
            "to generate results/*.json"
        )
    report = json.loads(path.read_text())
    rows = [{k: _fx(v) for k, v in r.items()} for r in report["results"]]
    if well_represented is not None:
        rows = [r for r in rows if int(r["feature_idx"]) in well_represented]
    recovered = [r for r in rows if r["cosine"] >= threshold]
    return rows, recovered


def _cap(vals):
    return [min(v, CAP) if np.isfinite(v) else CAP for v in vals]


def _well_represented_indices() -> set[int]:
    """Recompute which toy features are well-represented, matching toy.py's
    definition (‖W_i‖² >= 0.1), so figures use the same population as the
    reproduction script's printed census."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = train_toy_model(ToyConfig(seed=0), steps=4000)
    mask = well_represented_mask(model)
    return {i for i in range(len(mask)) if bool(mask[i])}


def fig_specificity_boxplot(g_rec, b_rec, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    gd, bd = _cap([r["ablation_specificity"] for r in g_rec]), _cap(
        [r["ablation_specificity"] for r in b_rec]
    )
    bp = ax.boxplot(
        [gd, bd],
        tick_labels=[f"TopK k=4 (good)\nn={len(gd)}", f"TopK k=13 (bad)\nn={len(bd)}"],
        widths=0.45,
        patch_artist=True,
        medianprops=dict(color="black", lw=2),
    )
    for patch, c in zip(bp["boxes"], [GOOD_C, BAD_C], strict=True):
        patch.set_facecolor(c)
        patch.set_alpha(0.25)
        patch.set_edgecolor(c)
    rng = np.random.default_rng(0)
    for i, (vals, res, c) in enumerate([(gd, g_rec, GOOD_C), (bd, b_rec, BAD_C)], start=1):
        x = rng.normal(i, 0.055, len(vals))
        inert = np.array([r["fired_frac"] == 0.0 for r in res])
        ax.scatter(
            np.array(x)[~inert], np.array(vals)[~inert], s=34, color=c, alpha=0.85, zorder=3
        )
        ax.scatter(
            np.array(x)[inert], np.array(vals)[inert], s=70, marker="X",
            color=INERT_C, edgecolor="black", lw=0.6, zorder=4,
        )
    ax.set_yscale("symlog", linthresh=1)
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.text(2.42, 1.15, "specificity = 1\n(no better than collateral)", fontsize=8.5,
             color="gray", va="bottom")
    ax.set_ylabel("Ablation specificity ratio (symlog)")
    ax.set_title(
        "Causal specificity of 'recovered' features (well-represented, cos >= 0.90)"
    )
    ax.scatter([], [], s=70, marker="X", color=INERT_C, edgecolor="black", lw=0.6,
               label="causally inert (atom never fires)")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.13),
        frameon=False, fontsize=10.5, ncol=1,
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_cosine_vs_specificity(g_rec, b_rec, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for res, marker, name in [(g_rec, "o", "TopK k=4"), (b_rec, "s", "TopK k=13")]:
        cosines = [r["cosine"] for r in res]
        specs = _cap([r["ablation_specificity"] for r in res])
        fired = [r["fired_frac"] for r in res]
        sc = ax.scatter(
            cosines, specs, c=fired, cmap="RdYlBu", vmin=0, vmax=1,
            s=64, marker=marker, edgecolor="black", lw=0.5, label=name,
        )
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("Cosine similarity of match (unsigned)")
    ax.set_ylabel("Ablation specificity (symlog)")
    ax.set_title("A near-perfect geometric match does not guarantee a causal one")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("fired_frac (atom fires when feature is ON)")
    candidates = [r for r in g_rec + b_rec if r["cosine"] >= 0.99]
    if candidates:
        worst_f = min(candidates, key=lambda r: r["fired_frac"])
        spec_val = _cap([worst_f["ablation_specificity"]])[0]
        ax.set_xlim(left=0.84)
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5, alpha=0.9)
        if worst_f["fired_frac"] == 0:
            fired_desc = "never fires"
        else:
            fired_desc = f"fired_frac = {worst_f['fired_frac']:.3f}"
        text = (
            f"cos = {worst_f['cosine']:.4f},\n"
            f"{fired_desc},\n"
            f"ablation spec = {worst_f['ablation_specificity']:.1f}"
        )
        ax.annotate(
            text, xy=(worst_f["cosine"], spec_val), xytext=(0.85, max(10, spec_val)),
            bbox=bbox_props,
            arrowprops=dict(arrowstyle="->", color="black", lw=1, shrinkB=4), fontsize=9,
        )
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_inert_census(g_rec, b_rec, n_well_represented: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    firing = [
        sum(1 for r in g_rec if r["fired_frac"] > 0),
        sum(1 for r in b_rec if r["fired_frac"] > 0),
    ]
    inert = [
        sum(1 for r in g_rec if r["fired_frac"] == 0),
        sum(1 for r in b_rec if r["fired_frac"] == 0),
    ]
    x = np.arange(2)
    ax.bar(x, firing, 0.5, color=[GOOD_C, BAD_C], alpha=0.75, label="fires when feature ON")
    ax.bar(x, inert, 0.5, bottom=firing, color=INERT_C, edgecolor="black", lw=0.7,
           label="causally inert (never fires)")
    for i in range(2):
        tot = firing[i] + inert[i]
        ax.text(i, tot + 0.3, f"{inert[i]}/{tot} inert ({inert[i]/tot:.0%})", ha="center",
                 fontsize=10, weight="bold")
    ax.set_xticks(x, ["TopK k=4\n(good)", "TopK k=13\n(bad)"])
    ax.set_ylabel(
        f"Recovered features (cos \u2265 0.90, of {n_well_represented} well-represented)"
    )
    ax.set_ylim(0, n_well_represented + 4)
    ax.set_title("The inert census: recovery by cosine vs. recovery in fact")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_bootstrap(g_rec, b_rec, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for rec, c, name in [(g_rec, GOOD_C, "TopK k=4"), (b_rec, BAD_C, "TopK k=13")]:
        vals = _cap([r["ablation_specificity"] for r in rec])
        t = torch.tensor(vals, dtype=torch.float64)
        gen = torch.Generator().manual_seed(0)
        idx = torch.randint(0, len(t), (10000, len(t)), generator=gen)
        boot = t[idx].median(dim=1).values.numpy()
        ci = bootstrap_ci(vals, "median", seed=0)
        ax.hist(boot, bins=60, alpha=0.55, color=c,
                label=f"{name}: median {ci.point:.1f} [{ci.lo:.1f}, {ci.hi:.1f}]")
    ax.set_xlabel("Bootstrap-resampled median ablation specificity (10,000 resamples, seed 0)")
    ax.set_ylabel("Resample count")
    ax.set_title("Uncertainty of the headline number, made visible")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_pipeline(out: Path) -> None:
    """Architecture overview: how a SparseAutoencoder + FeatureProbe become
    an AuditReport. Static diagram, no data dependency — regenerable any time."""
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.set_xlim(0, 105)
    ax.set_ylim(-3, 33)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fontsize=9.5, weight="bold"):
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.6,
                              joinstyle="round", zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                 fontsize=fontsize, weight=weight, color=ec, zorder=3)

    def arrow(x0, y0, x1, y1, color="#374151"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6), zorder=1)

    box(1, 11, 16, 10, "Your SAE\n(encode / decode\n/ W_dec)", "#eff6ff", GOOD_C)
    box(20, 11, 18, 10, "FeatureProbe\n(ON / OFF\nactivations)", "#eff6ff", GOOD_C)

    box(41, 22, 22, 7, "Matching\n(W_dec, signed cosine)", "#fef3c7", "#b45309", fontsize=9)
    box(41, 1, 22, 7, "Causal battery\nfired_frac . ablation . steering",
        "#fee2e2", BAD_C, fontsize=8.5)

    box(66, 5, 15, 22, "AuditReport\n(census +\nbootstrap CIs)",
        "#f0fdf4", "#15803d", fontsize=9.5)

    box(84, 11, 19, 10, "JSON + Markdown\n(hash-verified)", "#f5f3ff", "#6d28d9", fontsize=9)

    arrow(17, 16, 20, 16)
    arrow(38, 16, 41, 25.5)
    arrow(38, 16, 41, 4.5)
    arrow(63, 25.5, 66, 20)
    arrow(63, 4.5, 66, 12)
    arrow(81, 16, 84, 16)

    ax.text(52, 30.5, "decoder geometry — correlational", ha="center", fontsize=8,
             color="#b45309", style="italic")
    ax.text(52, -1.6, "encoder behavior — causal", ha="center", fontsize=8,
             color=BAD_C, style="italic")

    fig.tight_layout()
    fig.savefig(out, transparent=False)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    wr = _well_represented_indices()
    _, g_rec = _load_recovered(args.results / "audit_good_k4.json", wr)
    _, b_rec = _load_recovered(args.results / "audit_bad_k13.json", wr)

    fig_pipeline(args.out / "fig0_pipeline.png")
    fig_inert_census(g_rec, b_rec, len(wr), args.out / "fig3_inert_census.png")
    fig_specificity_boxplot(g_rec, b_rec, args.out / "fig1_specificity_boxplot.png")
    fig_cosine_vs_specificity(g_rec, b_rec, args.out / "fig2_cosine_vs_specificity.png")
    fig_bootstrap(g_rec, b_rec, args.out / "fig4_bootstrap.png")
    print(f"figures written to {args.out}/")


if __name__ == "__main__":
    main()
