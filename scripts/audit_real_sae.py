"""Causal audit of a *published, production* SAE — the real-model replication.

Research question (the toy result, promoted to a real model):
    Of the SAE features that pass a standard correlational bar for a
    labeled concept, what fraction are causally inert — their atom never
    fires when the concept is actually present?

This script runs the identical pipeline as the toy regime, with the two
substitutions the real setting forces:

1. **Ground truth → probe datasets.** No accessible ground-truth feature
   directions exist for a real model, so "feature i present" is defined
   by positive/negative text datasets for a labeled concept, and the
   concept's direction in activation space is estimated as the
   difference-in-means of activations (a standard linear-probe direction).
   This is a *weaker* notion of ground truth than the toy setting — that
   is precisely the point of having calibrated the pipeline on the toy
   setting first.
2. **Toy readout → logit readout.** ``downstream`` splices the SAE
   reconstruction back into the residual stream at the hook point and
   returns final-token logits restricted to a token set characteristic
   of the concept.

Hardware: GPT-2-small fits comfortably on a free Colab T4 (or CPU,
slowly). Gemma-2-2b + Gemma Scope wants ~16 GB of GPU RAM.

Usage (Colab or any GPU box):
    pip install -r scripts/requirements-real.txt
    python scripts/audit_real_sae.py --model gpt2 \
        --sae-release gpt2-small-res-jb --sae-id blocks.8.hook_resid_pre \
        --concepts concepts/gpt2_demo_concepts.json \
        --out results/real_audit_gpt2.json

The concepts JSON format (one entry per concept):
    [
      {
        "name": "france",
        "positive": ["The capital of France is", ...],   # concept present
        "negative": ["The capital of Japan is", ...],    # concept absent
        "readout_tokens": [" Paris", " French"]          # concept-diagnostic tokens
      },
      ...
    ]

Design notes:
    * This file has *zero* imports from sae_lens/transformer_lens at
      module import time guarded incorrectly — imports are performed
      inside main() with an actionable error message, so the core
      package never grows heavyweight dependencies.
    * Everything is seeded; the output JSON is the same deterministic,
      hashable format as the toy reports (schema_version shared).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from sae_causal_audit import AuditConfig, render_markdown, run_audit, save_json
from sae_causal_audit.interfaces import FeatureProbe


def _load_real_stack():
    """Import the heavyweight real-model stack with an actionable failure."""
    try:
        from sae_lens import SAE  # type: ignore
        from transformer_lens import HookedTransformer  # type: ignore
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "Real-model audit requires the optional stack:\n"
            "  pip install -r scripts/requirements-real.txt\n"
            f"(import failed: {e})"
        ) from e
    return SAE, HookedTransformer


class TextConceptProbe(FeatureProbe):
    """FeatureProbe over positive/negative text datasets.

    ``feature_idx`` indexes into the concepts list; activations are taken
    at the SAE's hook point on the **final token** of each prompt (the
    position whose next-token logits the readout inspects). Activations
    are computed once per concept and cached — model forward passes are
    the dominant cost at real scale.
    """

    def __init__(self, model, hook_name: str, concepts: list[dict], device: str) -> None:
        self._model = model
        self._hook = hook_name
        self._concepts = concepts
        self._device = device
        self._cache: dict[tuple[int, bool], torch.Tensor] = {}

    def _acts_for(self, prompts: list[str]) -> torch.Tensor:
        outs = []
        with torch.no_grad():
            for p in prompts:  # small n; batching left to a future optimization
                _, cache = self._model.run_with_cache(
                    p, names_filter=self._hook, return_type=None
                )
                outs.append(cache[self._hook][0, -1, :])  # final token position
        return torch.stack(outs).to(self._device)

    def _get(self, feature_idx: int, positive: bool, n: int) -> torch.Tensor:
        key = (feature_idx, positive)
        if key not in self._cache:
            side = "positive" if positive else "negative"
            self._cache[key] = self._acts_for(self._concepts[feature_idx][side])
        acts = self._cache[key]
        if acts.shape[0] < n:
            # Sampling with replacement is statistically fine for the
            # fired_frac / mean-effect estimates; refusing would force
            # users to hand-write hundreds of prompts per concept.
            idx = torch.randint(0, acts.shape[0], (n,))
            return acts[idx]
        return acts[:n]

    def activations_with_feature(self, feature_idx: int, n_samples: int) -> torch.Tensor:
        return self._get(feature_idx, True, n_samples)

    def activations_without_feature(self, feature_idx: int, n_samples: int) -> torch.Tensor:
        return self._get(feature_idx, False, n_samples)


def probe_directions(probe: TextConceptProbe, n_concepts: int, n: int = 64) -> torch.Tensor:
    """Difference-in-means direction per concept — the linear-probe estimate
    standing in for ground truth. Explicitly the weakest link of the real
    regime; reported as such, not hidden."""
    dirs = []
    for i in range(n_concepts):
        pos = probe.activations_with_feature(i, n).mean(dim=0)
        neg = probe.activations_without_feature(i, n).mean(dim=0)
        dirs.append(pos - neg)
    return torch.stack(dirs)


def build_downstream(model, hook_name: str, readout_token_ids: list[list[int]]):
    """Downstream readout: splice h_hat into the residual stream at the hook,
    run the remainder of the model, return per-concept mean logits over each
    concept's diagnostic token set. Output shape (batch, n_concepts)."""

    def downstream(h_hat: torch.Tensor) -> torch.Tensor:
        # Run from the hook point forward. TransformerLens exposes this via
        # start_at_layer using the hook's layer index.
        layer = int(hook_name.split(".")[1])
        with torch.no_grad():
            logits = model(
                h_hat.unsqueeze(1),  # (batch, seq=1, d_model)
                start_at_layer=layer,
            )[:, -1, :]  # (batch, d_vocab)
        cols = [logits[:, ids].mean(dim=1, keepdim=True) for ids in readout_token_ids]
        return torch.cat(cols, dim=1)

    return downstream


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--sae-release", default="gpt2-small-res-jb")
    ap.add_argument("--sae-id", default="blocks.8.hook_resid_pre")
    ap.add_argument("--concepts", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/real_audit.json"))
    ap.add_argument("--n-on", type=int, default=200)
    ap.add_argument("--n-off", type=int, default=200)
    ap.add_argument("--cosine-threshold", type=float, default=0.5,
                    help="Real-model matches are far weaker than toy ones; "
                         "0.5 is a probe-direction bar, not the toy 0.90.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    SAE, HookedTransformer = _load_real_stack()
    torch.manual_seed(args.seed)

    concepts = json.loads(args.concepts.read_text(encoding="utf-8"))
    if not concepts:
        raise SystemExit("concepts file is empty")
    for c in concepts:
        for k in ("name", "positive", "negative", "readout_tokens"):
            if k not in c:
                raise SystemExit(f"concept missing required key {k!r}: {c.get('name')}")

    print(f"loading {args.model} + SAE {args.sae_release}/{args.sae_id} on {args.device}")
    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    sae = SAE.from_pretrained(args.sae_release, args.sae_id, device=args.device)[0]
    sae.eval()

    probe = TextConceptProbe(model, args.sae_id, concepts, args.device)
    dirs = probe_directions(probe, len(concepts))

    readout_ids = [
        [model.to_single_token(t) for t in c["readout_tokens"]] for c in concepts
    ]
    downstream = build_downstream(model, args.sae_id, readout_ids)

    report = run_audit(
        sae=sae,
        downstream=downstream,
        probe=probe,
        true_directions=dirs,
        config=AuditConfig(
            n_samples_on=args.n_on,
            n_samples_off=args.n_off,
            cosine_threshold=args.cosine_threshold,
            seed=args.seed,
        ),
        metadata={
            "regime": "real",
            "model": args.model,
            "sae_release": args.sae_release,
            "sae_id": args.sae_id,
            "concepts": ",".join(c["name"] for c in concepts),
            "direction_estimate": "difference-in-means (linear probe)",
        },
    )
    path = save_json(report, args.out)
    print(render_markdown(report, title=f"Real-model causal audit: {args.sae_release}"))
    print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
