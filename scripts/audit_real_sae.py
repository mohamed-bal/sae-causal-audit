"""Causal audit of a *published, production* SAE — the real-model replication.

Research question (the toy result, promoted to a real model):
    Of the SAE features that pass a standard correlational bar for a
    labeled concept, what fraction are causally inert — their atom never
    fires when the concept is actually present?

This script runs the identical pipeline as the toy regime, with the two
substitutions the real setting forces:

1. **Ground truth -> probe datasets.** No accessible ground-truth feature
   directions exist for a real model, so "feature i present" is defined
   by positive/negative text datasets for a labeled concept, and the
   concept's direction in activation space is estimated as the
   difference-in-means of activations (a standard linear-probe direction).
   This is a *weaker* notion of ground truth than the toy setting -- that
   is precisely the point of having calibrated the pipeline on the toy
   setting first.
2. **Toy readout -> logit readout.** ``downstream`` splices the SAE
   reconstruction back into the residual stream at the hook point and
   returns final-token logits restricted to a token set characteristic
   of the concept.

Hardware: GPT-2-small fits comfortably on a free Colab T4 (or CPU,
slowly). Gemma-2-2b + Gemma Scope wants ~16 GB of GPU RAM.

Quick usage (validate a concepts file in seconds, no model download):
    python scripts/audit_real_sae.py --concepts concepts/gpt2_demo_concepts.json --dry-run

Full run:
    pip install -r scripts/requirements-real.txt
    python scripts/audit_real_sae.py --model gpt2 \
        --sae-release gpt2-small-res-jb --sae-id blocks.8.hook_resid_pre \
        --concepts concepts/gpt2_demo_concepts.json \
        --cache-dir .cache/gpt2_demo \
        --out results/real_audit_gpt2.json

The concepts JSON format (one entry per concept):
    [
      {
        "name": "france",
        "positive": ["The capital of France is", ...],   # concept present
        "negative": ["The capital of Japan is", ...],    # concept absent
        "readout_tokens": [" Paris", " French"]           # concept-diagnostic tokens
      },
      ...
    ]

Engineering notes: --dry-run validates the concepts file without loading
any model. Activations are batched and, with --cache-dir, cached to disk
per concept so an interrupted run can resume. Model/SAE loading and token
lookups raise with the concept name attached instead of a bare traceback.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

from sae_causal_audit import AuditConfig, render_markdown, run_audit, save_json
from sae_causal_audit.interfaces import FeatureProbe

logger = logging.getLogger("audit_real_sae")


def _capture_environment() -> dict:
    """Library versions, GPU, and git commit, for reproducibility metadata."""
    env: dict = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
    }
    try:
        import sae_lens
        env["sae_lens_version"] = getattr(sae_lens, "__version__", "unknown")
    except ImportError:
        pass
    try:
        import transformer_lens
        version = getattr(transformer_lens, "__version__", None)
        if version is None:
            from importlib.metadata import version as _pkg_version
            version = _pkg_version("transformer_lens")
        env["transformer_lens_version"] = version
    except ImportError:
        pass
    except Exception:
        env["transformer_lens_version"] = "unknown"
    if torch.cuda.is_available():
        try:
            env["gpu"] = torch.cuda.get_device_name(0)
            env["cuda_version"] = torch.version.cuda
        except Exception:  
            pass
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2,
            cwd=Path(__file__).resolve().parent,
        )
        if commit.returncode == 0:
            env["git_commit"] = commit.stdout.strip()
    except Exception:  
        pass
    return env



@dataclasses.dataclass(frozen=True, slots=True)
class Concept:
    """A validated concept. Constructing one is the only way a concept
    enters the pipeline, so every downstream consumer can assume
    well-formed data -- no scattered defensive checks later on."""

    name: str
    positive: list[str]
    negative: list[str]
    readout_tokens: list[str]

    @staticmethod
    def from_dict(d: dict, index: int) -> Concept:
        if not isinstance(d, dict):
            raise ValueError(f"concept #{index} is not a JSON object")
        missing = [k for k in ("name", "positive", "negative", "readout_tokens") if k not in d]
        if missing:
            raise ValueError(f"concept #{index} missing required key(s): {missing}")

        name = d["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"concept #{index}: 'name' must be a non-empty string")

        for key in ("positive", "negative", "readout_tokens"):
            val = d[key]
            if not isinstance(val, list) or not val:
                raise ValueError(f"concept {name!r}: {key!r} must be a non-empty list")
            if not all(isinstance(x, str) and x.strip() for x in val):
                raise ValueError(
                    f"concept {name!r}: every entry in {key!r} must be a non-empty string"
                )

        return Concept(
            name=name,
            positive=list(d["positive"]),
            negative=list(d["negative"]),
            readout_tokens=list(d["readout_tokens"]),
        )


def load_concepts(path: Path) -> list[Concept]:
    """Parse and fully validate the concepts file. Raises SystemExit with a
    specific, actionable message on any problem -- this is deliberately the
    only validation gate; nothing downstream re-checks concept shape."""
    if not path.exists():
        raise SystemExit(f"concepts file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"concepts file is not valid JSON ({path}): {e}") from e
    if not isinstance(raw, list) or not raw:
        raise SystemExit(f"concepts file must contain a non-empty JSON array: {path}")

    try:
        concepts = [Concept.from_dict(d, i) for i, d in enumerate(raw)]
    except ValueError as e:
        raise SystemExit(str(e)) from e

    names = [c.name for c in concepts]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise SystemExit(f"duplicate concept names in {path}: {dupes}")

    return concepts



def _load_real_stack():
    """Import the heavyweight real-model stack with an actionable failure."""
    try:
        from sae_lens import SAE  
        from transformer_lens import HookedTransformer  
    except ImportError as e: 
        raise SystemExit(
            "Real-model audit requires the optional stack:\n"
            "  pip install -r scripts/requirements-real.txt\n"
            f"(import failed: {e})"
        ) from e
    return SAE, HookedTransformer


def _validate_readout_tokens(model, concepts: list[Concept]) -> list[list[int]]:
    """Resolve readout token strings to ids, failing with a message that
    names the offending concept and token rather than a bare tokenizer
    exception surfacing mid-audit."""
    resolved: list[list[int]] = []
    for c in concepts:
        ids = []
        for t in c.readout_tokens:
            try:
                ids.append(model.to_single_token(t))
            except Exception as e:
                raise SystemExit(
                    f"concept {c.name!r}: readout token {t!r} does not map to a "
                    f"single model token ({e}). Try a leading space (' word') or "
                    "a different surface form; check with model.to_str_tokens(t)."
                ) from e
        resolved.append(ids)
    return resolved



class TextConceptProbe(FeatureProbe):
    """FeatureProbe over positive/negative text datasets. Batches prompts
    and, with cache_dir, persists activations to disk for resumption."""

    def __init__(
        self,
        model,
        hook_name: str,
        concepts: list[Concept],
        device: str,
        batch_size: int = 16,
        cache_dir: Path | None = None,
    ) -> None:
        self._model = model
        self._hook = hook_name
        self._concepts = concepts
        self._device = device
        self._batch_size = max(1, batch_size)
        self._cache_dir = cache_dir
        self._mem_cache: dict[tuple[int, bool], torch.Tensor] = {}
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, feature_idx: int, positive: bool) -> Path | None:
        if self._cache_dir is None:
            return None
        side = "pos" if positive else "neg"
        name = self._concepts[feature_idx].name
        return self._cache_dir / f"{name}.{side}.pt"

    def _acts_for(self, prompts: list[str], desc: str) -> torch.Tensor:
        try:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(prompts), self._batch_size), desc=desc, leave=False)
        except ImportError:  
            iterator = range(0, len(prompts), self._batch_size)

        outs = []
        with torch.no_grad():
            for start in iterator:
                batch = prompts[start : start + self._batch_size]
                tokens = self._model.to_tokens(batch, padding_side="left")
                _, cache = self._model.run_with_cache(
                    tokens, names_filter=self._hook, return_type=None
                )
                outs.append(cache[self._hook][:, -1, :])  
        return torch.cat(outs, dim=0).to(self._device)

    def _get(self, feature_idx: int, positive: bool, n: int) -> torch.Tensor:
        key = (feature_idx, positive)
        if key in self._mem_cache:
            acts = self._mem_cache[key]
        else:
            cache_path = self._cache_path(feature_idx, positive)
            if cache_path is not None and cache_path.exists():
                logger.info("loading cached activations: %s", cache_path)
                acts = torch.load(cache_path, map_location=self._device)
            else:
                concept = self._concepts[feature_idx]
                side_prompts = concept.positive if positive else concept.negative
                side_label = "on" if positive else "off"
                logger.info(
                    "computing activations: concept=%r side=%s n_prompts=%d",
                    concept.name, side_label, len(side_prompts),
                )
                acts = self._acts_for(side_prompts, desc=f"{concept.name} ({side_label})")
                if cache_path is not None:
                    torch.save(acts.cpu(), cache_path)
                    logger.debug("cached activations: %s", cache_path)
            self._mem_cache[key] = acts

        if acts.shape[0] < n:
            logger.warning(
                "concept %r: only %d prompts on the %s side but %d requested; "
                "sampling with replacement (fine for fired_frac / mean-effect "
                "estimates, not for exact counts)",
                self._concepts[feature_idx].name, acts.shape[0],
                "positive" if positive else "negative", n,
            )
            idx = torch.randint(0, acts.shape[0], (n,))
            return acts[idx]
        return acts[:n]

    def activations_with_feature(self, feature_idx: int, n_samples: int) -> torch.Tensor:
        return self._get(feature_idx, True, n_samples)

    def activations_without_feature(self, feature_idx: int, n_samples: int) -> torch.Tensor:
        return self._get(feature_idx, False, n_samples)


def probe_directions(probe: TextConceptProbe, n_concepts: int, n: int = 64) -> torch.Tensor:
    """Difference-in-means direction per concept -- the linear-probe estimate
    standing in for ground truth. Explicitly the weakest link of the real
    regime; reported as such in report metadata, never hidden."""
    dirs = []
    for i in range(n_concepts):
        pos = probe.activations_with_feature(i, n).mean(dim=0)
        neg = probe.activations_without_feature(i, n).mean(dim=0)
        dirs.append(pos - neg)
    return torch.stack(dirs)



def _parse_hook_layer(hook_name: str) -> int:
    """Parse the layer index out of a TransformerLens hook name, failing
    with a clear message instead of a bare IndexError/ValueError on an
    unexpected format (e.g. a non-TransformerLens hook naming scheme)."""
    parts = hook_name.split(".")
    if len(parts) < 2 or parts[0] != "blocks" or not parts[1].isdigit():
        raise SystemExit(
            f"cannot parse a layer index from hook name {hook_name!r}; "
            "expected TransformerLens format 'blocks.<N>.hook_...'"
        )
    return int(parts[1])


def build_downstream(model, hook_name: str, readout_token_ids: list[list[int]]):
    """Downstream readout: splice h_hat into the residual stream at the
    hook, run the remainder of the model, return per-concept mean logits
    over each concept's diagnostic token set. Output shape (batch, n_concepts)."""
    layer = _parse_hook_layer(hook_name)

    def downstream(h_hat: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = model(
                h_hat.unsqueeze(1),  
                start_at_layer=layer,
            )[:, -1, :] 
        cols = [logits[:, ids].mean(dim=1, keepdim=True) for ids in readout_token_ids]
        return torch.cat(cols, dim=1)

    return downstream


# CLI
def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--sae-release", default="gpt2-small-res-jb")
    ap.add_argument("--sae-id", default="blocks.8.hook_resid_pre")
    ap.add_argument("--concepts", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/real_audit.json"))
    ap.add_argument("--n-on", type=int, default=200)
    ap.add_argument("--n-off", type=int, default=200)
    ap.add_argument(
        "--cosine-threshold", type=float, default=0.5,
        help="Real-model matches are far weaker than toy ones; 0.5 is a "
             "probe-direction bar, not the toy regime's 0.90.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32",
        help="float32 is safest for reproducibility; float16/bfloat16 trade "
             "precision for GPU memory headroom on larger models.",
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Persist per-concept activations here for crash-safe resumption "
             "across interrupted runs.",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Validate the concepts file and exit -- no model, no GPU, no "
             "network access required. Use this to iterate on a concepts "
             "file in seconds instead of minutes.",
    )
    ap.add_argument("--verbose", action="store_true", help="Debug-level logging.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _setup_logging(args.verbose)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no CUDA device is visible")

    logger.info("validating concepts file: %s", args.concepts)
    concepts = load_concepts(args.concepts)
    logger.info("%d concept(s) loaded: %s", len(concepts), ", ".join(c.name for c in concepts))

    if args.dry_run:
        logger.info("--dry-run: concepts file is valid; exiting without loading a model")
        return 0

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    SAE, HookedTransformer = _load_real_stack()
    dtype = getattr(torch, args.dtype)

    env = _capture_environment()
    logger.info("environment: %s", json.dumps(env, indent=2))

    logger.info(
        "loading %s + SAE %s/%s on %s (%s)",
        args.model, args.sae_release, args.sae_id, args.device, args.dtype,
    )
    t0 = time.time()
    try:
        model = HookedTransformer.from_pretrained(args.model, device=args.device, dtype=dtype)
    except Exception as e:
        raise SystemExit(f"failed to load model {args.model!r}: {e}") from e
    try:
        sae = SAE.from_pretrained(args.sae_release, args.sae_id, device=args.device)
    except Exception as e:
        raise SystemExit(
            f"failed to load SAE {args.sae_release!r}/{args.sae_id!r}: {e}\n"
            "Check the release/id against SAELens's pretrained_saes.yaml: "
            "https://github.com/decoderesearch/SAELens"
        ) from e
    sae.eval()
    logger.info("models loaded in %.1fs", time.time() - t0)

    readout_ids = _validate_readout_tokens(model, concepts)

    probe = TextConceptProbe(
        model, args.sae_id, concepts, args.device,
        batch_size=args.batch_size, cache_dir=args.cache_dir,
    )

    logger.info("computing probe directions (difference-in-means)")
    t0 = time.time()
    try:
        dirs = probe_directions(probe, len(concepts))
    except KeyboardInterrupt:
        logger.warning(
            "interrupted while computing probe directions; activations "
            "already written to %s were saved and will be reused on the "
            "next run with the same --cache-dir",
            args.cache_dir,
        )
        return 130
    logger.info("probe directions computed in %.1fs", time.time() - t0)

    downstream = build_downstream(model, args.sae_id, readout_ids)

    logger.info("running causal audit battery (fired_frac, ablation, steering)")
    t0 = time.time()
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
            "concepts": ",".join(c.name for c in concepts),
            "direction_estimate": "difference-in-means (linear probe)",
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            **env,
        },
    )
    logger.info("audit finished in %.1fs", time.time() - t0)

    path = save_json(report, args.out)
    print(render_markdown(report, title=f"Real-model causal audit: {args.sae_release}"))
    logger.info("saved: %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
