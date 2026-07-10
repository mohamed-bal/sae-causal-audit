# sae-causal-audit

[![CI](https://github.com/mohamed-bal/sae-causal-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/mohamed-bal/sae-causal-audit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Tests: 37 passing](https://img.shields.io/badge/tests-37%20passing-brightgreen.svg)](tests/)

**Causal validation for sparse-autoencoder features: separate genuinely load-bearing features from geometrically plausible, causally inert ones.**

A cosine-similarity match between an SAE decoder atom and a concept direction is a _correlational_ claim. In controlled experiments with full ground truth ([write-up](https://github.com/mohamed-bal/superposition-to-monosemanticity)), a meaningful share of features that passed a standard cosine recovery bar (≥ 0.90) turned out to be **causally inert** — the matched atom never fired once when its feature was actually present, including matches at cosine > 0.999. Decoder geometry and encoder behavior are different empirical claims, and only an intervention can tell them apart.

This package is that intervention, packaged: a model-agnostic causal audit you can run on any SAE — toy or production.

📄 **Full write-up:** [A Cosine Similarity of 1.000 and the Feature Still Never Fires](https://dev.to/mohamed_bal/a-cosine-similarity-of-1000-and-the-feature-still-never-fires-turning-a-causal-inertness-finding-5dhp) — the reproducibility debugging story, the antipodal-pair mechanism, and the full results walkthrough behind this package.

![Audit pipeline diagram: a SparseAutoencoder and a FeatureProbe feed two parallel paths — matching against W_dec (decoder geometry, correlational) and a causal battery of fired_frac, ablation, and steering (encoder behavior, causal) — which combine into an AuditReport serialized as hash-verified JSON and Markdown](https://dev-to-uploads.s3.us-east-2.amazonaws.com/uploads/articles/h9flfp6a17cwz6plaizr.png)

Three measurements per matched feature, in increasing order of causal strength:

1. **`fired_frac`** — does the encoder _ever_ activate this atom when the feature is present? (The cheap screen that already explains most inert cases.)
2. **Ablation specificity** — zero the atom on feature-ON inputs: how hard does the _targeted_ readout drop relative to collateral movement everywhere else?
3. **Sign-correct steering** — force the atom on (with the _signed_ match, never an assumed `+`) on feature-OFF inputs: does the targeted readout rise?

Every summary number ships with a seeded percentile-bootstrap confidence interval, and every report serializes to deterministic JSON — **two runs producing identical science produce byte-identical files** within a pinned environment, so CI can hash results and fail on scientific regressions, not just code regressions.

## Install

```bash
pip install -e .            # core: torch only, zero other runtime deps
pip install -e ".[dev]"     # + pytest, hypothesis, ruff, mypy
```

## Quick start (toy regime, ground truth known)

```python
from sae_causal_audit import AuditConfig, run_audit, render_markdown
from sae_causal_audit.toy import (
    ToyConfig, ToyProbe, train_topk_sae, train_toy_model, true_directions,
)

model = train_toy_model(ToyConfig())            # Elhage et al. (2022) toy model
sae = train_topk_sae(model, d_sae=128, k=4)     # TopK SAE on its activations

report = run_audit(
    sae=sae,
    downstream=model.output,                     # any (batch, d_in) -> (batch, d_out)
    probe=ToyProbe(model),                       # supplies feature-ON / feature-OFF inputs
    true_directions=true_directions(model),
    config=AuditConfig(cosine_threshold=0.90),
)

print(report.census)         # InertCensus(n_matched=32, n_recovered=…, n_recovered_inert=…)
print(render_markdown(report))
```

## Example output

Running the quick start above on two TopK SAEs from opposite ends of a Pareto front (`k=4`, well-fit; `k=13`, deliberately under-sparse) produces this census over the 22 well-represented toy features, cos ≥ 0.90:

![Stacked bar chart: recovered features split into firing vs. causally inert, for the good (2/22 inert, 9%) and bad (4/17 inert, 24%) SAEs](https://dev-to-uploads.s3.us-east-2.amazonaws.com/uploads/articles/cw5r8xbss8uc7crjqaqp.png)

A cosine match can be maximally confident and still causally wrong — several of the inert features across both SAEs are matched at cosine ≥ 0.999:

![Boxplot with individual points on a symlog scale: ablation specificity for both SAEs, inert pairs marked as amber X's sitting at exactly zero, with the specificity=1 collateral-damage line marked](https://dev-to-uploads.s3.us-east-2.amazonaws.com/uploads/articles/afhbe1x03xfuxabmdrjf.png)

Every summary statistic ships as a bootstrap distribution, not a point estimate — note the bad SAE's median sitting lower with a much wider interval:

![Scatter of cosine similarity vs. ablation specificity on a symlog axis, points colored by fired_frac: high-cosine points span the full range from specificity in the hundreds down to exactly zero, with the antipodal inert matches at cosine ≈ 1.000 annotated](https://dev-to-uploads.s3.us-east-2.amazonaws.com/uploads/articles/gz8bcni9vawdvz30sxh2.png)

![Overlaid histograms of 10,000 bootstrap-resampled medians for each SAE: the good SAE's distribution sits tightly around 130, the bad SAE's is wide with mass spread from 46 up past 110](https://dev-to-uploads.s3.us-east-2.amazonaws.com/uploads/articles/1dq56oeglxg1yqzzf8fb.png)

All four figures — and every number behind them — regenerate from scratch with `make reproduce && python scripts/generate_figures.py` (see [Reproduce every number](#reproduce-every-number)).

## Auditing a real, published SAE (GPT-2 / Gemma Scope)

No ground truth exists for a real model, so the audit substitutes:
**probe datasets** (positive/negative prompts per labeled concept) define "feature present", a difference-in-means direction stands in for the true direction, and the downstream readout is concept-diagnostic logits with the SAE reconstruction spliced into the residual stream.

```bash
pip install -r scripts/requirements-real.txt     # sae-lens + transformer-lens
python scripts/audit_real_sae.py \
    --model gpt2 --sae-release gpt2-small-res-jb --sae-id blocks.8.hook_resid_pre \
    --concepts concepts/gpt2_demo_concepts.json \
    --out results/real_audit_gpt2.json
```

GPT-2-small runs on a free Colab T4 (or CPU, slowly). The probe-direction estimate is explicitly the weakest link of the real regime — which is exactly why the pipeline is calibrated first on the toy regime, where the right answers are known and the test suite asserts them.

## Bring your own SAE

Anything satisfying a three-member structural protocol can be audited — no inheritance, no registration:

```python
class SparseAutoencoder(Protocol):
    W_dec: torch.Tensor                                  # (d_sae, d_in), rows = atoms
    def encode(self, h: torch.Tensor) -> torch.Tensor: ...
    def decode(self, f: torch.Tensor) -> torch.Tensor: ...
```

`sae_lens.SAE` satisfies it out of the box. Matching uses `W_dec` (correlational); **all causal metrics go through `encode`/`decode`**, so encoder-side behavior — the thing decoder geometry cannot certify — is what is actually measured.

## Reproduce every number

```bash
make reproduce                          # regenerates results/*.json from scratch, deterministically (~3 min, CPU)
python scripts/generate_figures.py      # regenerates every figure in this README from results/*.json
make verify                             # reproduce + tolerance check against expected values (any OS)
make verify-hashes                      # reproduce + byte-exact hash check (CI reference environment only)
make test lint                          # 37 tests incl. hypothesis property tests; ruff clean
docker build -t sae-audit . && docker run sae-audit   # fully pinned environment
```

### Reproducibility guarantees

| Guarantee                                        | Scope                                                 | Verified by                   |
| ------------------------------------------------ | ----------------------------------------------------- | ----------------------------- |
| **Byte-exact** (SHA-256 hash match)              | best-effort / advisory in CI (runner hardware varies) | `make verify-hashes` in CI    |
| **Semantic** (numeric values within `rtol=1e-4`) | Any platform (enforced gate)                          | `make verify` in CI / locally |

Cross-platform byte-exactness is impossible: the `torch` CPU wheel for each OS is compiled with a different compiler (GCC on Linux, MSVC on Windows, Clang on macOS) and linked against a different MKL build.
These different binaries produce different float rounding at the bit level, even with identical seeds and single-threaded execution. Even _within_ GitHub-hosted Linux runners, exposed CPU capability (AVX2 vs AVX512) can vary run to run, changing PyTorch's compute path regardless of thread-count pinning.
The semantic check confirms that the scientific conclusions (recovery counts, inert rates, specificity CIs) are unchanged despite this irreducible platform and runner variance — which is why it, not the hash check, is the enforced CI gate.

The deliberately-degraded `bad_k13` SAE sits at a TopK selection boundary: several of its cosine similarities cluster near the 0.90 recovery threshold, so different BLAS builds or runner hardware can flip individual features across that line, shifting both the exact recovered/inert counts and which specific features they refer to run to run.
This is expected behavior of a pathological SAE designed to probe the audit's sensitivity — it is consistent with the boundary-sensitivity mechanism identified in the root-cause analysis, not a reproducibility failure. `expected_results.json` tolerances are set accordingly.

## Design decisions worth knowing

- **Signed matching is mandatory for interventions.** Absolute-cosine matching is correct for recovery metrics but silently wrong for steering (a `-0.82`-aligned atom steered positive reads back exactly `0.0` through an output ReLU). The API threads the match sign through explicitly, making that bug class unrepresentable — and the original bug is encoded as a permanent regression test.
- **Zeros carry their cause.** A specificity of `0.0` is always reported alongside `fired_frac`, so "the atom never fires" is never confusable with "the effect is weak". `inf` (perfectly surgical intervention) is legal, serialized safely, and rejected where it would be meaningless (means).
- **The audit computes; it never trains, never does I/O in core modules, and fails loudly on shape mismatches** — numbers computed on misaligned tensors are worse than no numbers.

## Citation

See [`CITATION.cff`](CITATION.cff). If you use the inert-census methodology, please also cite the originating write-up and the upstream work it builds on (Elhage et al. 2022; Bricken et al. 2023; Gao et al. 2024).

## License

MIT — see [`LICENSE`](LICENSE).
