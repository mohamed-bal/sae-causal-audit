# Integration status: `sae-causal-audit` (standalone) × `superposition-to-monosemanticity`

Supersedes the original draft, which recommended a `tool/` subfolder inside
the article-1 monorepo. **That recommendation was overridden**: the tool
ships as its own repository, `mohamed-bal/sae-causal-audit`. This document
tracks what's actually done versus what's still pending.

## 1. Repo layout — DECIDED (standalone, not `tool/`)

```
sae-causal-audit/              # own repo, own root — no tool/ wrapper
├── .github/workflows/ci.yml
├── concepts/gpt2_demo_concepts.json
├── figures/*.png                    # fig0 (pipeline) + fig1..4 (results)
├── results/*.json, *.md
├── scripts/*.py                     # includes generate_figures.py
├── src/sae_causal_audit/*.py
├── tests/*.py
├── CITATION.cff
├── Dockerfile
├── expected_hashes.json
├── expected_results.json            # tolerance-check baseline, added post-launch
├── LICENSE
├── Makefile
├── pyproject.toml
└── README.md
```

`superposition-to-monosemanticity` (article 1's repo) is untouched — no
`tool/` subfolder was ever added there, and none should be.

## 2. Completed so far

**Package**

- [x] Built: `interfaces.py`, `matching.py` (signed cosine, sign bug now
      unrepresentable), `metrics.py` (fired_frac, ablation, steering),
      `stats.py` (bootstrap CIs), `audit.py` (orchestration), `report.py`
      (deterministic JSON + Markdown render), `toy.py` (calibration fixture).
- [x] 37 tests (unit + Hypothesis property tests + toy-regime integration),
      all green.
- [x] Deterministic reproduction pipeline (`make reproduce`) with a
      two-tier scientific-regression gate: - `make verify-hashes` — byte-exact SHA-256 check. **Advisory in CI**
      (`continue-on-error: true`), not the enforced gate — GitHub-hosted
      runners were found to expose different CPU capabilities (AVX2 vs
      AVX512) across runs of the _same_ commit, which changes PyTorch's
      compute path regardless of thread-count pinning. Byte-exactness is
      real and useful locally/in a pinned Docker environment, but cannot
      be guaranteed on shared CI infrastructure. - `make verify` — tolerance check (`rtol=1e-4`, plus an explicit
      `atol=1` band on `bad_k13`'s boundary-sensitive discrete counts).
      **This is the enforced CI gate.** Proven by a deliberately
      corrupted result file failing loudly, and by the gate passing
      consistently despite the hash check drifting across runs. - A fourth reproducibility bug was found and fixed during this
      process: `reproduce_toy_results.py` was computing the bootstrap
      CI over the _full_ recovered set (all 32 features) while the
      census it accompanied was filtered to the 22 well-represented
      features — two different populations reported side by side as if
      one. Fixed so both are computed over the identical filtered
      population; verified by asserting the bootstrap's own `n_samples`
      equals the census denominator.
- [x] Real-model audit harness (`scripts/audit_real_sae.py`, SAELens /
      TransformerLens) + demo concepts file — written, lints clean, **not
      yet executed against a real SAE**.
- [x] Dead `src/sae_causal_audit/adapters/` (empty, unused, unreferenced)
      identified and removed.
- [x] Emoji (the warning-triangle glyph) removed from `report.py`'s
      Markdown renderer, replaced with plain-text `INERT` / `—`; all
      result files regenerated and hashes re-committed under the
      intentional-change process (`verify_result_hashes.py --update`).

**Figures & docs**

- [x] `scripts/generate_figures.py` — a checked-in, deterministic script
      that regenerates every figure from `results/*.json`. Closes the gap
      where figures were previously produced ad hoc with no script of
      record; `make figures` runs it. Legend placement on the specificity
      boxplot (fig1) was fixed to render in a dedicated strip below the
      axes, eliminating collisions with data points seen at earlier
      revisions.
- [x] `fig0_pipeline.png` — new architecture diagram (SAE + FeatureProbe →
      matching/decoder path + causal-battery/encoder path → AuditReport →
      JSON/Markdown). Static, no data dependency, regenerates every run.
- [x] `fig1`–`fig4` (specificity boxplot, cosine-vs-specificity scatter,
      inert census, bootstrap histograms) folded into the same script.
- [x] `README.md` rebuilt: CI/License/Python/Tests badges at the top, the
      pipeline diagram inline, a link to article 2's full write-up, and an
      "Example output" section embedding all four result figures — captions
      now describe the pattern qualitatively (e.g. "several features
      matched at cosine ≥ 0.999") rather than hardcoding specific counts or
      feature indices, since the toy setting's exact census is
      run-sensitive by design (see boundary-sensitivity note below) and a
      caption tied to one run's numbers goes stale on the next.
- [x] `LICENSE` (MIT, actual file) added — previously only referenced in
      text (`pyproject.toml`, `CITATION.cff`, README) with no file for
      GitHub's license badge/detector to find.
- [x] `Makefile` updated with a `figures` target; `all` now runs
      lint → test → reproduce → figures.

**Naming & links**

- [x] Repo name decided: `mohamed-bal/sae-causal-audit`. **Pushed and live**
      — CI has run and gone green (`quality` + `scientific-regression`)
      across multiple commits, including a manually-triggered
      `regenerate-hashes` run.
- [x] `pyproject.toml` `Repository` and `CITATION.cff` `repository-code`
      updated to point at the new repo.
- [x] Article 2's repo references fixed throughout: opening blockquote,
      closing paragraph, and Sources list now correctly distinguish "this
      tool's repo" from "the originating write-up's repo" (previously
      conflated).

**Article 2**

- [x] Written in full — not a small addendum to article 1, a standalone
      sequel piece with its own recap section, the antipodal-pair /
      read-write inertness finding, a fourth reproducibility bug
      (population-mismatch in the bootstrap CI, see above) documented
      alongside the first three, and its own figures.
- [x] Four result figures (fig1–fig4) embedded — final numbers reconciled
      against the actual CI-produced `results/*.json` (downloaded via
      `gh run download`), **not** the numbers from any local/pre-fix run.
      Toy-regime headline: `good_k4` 22/22 recovered, 2 inert (9%);
      `bad_k13` 17/22 recovered, 4 inert (24%).
- [x] `fig0_pipeline.png` integrated into the "The audit, precisely"
      section. Uploaded through the Dev.to editor; URL resolved (no longer
      a placeholder).
- [x] Published: https://dev.to/mohamed_bal/a-cosine-similarity-of-1000-and-the-feature-still-never-fires-turning-a-causal-inertness-finding-5dhp

## 3. Immediately pending

- [ ] **Article 1 (Dev.to) addendum**: a short pointer section, near the
      end, linking to the standalone repo and to article 2. **Not written
      yet** — the only item from the original pending list that is still
      open. Suggested text (update the inert-rate figures to match the
      current toy-regime numbers before posting):

  > **Update (July 2026): the methodology is now a tool, in its own repo.**
  > The causal-audit pipeline from this piece is packaged as
  > [`sae-causal-audit`](https://github.com/mohamed-bal/sae-causal-audit) —
  > structural-typing interface any SAE satisfies (including `sae_lens.SAE`),
  > 37 tests, a two-tier CI gate (byte-exact advisory + tolerance-based
  > enforced) that regenerates every scientific result from scratch, and a
  > harness for auditing published production SAEs. Full write-up:
  > [article 2](https://dev.to/mohamed_bal/a-cosine-similarity-of-1000-and-the-feature-still-never-fires-turning-a-causal-inertness-finding-5dhp).

- [ ] Confirm the README's CI/Tests badges render green on the current
      default branch (expected — CI has passed multiple times — but worth
      a final visual check after the most recent pushes).

## 4. The real-model replication (Tier-1 move) — not started, execution order unchanged

1. **Colab, GPT-2-small first** (free T4): `gpt2-small-res-jb`,
   `blocks.8.hook_resid_pre`, using `concepts/gpt2_demo_concepts.json`
   expanded to 10–15 concepts × 20+ prompts each. Deliverable: one
   `results/real_audit_gpt2.json` + the inert census.
2. **Sensitivity pass**: repeat at 2–3 hook layers and 2 cosine thresholds
   (0.4 / 0.6). The claim to publish is the _shape_ (inert features exist
   in production SAEs and the census quantifies them), not one number.
3. **Gemma Scope** (needs ~16 GB GPU): same pipeline, one layer, as the
   headline.
4. **Write-up**: article 3, standalone — _"How many Gemma Scope features
   are causally inert? A fired-fraction census"_ — linking back to
   articles 1 and 2. Target: Dev.to + LessWrong/Alignment Forum cross-post.

Honesty constraints (unchanged, still the brand):

- The difference-in-means probe direction is a weak ground-truth proxy;
  say so in the methods, and report results at multiple thresholds.
- Real-model inert rates are **not** comparable 1:1 to the toy
  77% / 9% / 24% figures; frame as "the failure mode exists at production
  scale and is cheap to census," never as "N% of Gemma Scope is fake."
- The toy-regime headline numbers are themselves run-sensitive (documented
  extensively in article 2's reproducibility section) — report real-model
  results the same way: with bootstrap CIs, not bare point estimates, and
  with an explicit note on which environment produced them.

## 5. Zenodo DOI — not started, do after a tagged release

1. Log in to zenodo.org with GitHub → enable the repo under
   _GitHub → flip the toggle_.
2. Create a GitHub Release `v0.1.0`.
3. Zenodo mints a DOI automatically → add the badge to `README.md` and the
   `doi:` field to `CITATION.cff`.

## 6. Known follow-ups (unchanged, still deliberately deferred)

- Batched forward passes in `TextConceptProbe` (current loop is fine for
  ≤ ~30 prompts/concept; batch before scaling concepts).
- `mypy --strict` full pass (config is in place; torch stubs make strict a
  half-day task, not blocking).
- Bipartite (Hungarian) matching as an _option_ alongside greedy argmax —
  greedy stays default because atom collisions are a finding to surface,
  not noise to optimize away.
- The `diffuse sharing` formalization (Gram-spectrum vs ETF comparison)
  from the original repo is a separate research thread; it does not
  belong in this tool.
- Consider pinning `ATEN_CPU_CAPABILITY=default` more broadly or
  documenting it as a deliberate escape hatch, rather than the current
  best-effort setting — revisit if hash-check drift on CI becomes a
  recurring annoyance rather than an accepted advisory signal.
