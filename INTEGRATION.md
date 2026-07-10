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
- [x] Deterministic reproduction pipeline (`make reproduce`) + hash-based
      scientific-regression gate (`make verify`) — proven by two clean
      reproductions hashing identically, and by a deliberately corrupted
      result file failing the gate.
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
      record; `make figures` runs it.
- [x] `fig0_pipeline.png` — new architecture diagram (SAE + FeatureProbe →
      matching/decoder path + causal-battery/encoder path → AuditReport →
      JSON/Markdown). Static, no data dependency, regenerates every run.
- [x] `fig1`–`fig4` (specificity boxplot, cosine-vs-specificity scatter,
      inert census, bootstrap histograms) folded into the same script.
- [x] `README.md` rebuilt: CI/License/Python/Tests badges at the top, the
      pipeline diagram inline, a new "Example output" section embedding
      all four result figures with captions matched to the actual numbers
      in `results/*.json`.
- [x] `LICENSE` (MIT, actual file) added — previously only referenced in
      text (`pyproject.toml`, `CITATION.cff`, README) with no file for
      GitHub's license badge/detector to find.
- [x] `Makefile` updated with a `figures` target; `all` now runs
      lint → test → reproduce → figures.

**Naming & links**
- [x] Repo name decided: `mohamed-bal/sae-causal-audit`. GitHub repo created
      (empty, not yet pushed to).
- [x] `pyproject.toml` `Repository` and `CITATION.cff` `repository-code`
      updated to point at the new repo.
- [x] Article 2's repo references fixed throughout: opening blockquote,
      closing paragraph, and Sources list now correctly distinguish "this
      tool's repo" from "the originating write-up's repo" (previously
      conflated).

**Article 2**
- [x] Written in full (`article_v2_full_depth.md`) — not a small addendum
      to article 1, a standalone sequel piece with its own recap section,
      the antipodal-pair / read-write inertness finding, and its own figures.
- [x] Four result figures (fig1–fig4) embedded with live Dev.to CDN URLs —
      already uploaded through the Dev.to editor.
- [x] `fig0_pipeline.png` integrated into the "The audit, precisely"
      section (placed after the intro sentence, before the numbered
      measurement list, with a bridging sentence tying the diagram to the
      decoder/encoder split). **Image URL is a placeholder**
      (`PASTE_FIG0_PIPELINE_URL_HERE`) pending upload — see Pending below.

## 3. Immediately pending

- [ ] **`git push` the actual code** to `github.com/mohamed-bal/sae-causal-audit`.
      Nothing above is live yet — it exists only in this local workspace.
      This blocks every link in the article and README badges from resolving.
- [ ] Upload `figures/fig0_pipeline.png` through the Dev.to editor and
      replace `PASTE_FIG0_PIPELINE_URL_HERE` in `article_v2_full_depth.md`
      with the returned CDN URL — the one remaining manual step before
      article 2 is fully publish-ready.
- [ ] Article 1 (Dev.to) addendum: a short pointer section, near the end,
      linking to the new standalone repo and to article 2. **Not written
      yet** — do not confuse this with article 2 itself, which is already
      done. Suggested text:

  > **Update (July 2026): the methodology is now a tool, in its own repo.**
  > The causal-audit pipeline from this piece is packaged as
  > [`sae-causal-audit`](https://github.com/mohamed-bal/sae-causal-audit) —
  > structural-typing interface any SAE satisfies (including `sae_lens.SAE`),
  > 37 tests, a CI gate that regenerates every scientific result from
  > scratch and fails on any changed hash, and a harness for auditing
  > published production SAEs. Full write-up: [link to article 2].

- [ ] Git commit sequencing for the *first push* (adjust prefixes now that
      there's no `tool/` scope — drop the `(tool)` qualifier, it reads
      oddly at repo root):

  ```
  feat: sae-causal-audit package (matching, metrics, stats, audit, report)
  test: unit + hypothesis property tests + toy integration suite (37 tests)
  feat: deterministic reproduction pipeline + hash-based scientific regression gate
  feat: real-model audit harness (SAELens/TransformerLens) + demo concepts
  feat: figure-generation script + architecture pipeline diagram
  ci: tests, lint, and scientific-regression verification
  docs: README (badges + example output), CITATION.cff, LICENSE
  ```

  Do **not** squash: independently reviewable history is part of the
  professionalism signal for a citable research tool.
- [ ] After the first push, confirm the README's CI/Tests badges actually
      go green — they render "no status" until the first Actions run
      completes, which is expected and not a bug.

## 4. The real-model replication (Tier-1 move) — not started, execution order unchanged

1. **Colab, GPT-2-small first** (free T4): `gpt2-small-res-jb`,
   `blocks.8.hook_resid_pre`, using `concepts/gpt2_demo_concepts.json`
   expanded to 10–15 concepts × 20+ prompts each. Deliverable: one
   `results/real_audit_gpt2.json` + the inert census.
2. **Sensitivity pass**: repeat at 2–3 hook layers and 2 cosine thresholds
   (0.4 / 0.6). The claim to publish is the *shape* (inert features exist
   in production SAEs and the census quantifies them), not one number.
3. **Gemma Scope** (needs ~16 GB GPU): same pipeline, one layer, as the
   headline.
4. **Write-up**: article 3, standalone — *"How many Gemma Scope features
   are causally inert? A fired-fraction census"* — linking back to
   articles 1 and 2. Target: Dev.to + LessWrong/Alignment Forum cross-post.

Honesty constraints (unchanged, still the brand):
- The difference-in-means probe direction is a weak ground-truth proxy;
  say so in the methods, and report results at multiple thresholds.
- Real-model inert rates are **not** comparable 1:1 to the toy 77%/9%/40%
  figures; frame as "the failure mode exists at production scale and is
  cheap to census," never as "N% of Gemma Scope is fake."

## 5. Zenodo DOI — not started, do after the first push

1. Log in to zenodo.org with GitHub → enable the repo under
   *GitHub → flip the toggle*.
2. Create a GitHub Release `v0.1.0`.
3. Zenodo mints a DOI automatically → add the badge to `README.md` and the
   `doi:` field to `CITATION.cff`.

## 6. Known follow-ups (unchanged, still deliberately deferred)

- Batched forward passes in `TextConceptProbe` (current loop is fine for
  ≤ ~30 prompts/concept; batch before scaling concepts).
- `mypy --strict` full pass (config is in place; torch stubs make strict a
  half-day task, not blocking).
- Bipartite (Hungarian) matching as an *option* alongside greedy argmax —
  greedy stays default because atom collisions are a finding to surface,
  not noise to optimize away.
- The `diffuse sharing` formalization (Gram-spectrum vs ETF comparison)
  from the original repo is a separate research thread; it does not
  belong in this tool.
