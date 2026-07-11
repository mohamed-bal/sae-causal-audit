"""End-to-end causal audit orchestration.

One call — :func:`run_audit` — takes an SAE, a downstream readout, a
feature probe, and the ground-truth (or probe-estimated) directions, and
returns a :class:`AuditReport`: per-pair causal results, bootstrap CIs
on the headline numbers, and the inert-pair census that is the audit's
main deliverable.

The report is a plain, JSON-serializable dataclass. Serialization and
rendering live in :mod:`sae_causal_audit.report`; this module never does
I/O — it computes.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import torch

from .interfaces import DownstreamFn, FeatureProbe, SparseAutoencoder
from .matching import MatchResult, match_features_to_atoms
from .metrics import CausalResult, causal_result_for_match
from .stats import BootstrapCI, bootstrap_ci

SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class AuditConfig:
    """Everything that parameterizes an audit run — recorded in the report.

    Attributes:
        n_samples_on: Feature-ON samples per pair for fired_frac/ablation.
        n_samples_off: Feature-OFF samples per pair for steering.
        cosine_threshold: Pairs at/above this unsigned cosine count as
            "correlationally recovered" — the population whose causal
            reliability the audit is quantifying. Pairs below it are
            still measured, just excluded from the headline census.
        steering_magnitude: Value the atom is forced to (times its sign).
        bootstrap_resamples / bootstrap_confidence / seed: statistics.
    """

    n_samples_on: int = 500
    n_samples_off: int = 500
    cosine_threshold: float = 0.90
    steering_magnitude: float = 1.0
    bootstrap_resamples: int = 10_000
    bootstrap_confidence: float = 0.95
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_samples_on <= 0 or self.n_samples_off <= 0:
            raise ValueError("sample counts must be positive")
        if not (0.0 <= self.cosine_threshold <= 1.0):
            raise ValueError("cosine_threshold must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class InertCensus:
    """The audit's headline: how many 'recovered' features are causally inert."""

    n_matched: int
    n_recovered: int
    n_recovered_inert: int
    inert_rate_among_recovered: float

    @staticmethod
    def from_results(results: list[CausalResult], threshold: float) -> InertCensus:
        recovered = [r for r in results if r.cosine >= threshold]
        inert = [r for r in recovered if r.causally_inert]
        rate = (len(inert) / len(recovered)) if recovered else 0.0
        return InertCensus(
            n_matched=len(results),
            n_recovered=len(recovered),
            n_recovered_inert=len(inert),
            inert_rate_among_recovered=rate,
        )


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Complete, serializable outcome of one audit run."""

    schema_version: str
    config: AuditConfig
    results: list[CausalResult]
    census: InertCensus
    ablation_specificity_ci: BootstrapCI | None
    steering_specificity_ci: BootstrapCI | None
    runtime_seconds: float
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Plain-dict form for JSON serialization (see report.save_json)."""
        return asdict(self)


def run_audit(
    sae: SparseAutoencoder,
    downstream: DownstreamFn,
    probe: FeatureProbe,
    true_directions: torch.Tensor,
    config: AuditConfig | None = None,
    metadata: dict[str, str] | None = None,
) -> AuditReport:
    """Run the full correlational-then-causal audit.

    Pipeline:
        1. Signed cosine matching of every true direction to its best atom.
        2. For every match: fired_frac, ablation specificity, steering
           specificity (with the match's sign — never assumed positive).
        3. Inert census over the correlationally-recovered subset.
        4. Bootstrap CIs on median specificities (recovered subset).

    The probe is queried per feature, so expensive real-model probes can
    cache internally; this function imposes no caching policy.

    Raises:
        ValueError / IndexError: propagated from matching/metrics on
            malformed shapes — an audit must fail loudly, never report
            numbers computed on misaligned tensors.
    """
    cfg = config or AuditConfig()
    torch.manual_seed(cfg.seed)
    t0 = time.perf_counter()

    matches: list[MatchResult] = match_features_to_atoms(true_directions, sae.W_dec)

    results: list[CausalResult] = []
    for m in matches:
        acts_on = probe.activations_with_feature(m.feature_idx, cfg.n_samples_on)
        acts_off = probe.activations_without_feature(m.feature_idx, cfg.n_samples_off)
        results.append(
            causal_result_for_match(
                sae,
                downstream,
                m,
                acts_on,
                acts_off,
                steering_magnitude=cfg.steering_magnitude,
            )
        )

    census = InertCensus.from_results(results, cfg.cosine_threshold)

    recovered = [r for r in results if r.cosine >= cfg.cosine_threshold]
    abl_ci = st_ci = None
    if recovered:
        abl_ci = bootstrap_ci(
            [r.ablation_specificity for r in recovered],
            statistic="median",
            confidence=cfg.bootstrap_confidence,
            n_resamples=cfg.bootstrap_resamples,
            seed=cfg.seed,
        )
        st_ci = bootstrap_ci(
            [r.steering_specificity for r in recovered],
            statistic="median",
            confidence=cfg.bootstrap_confidence,
            n_resamples=cfg.bootstrap_resamples,
            seed=cfg.seed,
        )

    return AuditReport(
        schema_version=SCHEMA_VERSION,
        config=cfg,
        results=results,
        census=census,
        ablation_specificity_ci=abl_ci,
        steering_specificity_ci=st_ci,
        runtime_seconds=time.perf_counter() - t0,
        metadata=dict(metadata or {}),
    )
