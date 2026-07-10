"""sae-causal-audit — causal validation for sparse-autoencoder features.

A cosine-similarity match between a decoder atom and a concept direction
is a *correlational* claim. This package runs the *causal* battery —
fired-fraction screening, ablation specificity, sign-correct steering —
that separates genuinely load-bearing features from geometrically
plausible, causally inert ones.

Quick start (toy regime, ground truth known)::

    from sae_causal_audit import AuditConfig, run_audit
    from sae_causal_audit.toy import (
        ToyConfig, ToyProbe, train_topk_sae, train_toy_model, true_directions,
    )

    model = train_toy_model(ToyConfig())
    sae = train_topk_sae(model, d_sae=128, k=4)
    report = run_audit(
        sae=sae,
        downstream=model.output,
        probe=ToyProbe(model),
        true_directions=true_directions(model),
        config=AuditConfig(),
    )
    print(report.census)

Real-model regime: implement ``FeatureProbe`` over a labeled probe
dataset at your SAE's hook point, pass logits (or a probe readout) as
``downstream``, and the identical pipeline applies. See
``scripts/audit_real_sae.py`` for a SAELens/TransformerLens harness.
"""

from .audit import SCHEMA_VERSION, AuditConfig, AuditReport, InertCensus, run_audit
from .interfaces import DownstreamFn, FeatureProbe, SparseAutoencoder
from .matching import MatchResult, match_features_to_atoms
from .metrics import (
    CausalResult,
    ablation_effect,
    causal_result_for_match,
    fired_fraction,
    steering_effect,
)
from .report import load_json, render_markdown, save_json
from .stats import BootstrapCI, bootstrap_ci

__version__ = "0.1.0"

__all__ = [
    "SCHEMA_VERSION",
    "AuditConfig",
    "AuditReport",
    "BootstrapCI",
    "CausalResult",
    "DownstreamFn",
    "FeatureProbe",
    "InertCensus",
    "MatchResult",
    "SparseAutoencoder",
    "__version__",
    "ablation_effect",
    "bootstrap_ci",
    "causal_result_for_match",
    "fired_fraction",
    "load_json",
    "match_features_to_atoms",
    "render_markdown",
    "run_audit",
    "save_json",
    "steering_effect",
]
