"""Causal validation metrics for matched (feature, atom) pairs.

Three measurements per pair, in increasing order of strength:

1. **fired_frac** — does the encoder ever activate this atom when the
   feature is present? This is the cheap first-pass screen that, in the
   originating experiments, explained 17/22 "recovered" features being
   causally inert: their atoms simply never fired (decoder geometry was
   right; encoder selection never chose them).
2. **Ablation specificity** — zero the atom's code on feature-ON inputs;
   how much does the *targeted* readout drop relative to collateral
   movement everywhere else?
3. **Steering specificity** — force the atom on (with the correct sign)
   on feature-OFF inputs; how much does the targeted readout rise
   relative to collateral movement?

Numerical honesty rules encoded here, matching the originating write-up:

* A pair whose atom never fires gets ``specificity = 0.0`` **and**
  ``fired_frac = 0.0`` — the zero is reported alongside its cause, never
  as an ambiguous "weak effect".
* Division by a zero off-target effect (a perfectly surgical
  intervention) is reported as ``inf`` when the targeted effect is
  nonzero, and ``0.0`` when both are zero — no NaNs escape this module.
* Everything runs under ``torch.no_grad()``; the audit measures, it
  never trains.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .interfaces import DownstreamFn, SparseAutoencoder
from .matching import MatchResult

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class CausalResult:
    """Full causal readout for one matched (feature, atom) pair."""

    feature_idx: int
    atom_idx: int
    cosine: float
    sign: float
    fired_frac: float
    ablation_specificity: float
    ablation_targeted_drop: float
    ablation_off_target: float
    steering_specificity: float
    steering_targeted_rise: float
    steering_off_target: float

    @property
    def causally_inert(self) -> bool:
        """True when the atom never fires for its matched feature.

        This is the single most decision-relevant flag in the audit: a
        pair can pass any cosine threshold and still be inert.
        """
        return self.fired_frac == 0.0


def _safe_ratio(targeted: float, off_target: float) -> float:
    """targeted / off_target with explicit, documented edge semantics."""
    if abs(off_target) < _EPS:
        return 0.0 if abs(targeted) < _EPS else float("inf")
    return targeted / off_target


def fired_fraction(
    sae: SparseAutoencoder, activations_on: torch.Tensor, atom_idx: int
) -> float:
    """Fraction of feature-ON samples on which the atom's code is nonzero."""
    if activations_on.shape[0] == 0:
        raise ValueError("activations_on is empty; cannot estimate fired_frac")
    with torch.no_grad():
        f = sae.encode(activations_on)
        _check_code(f, atom_idx)
        return float((f[:, atom_idx].abs() > _EPS).float().mean().item())


def ablation_effect(
    sae: SparseAutoencoder,
    downstream: DownstreamFn,
    activations_on: torch.Tensor,
    feature_idx: int,
    atom_idx: int,
) -> tuple[float, float, float, float]:
    """Ablate atom ``atom_idx`` on feature-ON inputs; measure specificity.

    Returns:
        ``(specificity, targeted_drop, off_target, fired_frac)``.

    ``feature_idx`` indexes the **downstream readout** dimension being
    targeted (toy regime: the reconstructed feature; real-model regime:
    e.g. the logit/probe dimension for the concept).
    """
    with torch.no_grad():
        f = sae.encode(activations_on)
        _check_code(f, atom_idx)
        fired = float((f[:, atom_idx].abs() > _EPS).float().mean().item())

        y_base = downstream(sae.decode(f))
        _check_readout(y_base, feature_idx)

        f_abl = f.clone()
        f_abl[:, atom_idx] = 0.0
        y_abl = downstream(sae.decode(f_abl))

        targeted = float((y_base[:, feature_idx] - y_abl[:, feature_idx]).mean().item())
        off_mask = torch.ones(y_base.shape[1], dtype=torch.bool, device=y_base.device)
        off_mask[feature_idx] = False
        off = (
            float((y_base - y_abl).abs()[:, off_mask].mean().item())
            if off_mask.any()
            else 0.0
        )
        return _safe_ratio(targeted, off), targeted, off, fired


def steering_effect(
    sae: SparseAutoencoder,
    downstream: DownstreamFn,
    activations_off: torch.Tensor,
    feature_idx: int,
    atom_idx: int,
    sign: float,
    magnitude: float = 1.0,
) -> tuple[float, float, float]:
    """Force atom ``atom_idx`` to ``sign * magnitude`` on feature-OFF inputs.

    ``sign`` is **required** (no default): the originating sign bug —
    steering an anti-aligned atom positive and reading back an exact 0.0
    through the output ReLU — is structurally unrepresentable when the
    signed match must be threaded through explicitly.

    Returns:
        ``(specificity, targeted_rise, off_target)``.
    """
    if sign not in (1.0, -1.0):
        raise ValueError(f"sign must be ±1.0 (from MatchResult), got {sign!r}")
    if magnitude <= 0.0:
        raise ValueError(f"magnitude must be positive, got {magnitude!r}")

    with torch.no_grad():
        f = sae.encode(activations_off)
        _check_code(f, atom_idx)

        y_base = downstream(sae.decode(f))
        _check_readout(y_base, feature_idx)

        f_st = f.clone()
        f_st[:, atom_idx] = sign * magnitude
        y_st = downstream(sae.decode(f_st))

        targeted = float((y_st[:, feature_idx] - y_base[:, feature_idx]).mean().item())
        off_mask = torch.ones(y_base.shape[1], dtype=torch.bool, device=y_base.device)
        off_mask[feature_idx] = False
        off = (
            float((y_st - y_base).abs()[:, off_mask].mean().item())
            if off_mask.any()
            else 0.0
        )
        return _safe_ratio(targeted, off), targeted, off


def causal_result_for_match(
    sae: SparseAutoencoder,
    downstream: DownstreamFn,
    match: MatchResult,
    activations_on: torch.Tensor,
    activations_off: torch.Tensor,
    readout_idx: int | None = None,
    steering_magnitude: float = 1.0,
) -> CausalResult:
    """Run the full causal battery for one matched pair.

    Args:
        readout_idx: Downstream dimension targeted by the interventions.
            Defaults to ``match.feature_idx`` (correct for the toy regime,
            where readout dims coincide with ground-truth features);
            real-model audits pass the probe/logit index explicitly.
    """
    target = match.feature_idx if readout_idx is None else readout_idx
    abl_spec, abl_drop, abl_off, fired = ablation_effect(
        sae, downstream, activations_on, target, match.atom_idx
    )
    st_spec, st_rise, st_off = steering_effect(
        sae,
        downstream,
        activations_off,
        target,
        match.atom_idx,
        sign=match.sign,
        magnitude=steering_magnitude,
    )
    return CausalResult(
        feature_idx=match.feature_idx,
        atom_idx=match.atom_idx,
        cosine=match.cosine,
        sign=match.sign,
        fired_frac=fired,
        ablation_specificity=abl_spec,
        ablation_targeted_drop=abl_drop,
        ablation_off_target=abl_off,
        steering_specificity=st_spec,
        steering_targeted_rise=st_rise,
        steering_off_target=st_off,
    )


def _check_code(f: torch.Tensor, atom_idx: int) -> None:
    if f.ndim != 2:
        raise ValueError(f"encode() must return (batch, d_sae); got shape {tuple(f.shape)}")
    if not (0 <= atom_idx < f.shape[1]):
        raise IndexError(f"atom_idx {atom_idx} out of range for d_sae={f.shape[1]}")


def _check_readout(y: torch.Tensor, feature_idx: int) -> None:
    if y.ndim != 2:
        raise ValueError(
            f"downstream() must return (batch, d_out); got shape {tuple(y.shape)}"
        )
    if not (0 <= feature_idx < y.shape[1]):
        raise IndexError(
            f"readout feature_idx {feature_idx} out of range for d_out={y.shape[1]}"
        )
    if torch.isnan(y).any():
        raise ValueError("downstream readout contains NaNs; refusing to audit")
