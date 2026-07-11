"""Correlational matching between ground-truth directions and dictionary atoms.

This module intentionally exposes **signed** matching as the primary API.

Rationale (measured, not theoretical): absolute-value cosine matching is
correct for *recovery* metrics — a feature and its exact negation are
equally good correlational matches — but silently wrong for
*interventions*, which are direction-sensitive. In the originating
experiments, an unsigned match caused a steering intervention to push an
anti-aligned atom (dominant weight -0.82) toward +1, driving the
reconstruction into a ReLU-clipped region and yielding an exactly-zero
effect. The signed API makes that entire bug class unrepresentable:
every match carries its sign, and downstream interventions require it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class MatchResult:
    """One ground-truth-direction → dictionary-atom match.

    Attributes:
        feature_idx: Index of the ground-truth / concept direction.
        atom_idx: Index of the best-matching decoder atom (row of W_dec).
        cosine: **Unsigned** cosine similarity in [0, 1] — the recovery
            metric, comparable to thresholds like ≥ 0.90.
        sign: +1.0 if the atom is aligned with the direction, -1.0 if
            anti-aligned. Interventions MUST use this.
    """

    feature_idx: int
    atom_idx: int
    cosine: float
    sign: float

    def __post_init__(self) -> None:
        if self.sign not in (1.0, -1.0):
            raise ValueError(f"sign must be ±1.0, got {self.sign!r}")
        if not (0.0 <= self.cosine <= 1.0 + 1e-6):
            raise ValueError(f"unsigned cosine out of range: {self.cosine!r}")


def _normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-normalize, guarding zero rows (dead atoms) instead of producing NaNs."""
    norms = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    return x / norms


def match_features_to_atoms(
    true_directions: torch.Tensor,
    dictionary: torch.Tensor,
) -> list[MatchResult]:
    """Match each ground-truth direction to its best decoder atom, with sign.

    Args:
        true_directions: ``(n_features, d_in)`` — one row per ground-truth
            (or probe-estimated) feature direction in activation space.
        dictionary: ``(d_sae, d_in)`` — decoder atoms, rows are atoms
            (``W_dec`` convention).

    Returns:
        One :class:`MatchResult` per row of ``true_directions``, in order.

    Raises:
        ValueError: on empty inputs or mismatched trailing dimensions.

    Notes:
        * Matching maximizes ``|cos|`` (recovery semantics), then reports
          the sign separately (intervention semantics) — both facts are
          preserved instead of collapsing them into one lossy number.
        * Dead atoms (zero rows) are handled by norm clamping: their
          cosine against anything is ~0, so they can never win a match
          spuriously via a 0/0.
        * Greedy per-feature argmax (not a bipartite assignment): two
          features may match the same atom. That is a *finding* worth
          surfacing in an audit (atom collisions indicate under-splitting),
          not an error to be optimized away silently. Collisions are
          detectable downstream via duplicate ``atom_idx`` values.
    """
    if true_directions.ndim != 2 or dictionary.ndim != 2:
        raise ValueError(
            "expected 2-D tensors, got shapes "
            f"{tuple(true_directions.shape)} and {tuple(dictionary.shape)}"
        )
    if true_directions.shape[0] == 0 or dictionary.shape[0] == 0:
        raise ValueError("true_directions and dictionary must be non-empty")
    if true_directions.shape[1] != dictionary.shape[1]:
        raise ValueError(
            f"dimension mismatch: directions have d={true_directions.shape[1]}, "
            f"dictionary atoms have d={dictionary.shape[1]}"
        )

    with torch.no_grad():
        t = _normalize_rows(true_directions.detach().to(torch.float32))
        d = _normalize_rows(dictionary.detach().to(torch.float32))
        cos = t @ d.T
        abs_cos = cos.abs()
        best = abs_cos.argmax(dim=1)

        results: list[MatchResult] = []
        for i in range(t.shape[0]):
            j = int(best[i].item())
            signed = float(cos[i, j].item())

            sign = 1.0 if signed >= 0.0 else -1.0
            results.append(
                MatchResult(
                    feature_idx=i,
                    atom_idx=j,
                    cosine=float(abs_cos[i, j].item()),
                    sign=sign,
                )
            )
    return results
