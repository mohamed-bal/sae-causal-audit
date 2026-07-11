"""Bootstrap confidence intervals for audit summary statistics.

The originating experiments reported point estimates (e.g. "median
ablation specificity 168.8") over n = 22 matched pairs and were explicit
that such n is small. This module upgrades every summary number to a
percentile-bootstrap interval so that the *shape* of the uncertainty is
part of the reported result, not a caveat in prose.

Choices, stated:

* **Percentile bootstrap** (not BCa): transparent, assumption-light,
  and adequate for the medians/means reported here. BCa's acceleration
  estimate is itself unstable at n ≈ 20.
* **Seeded**: every interval is exactly reproducible.
* **inf-aware**: specificity ratios can legitimately be ``inf`` (a
  perfectly surgical intervention). Medians handle that fine;
  means do not — so ``bootstrap_ci`` refuses non-finite values for the
  mean and accepts them for the median, loudly, instead of silently
  propagating ``inf``/``nan`` into a report.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Literal

import torch

Statistic = Literal["median", "mean"]


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    """A point estimate with its percentile-bootstrap interval."""

    statistic: Statistic
    point: float
    lo: float
    hi: float
    confidence: float
    n_samples: int
    n_resamples: int

    def __str__(self) -> str:  
        pct = round(self.confidence * 100)
        return (
            f"{self.statistic}={self.point:.4g} "
            f"[{pct}% CI {self.lo:.4g}, {self.hi:.4g}] (n={self.n_samples})"
        )


_STATS: dict[Statistic, Callable[[Sequence[float]], float]] = {
    "median": lambda xs: float(median(xs)),
    "mean": lambda xs: float(sum(xs) / len(xs)),
}


def bootstrap_ci(
    values: Sequence[float],
    statistic: Statistic = "median",
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile-bootstrap CI for a summary statistic of ``values``.

    Raises:
        ValueError: empty input; NaN anywhere; non-finite values with
            ``statistic="mean"``; confidence outside (0, 1).
    """
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("cannot bootstrap an empty sample")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    if any(math.isnan(v) for v in vals):
        raise ValueError("sample contains NaN; upstream metrics should never emit NaN")
    if statistic == "mean" and any(math.isinf(v) for v in vals):
        raise ValueError(
            "sample contains inf; a mean over inf is meaningless — "
            "use statistic='median' for specificity ratios"
        )
    if n_resamples < 100:
        raise ValueError(f"n_resamples too small to be meaningful: {n_resamples}")

    stat = _STATS[statistic]
    n = len(vals)
    gen = torch.Generator().manual_seed(seed)
    t = torch.tensor(vals, dtype=torch.float64)


    idx = torch.randint(0, n, (n_resamples, n), generator=gen)
    samples = t[idx]
    boot = samples.median(dim=1).values if statistic == "median" else samples.mean(dim=1)

    alpha = (1.0 - confidence) / 2.0
    lo = float(torch.quantile(boot, alpha).item())
    hi = float(torch.quantile(boot, 1.0 - alpha).item())
    return BootstrapCI(
        statistic=statistic,
        point=stat(vals),
        lo=lo,
        hi=hi,
        confidence=confidence,
        n_samples=n,
        n_resamples=n_resamples,
    )
