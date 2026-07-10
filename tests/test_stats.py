"""Tests for bootstrap CIs, including inf-handling policy."""
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sae_causal_audit.stats import bootstrap_ci


class TestBootstrapCI:
    def test_constant_sample_gives_degenerate_interval(self):
        ci = bootstrap_ci([5.0] * 20, statistic="median", seed=1)
        assert ci.point == 5.0
        assert ci.lo == 5.0 and ci.hi == 5.0

    def test_deterministic_under_seed(self):
        vals = [1.0, 2.0, 3.0, 10.0, 20.0]
        a = bootstrap_ci(vals, seed=42)
        b = bootstrap_ci(vals, seed=42)
        assert (a.lo, a.hi) == (b.lo, b.hi)

    def test_median_tolerates_inf(self):
        vals = [1.0, 2.0, 3.0, 4.0, float("inf")]
        ci = bootstrap_ci(vals, statistic="median", seed=0)
        assert math.isfinite(ci.point)

    def test_mean_rejects_inf(self):
        with pytest.raises(ValueError, match="inf"):
            bootstrap_ci([1.0, float("inf")], statistic="mean")

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            bootstrap_ci([1.0, float("nan")])

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            bootstrap_ci([])

    def test_rejects_bad_confidence(self):
        with pytest.raises(ValueError, match="confidence"):
            bootstrap_ci([1.0, 2.0], confidence=1.5)


@settings(max_examples=30, deadline=None)
@given(
    vals=st.lists(st.floats(-1e6, 1e6), min_size=2, max_size=40),
    seed=st.integers(0, 1000),
)
def test_ci_contains_ordering_invariants(vals, seed):
    """lo <= hi always; interval brackets are within sample range for the
    percentile bootstrap of a median."""
    ci = bootstrap_ci(vals, statistic="median", n_resamples=500, seed=seed)
    assert ci.lo <= ci.hi
    assert min(vals) - 1e-9 <= ci.lo
    assert ci.hi <= max(vals) + 1e-9
