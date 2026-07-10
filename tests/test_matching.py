"""Unit + property tests for signed cosine matching."""
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from sae_causal_audit.matching import MatchResult, match_features_to_atoms


class TestMatchResult:
    def test_rejects_invalid_sign(self):
        with pytest.raises(ValueError, match="sign"):
            MatchResult(feature_idx=0, atom_idx=0, cosine=0.5, sign=0.5)

    def test_rejects_out_of_range_cosine(self):
        with pytest.raises(ValueError, match="cosine"):
            MatchResult(feature_idx=0, atom_idx=0, cosine=1.5, sign=1.0)


class TestMatching:
    def test_identity_dictionary_matches_perfectly(self):
        d = torch.eye(4)
        res = match_features_to_atoms(d, d)
        assert [r.atom_idx for r in res] == [0, 1, 2, 3]
        assert all(r.cosine == pytest.approx(1.0) for r in res)
        assert all(r.sign == 1.0 for r in res)

    def test_antialigned_atom_reports_negative_sign_and_high_cosine(self):
        """The sign-bug regression test: a perfect *negation* must match
        with cosine 1.0 and sign -1.0 — never silently as +1."""
        direction = torch.tensor([[1.0, 2.0, -3.0]])
        dictionary = torch.stack([-direction[0], torch.tensor([0.0, 0.0, 1.0])])
        (res,) = match_features_to_atoms(direction, dictionary)
        assert res.atom_idx == 0
        assert res.cosine == pytest.approx(1.0, abs=1e-6)
        assert res.sign == -1.0

    def test_dead_atom_never_wins(self):
        direction = torch.tensor([[1.0, 0.0]])
        dictionary = torch.tensor([[0.0, 0.0], [0.7, 0.7]])  # dead atom first
        (res,) = match_features_to_atoms(direction, dictionary)
        assert res.atom_idx == 1

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            match_features_to_atoms(torch.ones(2, 3), torch.ones(4, 5))

    def test_empty_inputs_raise(self):
        with pytest.raises(ValueError, match="non-empty"):
            match_features_to_atoms(torch.ones(0, 3), torch.ones(4, 3))

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            match_features_to_atoms(torch.ones(3), torch.ones(4, 3))


@settings(max_examples=50, deadline=None)
@given(
    n_feat=st.integers(1, 6),
    d_sae=st.integers(1, 10),
    d_in=st.integers(2, 8),
    seed=st.integers(0, 10_000),
)
def test_matching_invariants(n_feat, d_sae, d_in, seed):
    """Properties that must hold for ANY input: cosine in [0,1], sign ±1,
    indices in range, output length == n_features, sign consistency with
    the actual signed cosine."""
    gen = torch.Generator().manual_seed(seed)
    dirs = torch.randn(n_feat, d_in, generator=gen)
    dic = torch.randn(d_sae, d_in, generator=gen)
    res = match_features_to_atoms(dirs, dic)
    assert len(res) == n_feat
    for r in res:
        assert 0 <= r.atom_idx < d_sae
        assert 0.0 <= r.cosine <= 1.0 + 1e-6
        assert r.sign in (1.0, -1.0)
        # sign must agree with the true signed cosine of the chosen pair
        a, b = dirs[r.feature_idx], dic[r.atom_idx]
        denom = (a.norm() * b.norm()).clamp_min(1e-12)
        signed = float((a @ b) / denom)
        if abs(signed) > 1e-6:
            assert (signed > 0) == (r.sign > 0)
