"""Unit tests for causal metrics using hand-constructed SAEs where every
expected number is derivable by hand — the same standard the article
applied to feature_dimensionality."""
import math

import pytest
import torch

from sae_causal_audit.matching import MatchResult
from sae_causal_audit.metrics import (
    ablation_effect,
    causal_result_for_match,
    fired_fraction,
    steering_effect,
)


class IdentitySAE(torch.nn.Module):
    """encode = identity on first d dims (codes == activations), decode = identity.
    d_sae == d_in, W_dec = I. Fully analyzable by hand."""

    def __init__(self, d: int):
        super().__init__()
        self.W_dec = torch.eye(d)

    def encode(self, h):
        return h.clone()

    def decode(self, f):
        return f.clone()


class NeverFiresSAE(IdentitySAE):
    """Identity SAE whose atom 0 is structurally dead: encode zeroes it.
    Models the article's decoder-right/encoder-silent failure mode."""

    def encode(self, h):
        f = h.clone()
        f[:, 0] = 0.0
        return f


def identity_downstream(h_hat):
    return h_hat


class TestFiredFraction:
    def test_always_fires(self):
        sae = IdentitySAE(3)
        acts = torch.ones(10, 3)
        assert fired_fraction(sae, acts, atom_idx=0) == 1.0

    def test_never_fires(self):
        sae = NeverFiresSAE(3)
        acts = torch.ones(10, 3)
        assert fired_fraction(sae, acts, atom_idx=0) == 0.0

    def test_partial(self):
        sae = IdentitySAE(2)
        acts = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
        assert fired_fraction(sae, acts, atom_idx=0) == pytest.approx(0.5)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            fired_fraction(IdentitySAE(2), torch.ones(0, 2), 0)

    def test_bad_atom_idx_raises(self):
        with pytest.raises(IndexError):
            fired_fraction(IdentitySAE(2), torch.ones(3, 2), 5)


class TestAblation:
    def test_perfectly_surgical_ablation_is_inf(self):
        """Identity SAE, input only on dim 0: ablating atom 0 drops dim 0
        by exactly its value and moves nothing else -> off-target 0,
        targeted > 0 -> specificity inf."""
        sae = IdentitySAE(3)
        acts = torch.zeros(8, 3)
        acts[:, 0] = 2.0
        spec, drop, off, fired = ablation_effect(sae, identity_downstream, acts, 0, 0)
        assert math.isinf(spec)
        assert drop == pytest.approx(2.0)
        assert off == pytest.approx(0.0)
        assert fired == 1.0

    def test_dead_atom_reports_exact_zero_with_zero_fired_frac(self):
        """The 17/22 mechanism: atom never fires -> ablation changes nothing
        -> specificity exactly 0.0, and fired_frac 0.0 explains WHY."""
        sae = NeverFiresSAE(3)
        acts = torch.zeros(8, 3)
        acts[:, 0] = 2.0
        spec, drop, off, fired = ablation_effect(sae, identity_downstream, acts, 0, 0)
        assert spec == 0.0
        assert drop == 0.0
        assert off == 0.0
        assert fired == 0.0

    def test_off_target_collateral_reduces_specificity(self):
        """Downstream mixes atom 0 into both readout dims equally:
        targeted drop == off-target movement -> specificity 1.0."""
        sae = IdentitySAE(2)
        mix = torch.tensor([[1.0, 1.0], [0.0, 1.0]])  

        def mixing_downstream(h_hat):
            return h_hat @ mix

        acts = torch.zeros(4, 2)
        acts[:, 0] = 3.0
        spec, drop, off, _ = ablation_effect(sae, mixing_downstream, acts, 0, 0)
        assert drop == pytest.approx(3.0)
        assert off == pytest.approx(3.0)
        assert spec == pytest.approx(1.0)


class TestSteering:
    def test_sign_is_mandatory_and_validated(self):
        sae = IdentitySAE(2)
        with pytest.raises(ValueError, match="sign"):
            steering_effect(sae, identity_downstream, torch.zeros(4, 2), 0, 0, sign=0.0)

    def test_wrong_sign_through_relu_reproduces_the_original_bug(self):
        """Anti-aligned atom + ReLU downstream: steering with the WRONG
        (+1) sign yields exactly 0 targeted rise; the CORRECT (-1) sign
        yields a positive rise. This encodes the article's sign bug as a
        permanent regression test."""
        d = 2

        class AntiAlignedSAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.W_dec = torch.tensor([[-1.0, 0.0], [0.0, 1.0]])

            def encode(self, h):
                return torch.zeros(h.shape[0], d)

            def decode(self, f):
                return f @ self.W_dec

        def relu_downstream(h_hat):
            return torch.relu(h_hat)

        sae = AntiAlignedSAE()
        acts_off = torch.zeros(6, d)
        spec_wrong, rise_wrong, _off_w = steering_effect(
            sae, relu_downstream, acts_off, 0, 0, sign=1.0
        )
        del spec_wrong
        assert rise_wrong == pytest.approx(0.0)
        spec_right, rise_right, _ = steering_effect(
            sae, relu_downstream, acts_off, 0, 0, sign=-1.0
        )
        assert rise_right == pytest.approx(1.0)
        assert math.isinf(spec_right)

    def test_magnitude_must_be_positive(self):
        sae = IdentitySAE(2)
        with pytest.raises(ValueError, match="magnitude"):
            steering_effect(
                sae, identity_downstream, torch.zeros(4, 2), 0, 0, sign=1.0, magnitude=0.0
            )


class TestCausalResultForMatch:
    def test_full_battery_and_inert_flag(self):
        sae = NeverFiresSAE(3)
        match = MatchResult(feature_idx=0, atom_idx=0, cosine=0.95, sign=1.0)
        acts_on = torch.zeros(8, 3)
        acts_on[:, 0] = 1.0
        acts_off = torch.zeros(8, 3)
        r = causal_result_for_match(
            sae, identity_downstream, match, acts_on, acts_off
        )
        assert r.causally_inert  # cosine 0.95 yet inert — the headline case
        assert r.ablation_specificity == 0.0
        assert r.cosine == 0.95

    def test_nan_in_downstream_is_rejected(self):
        sae = IdentitySAE(2)

        def nan_downstream(h_hat):
            out = h_hat.clone()
            out[:, 0] = float("nan")
            return out

        with pytest.raises(ValueError, match="NaN"):
            ablation_effect(sae, nan_downstream, torch.ones(4, 2), 0, 0)
