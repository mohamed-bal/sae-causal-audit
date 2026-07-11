"""Integration tests: train the actual toy setting end-to-end and assert
the audit reproduces the article's qualitative findings.

These are the package's ground-truth calibration: if the pipeline can't
recover the known answers in a setting where the answers ARE known, no
number it produces on a real model deserves trust.

Kept CPU-cheap (seconds, not minutes) by using reduced steps; module-
scoped fixtures train once and share across tests.
"""
import json

import pytest
import torch

from sae_causal_audit import (
    AuditConfig,
    load_json,
    render_markdown,
    run_audit,
    save_json,
)
from sae_causal_audit.toy import (
    ToyConfig,
    ToyProbe,
    train_topk_sae,
    train_toy_model,
    true_directions,
    well_represented_mask,
)


@pytest.fixture(scope="module")
def toy_model():
    return train_toy_model(ToyConfig(seed=0), steps=2500)


@pytest.fixture(scope="module")
def good_sae(toy_model):
    return train_topk_sae(toy_model, d_sae=128, k=4, steps=2500, seed=0)


@pytest.fixture(scope="module")
def good_report(toy_model, good_sae):
    mask = well_represented_mask(toy_model)
    dirs = true_directions(toy_model)[mask]
    dirs_all = true_directions(toy_model)
    report = run_audit(
        sae=good_sae,
        downstream=toy_model.output,
        probe=ToyProbe(toy_model, seed=0),
        true_directions=dirs_all,
        config=AuditConfig(n_samples_on=200, n_samples_off=200, seed=0),
        metadata={"regime": "toy", "sae": "topk_k4"},
    )
    return report, mask, dirs


class TestGoodSAEQualitative:
    def test_recovered_features_are_mostly_causally_specific(self, good_report):
        report, mask, _ = good_report
        keep = [r for r in report.results if mask[r.feature_idx] and r.cosine >= 0.9]
        assert len(keep) >= 10, "toy setting should recover most represented features"
        firing = [r for r in keep if not r.causally_inert]
        
        assert len(firing) / len(keep) >= 0.7
        specs = sorted(r.ablation_specificity for r in firing)
        assert specs[len(specs) // 2] > 5.0, "median specificity should be clearly > 1"

    def test_steering_rises_are_nonnegative_for_firing_matches(self, good_report):
        report, mask, _ = good_report
        keep = [
            r
            for r in report.results
            if mask[r.feature_idx] and r.cosine >= 0.9 and not r.causally_inert
        ]

        positive = [r for r in keep if r.steering_targeted_rise > -1e-6]
        assert len(positive) / len(keep) >= 0.8

    def test_census_math_is_internally_consistent(self, good_report):
        report, _, _ = good_report
        c = report.census
        assert c.n_recovered <= c.n_matched
        assert c.n_recovered_inert <= c.n_recovered
        if c.n_recovered:
            assert c.inert_rate_among_recovered == pytest.approx(
                c.n_recovered_inert / c.n_recovered
            )


class TestReportRoundTrip:
    def test_json_roundtrip_and_determinism(self, good_report, tmp_path):
        report, _, _ = good_report
        p1 = save_json(report, tmp_path / "a.json")
        p2 = save_json(report, tmp_path / "b.json")
        assert p1.read_text() == p2.read_text(), "serialization must be deterministic"
        loaded = load_json(p1)
        assert loaded["schema_version"] == report.schema_version
        assert len(loaded["results"]) == len(report.results)

    def test_json_is_strict(self, good_report, tmp_path):
        report, _, _ = good_report
        p = save_json(report, tmp_path / "r.json")
        json.loads(p.read_text())  

    def test_markdown_renders(self, good_report):
        report, _, _ = good_report
        md = render_markdown(report)
        assert "inert census" in md
        assert "| feat |" in md


class TestAuditRejectsBrokenSetups:
    def test_mismatched_dimensions_fail_loudly(self, toy_model, good_sae):
        bad_dirs = torch.randn(5, toy_model.cfg.n_hidden + 1)
        with pytest.raises(ValueError):
            run_audit(
                sae=good_sae,
                downstream=toy_model.output,
                probe=ToyProbe(toy_model),
                true_directions=bad_dirs,
            )
