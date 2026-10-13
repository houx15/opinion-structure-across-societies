"""Tests for analysis.correlation.semantic_adjustment (Fig 2G, Fig 2s)."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.correlation import semantic_adjustment as sa


def test_density_ratio_weights_uniform_for_identical_similarities():
    x = np.linspace(0.1, 0.5, 36)
    w = sa.density_ratio_weights(x, x)
    assert w.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(w, 1 / 36)


def test_density_ratio_weights_upweight_pairs_where_reference_is_dense():
    rng = np.random.default_rng(0)
    x_ref = rng.normal(0.4, 0.03, 200)
    x_other = np.r_[rng.normal(0.4, 0.03, 50), rng.normal(0.2, 0.03, 50)]
    w = sa.density_ratio_weights(x_ref, x_other)
    assert w[:50].sum() > 0.9


@pytest.mark.parametrize("method", sa.ADJUSTMENT_METHODS)
def test_adjusted_mean_recovers_reference_when_r_depends_only_on_similarity(method):
    """Two societies with the same |r|-similarity curve but different similarity
    mixes: after reweighting, the other society's mean equals the reference's."""
    rng = np.random.default_rng(1)
    x_ref = rng.uniform(0.3, 0.6, 400)
    x_other = rng.uniform(0.1, 0.6, 400)
    f = lambda x: 0.5 * x  # noqa: E731
    adj = sa.adjusted_mean(x_ref, x_other, f(x_other), method=method)
    assert f(x_other).mean() < f(x_ref).mean() - 0.04
    assert adj == pytest.approx(f(x_ref).mean(), abs=0.01)


def test_adjusted_mean_of_reference_against_itself_is_its_mean():
    x = np.linspace(0.1, 0.5, 36)
    y = np.linspace(0.0, 0.3, 36)
    for method in sa.ADJUSTMENT_METHODS:
        assert sa.adjusted_mean(x, x, y, method=method) == pytest.approx(y.mean(), abs=1e-9)


def test_adjusted_mean_rejects_unknown_method():
    with pytest.raises(ValueError):
        sa.adjusted_mean(np.ones(3), np.ones(3), np.ones(3), method="magic")


def test_gap_decomposition_parts_add_up():
    rng = np.random.default_rng(2)
    xr, xo = rng.uniform(0.2, 0.5, 36), rng.uniform(0.1, 0.5, 36)
    yr, yo = rng.uniform(0.2, 0.5, 36), rng.uniform(0.0, 0.2, 36)
    d = sa.gap_decomposition(xr, yr, xo, yo)
    assert d["raw_gap"] == pytest.approx(yr.mean() - yo.mean())
    assert d["similarity_part"] + d["society_part"] == pytest.approx(d["raw_gap"])


def test_shapley_shares_sum_to_full_r2():
    rng = np.random.default_rng(3)
    groups = np.repeat([0, 1, 2], 36)
    x = rng.uniform(0.1, 0.5, 108) + 0.05 * groups
    y = 0.3 * (groups == 0) + 0.4 * x + rng.normal(0, 0.05, 108)
    s = sa.shapley_r2(y, groups, x)
    assert s["society"] + s["similarity"] == pytest.approx(s["total"])
    assert s["unexplained"] == pytest.approx(1 - s["total"])
    assert 0 < s["similarity"] < s["society"]


def test_shapley_gives_similarity_nothing_when_r_is_set_by_society_alone():
    rng = np.random.default_rng(4)
    groups = np.repeat([0, 1, 2], 200)
    x = rng.uniform(0.1, 0.5, 600)
    y = np.choose(groups, [0.4, 0.1, 0.1]) + rng.normal(0, 0.01, 600)
    s = sa.shapley_r2(y, groups, x)
    assert s["society"] > 0.95
    assert s["similarity"] < 0.01


def test_matched_similarities_map_rank_k_to_the_kth_reference_value():
    x_ref = np.array([0.1, 0.4, 0.2, 0.3])
    x_other = np.array([0.9, 0.5, 0.7, 0.6])
    np.testing.assert_allclose(sa.matched_similarities(x_ref, x_other), [0.4, 0.1, 0.3, 0.2])


def test_ols_shift_keeps_residuals_and_moves_mean_by_slope_times_similarity_gap():
    rng = np.random.default_rng(5)
    x_ref, x_other = rng.uniform(0.3, 0.6, 36), rng.uniform(0.1, 0.4, 36)
    y_other = 0.2 * x_other + rng.normal(0, 0.02, 36)
    b, a = np.polyfit(x_other, y_other, 1)
    shifted = sa.shifted_values(x_ref, x_other, y_other, fit="ols")
    assert shifted.mean() == pytest.approx(y_other.mean() + b * (x_ref.mean() - x_other.mean()))
    resid_before = y_other - (a + b * x_other)
    resid_after = shifted - (a + b * sa.matched_similarities(x_ref, x_other))
    np.testing.assert_allclose(resid_after, resid_before)


def test_lowess_shift_is_flat_beyond_the_observed_range():
    x_other = np.linspace(0.2, 0.4, 36)
    y_other = np.linspace(0.0, 0.2, 36)
    x_ref = np.full(36, 0.9)
    shifted = sa.shifted_values(x_ref, x_other, y_other, fit="lowess")
    f = sa.fitted_curve(x_other, y_other, "lowess")
    np.testing.assert_allclose(shifted, y_other + f(0.4) - f(x_other))


def test_fitted_curve_rejects_unknown_fit():
    with pytest.raises(ValueError):
        sa.fitted_curve(np.ones(3), np.ones(3), fit="spline")
