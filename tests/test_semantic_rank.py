"""Semantic PR / eRank of the topic sets (Fig 3B)."""

import numpy as np
import pytest


def test_metrics_of_two_leaning_axes_match_fig3_toy():
    """Two unit topic axes 35 degrees apart: PR = 4 / (2 + 2 cos^2) = 1.20,
    the value printed in the Fig 3A toy; orthogonal axes give 2."""
    from analysis.semantic.semantic_rank import metrics

    a = np.radians(35)
    m = metrics(np.array([[1.0, 0.0], [np.cos(a), np.sin(a)]]))
    assert m["PR"] == pytest.approx(4 / (2 + 2 * np.cos(a) ** 2))
    assert round(m["PR"], 2) == 1.20
    assert metrics(np.eye(2))["PR"] == pytest.approx(2.0)
    assert metrics(np.eye(2))["eRank"] == pytest.approx(2.0)


def test_rebuilt_centroids_match_stored_and_results_cover_two_countries():
    """The main keyword sets rebuild the stored centroids (cosine 1), and the
    results hold the four U.S. / Chinese sources only."""
    from analysis.semantic.semantic_rank import SOURCES, load_results

    res = {k: v for k, v in load_results().items() if not k.startswith("_")}
    if not res:
        pytest.skip("semantic_rank has not been run")
    assert set(res) == set(SOURCES) == {"anes", "wvs", "twitter", "weibo"}
    for r in res.values():
        assert r["min_cos_rebuilt_vs_stored"] > 0.9999
        lo, hi = r["range"]["PR"]
        assert lo <= r["main"]["PR"] <= hi
