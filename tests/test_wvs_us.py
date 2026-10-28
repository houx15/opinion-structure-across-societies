"""Tests for robustness/wvs_us.py (robustness only: WVS wave 7 U.S. survey source)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robustness import wvs_us

RAW = os.environ.get(
    "WVS7_CSV",
    "/Users/houyuxin/03SchoolLearning/05database/World Value Survey/WVS_Cross-National_Wave_7_csv_v6_0.csv",
)
requires_raw = pytest.mark.skipif(not Path(RAW).exists(), reason="WVS wave 7 csv not present")


def test_recoding_rules():
    s = pd.Series([1, 2, 3, 10, -1, -2, 4])
    np.testing.assert_array_equal(wvs_us.ten_to_five(s).to_numpy(), [1, 1, 2, 5, np.nan, np.nan, 2])
    np.testing.assert_array_equal(wvs_us.ten_to_five(s, reverse=True).to_numpy(), [5, 5, 4, 1, np.nan, np.nan, 4])
    np.testing.assert_array_equal(wvs_us.recode(pd.Series([1, 2, 3, 4, -1]), "confidence").to_numpy(),
                                  [1, 2, 4, 5, np.nan])
    np.testing.assert_array_equal(wvs_us.recode(pd.Series([1, 2, 3, -2]), "environment").to_numpy(),
                                  [1, 5, 3, np.nan])


def test_topic_set_mirrors_europe_without_gun():
    from common.topics import WVS_US_TOPICS
    from plotting.figures import OFFLINE_ONLINE_TOPIC_MAP_BY_SURVEY, RESTRICTED_TOPICS

    assert list(wvs_us.ITEMS) == WVS_US_TOPICS and len(WVS_US_TOPICS) == 9
    codes = set(OFFLINE_ONLINE_TOPIC_MAP_BY_SURVEY["wvs_us"].values())
    assert codes == set(RESTRICTED_TOPICS["twitter_euset"]) == set(RESTRICTED_TOPICS["eutwitter"])
    assert "gun" not in codes
    # the main U.S. platform source keeps gun control
    assert "gun" in RESTRICTED_TOPICS["twitter"]


@requires_raw
def test_rules_rebuild_the_stored_china_file_exactly():
    assert all(v == 1.0 for v in wvs_us.validate_china(RAW).values())


def test_robustness_task_skips_until_platform_source_exists(tmp_figure_folder):
    import plotting.figures as v4

    if (v4.CORRELATION_DIR / "network_analysis_twitter_euset.csv").exists():
        pytest.skip("platform source present")
    assert v4.run_task("robust_12_wvs_us", figure_folder=str(tmp_figure_folder), n_bootstrap=10) == {}
