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


def test_eight_topic_sources_share_concepts_and_leave_main_untouched():
    from common.topics import TOPICS_8, WVS_US_TOPICS
    from plotting.figures import OFFLINE_ONLINE_TOPIC_MAP, OFFLINE_ONLINE_TOPIC_MAP_BY_SURVEY, RESTRICTED_TOPICS

    assert list(wvs_us.ITEMS) == WVS_US_TOPICS and len(WVS_US_TOPICS) == 9
    assert all(len(v) == 8 for v in TOPICS_8.values())
    # U.S. survey and both Twitter/X sources cover the same eight concepts
    us_codes = set(OFFLINE_ONLINE_TOPIC_MAP_BY_SURVEY["wvs_us8"].values())
    assert us_codes == set(TOPICS_8["twitter_8"]) == set(TOPICS_8["entwitter_8"])
    # robust_13: ANES without gun control covers the same eight concepts
    assert set(OFFLINE_ONLINE_TOPIC_MAP_BY_SURVEY["anes_8"].values()) == us_codes
    assert set(OFFLINE_ONLINE_TOPIC_MAP_BY_SURVEY["anes_8"]) == set(TOPICS_8["anes_8"])
    # China drops attitudes toward the U.S. in both media
    assert "Foreign" not in TOPICS_8["wvs_8"] and 0 not in TOPICS_8["weibo_8"]
    assert OFFLINE_ONLINE_TOPIC_MAP["CN"]["Foreign"] == 0
    # the main sources keep their nine topics
    assert "gun" in RESTRICTED_TOPICS["twitter"] and "swe" in RESTRICTED_TOPICS["entwitter"]
    assert len(RESTRICTED_TOPICS["wvs"]) == len(RESTRICTED_TOPICS["anes"]) == 9


def test_eight_topic_spectra_drop_the_right_column():
    from robustness import topics8

    assert topics8.keep_columns("twitter-us_8") == [0, 2, 3, 4, 5, 6, 7, 8]   # gun is column 1
    assert topics8.keep_columns("twitter-en_8") == [0, 2, 3, 4, 5, 6, 7, 8]   # swe is column 1
    assert topics8.keep_columns("weibo_8") == list(range(8))                  # topic 0 is last
    assert topics8.keep_columns("wvs_8") == list(range(8))                    # Foreign is last
    assert topics8.keep_columns("anes_8") == [0, 2, 3, 4, 5, 6, 7, 8]          # Gun is column 1


@requires_raw
def test_rules_rebuild_the_stored_china_file_exactly():
    assert all(v == 1.0 for v in wvs_us.validate_china(RAW).values())


def test_robustness_task_runs_fig2_and_skips_fig3_without_social_spectra(tmp_figure_folder):
    import plotting.figures as v4

    if (v4.dimension_results_dir("cov") / "twitter-us_8-none-summary.json").exists():
        pytest.skip("social eight-topic spectra present")
    paths = v4.run_task("robust_12_wvs_us", figure_folder=str(tmp_figure_folder), n_bootstrap=10,
                        write_stats=False)
    assert list(paths) == [2]
