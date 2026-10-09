"""Tests for plotting.figures and its supporting export script.

Strict TDD for the figure pipeline: every behavior tested before its
implementation lands. Tests are grouped by phase (scaffolding, loaders, stats,
fig1, fig2, fig3, orchestration).

Tests for `export_user_opinion_data` lock in the parquet contract the user
will run on the upstream lustre data — synthetic fixtures so they can be
exercised here without the upstream files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.paths import OPINION_DIR


def test_module_imports():
    """Smoke test: plotting.figures must be importable."""
    import plotting.figures  # noqa: F401


# -- export_user_opinion_data ------------------------------------------------


@pytest.fixture()
def fake_weibo_root(tmp_path) -> Path:
    """Mimic <weibo_root>/<topic_id>/avg_opinion.parquet with year columns and 'average'."""
    import cleaning.twitter.opinion.export_user_opinion_data as eud

    root = tmp_path / "weibo_opinion_data"
    for topic_id, base in (
        (eud.WEIBO_LGBT_TOPIC_ID, 0.5),
        (eud.WEIBO_ENV_TOPIC_ID, -0.3),
    ):
        topic_dir = root / str(topic_id)
        topic_dir.mkdir(parents=True)
        df = pd.DataFrame(
            {
                "2020": [base, base + 0.1, np.nan, 5.0],
                "2021": [base + 0.2, np.nan, -0.4, -5.0],
                "average": [base + 0.1, base + 0.1, -0.4, 0.0],
            },
            index=pd.Index([f"u{i}" for i in range(4)], name="user_id"),
        )
        df.to_parquet(topic_dir / "avg_opinion.parquet")
    return root


@pytest.fixture()
def fake_twitter_root(tmp_path) -> Path:
    """Mimic <twitter_root>/merged-<topic>.parquet (flat — matches network_analysis_twitter.py)."""
    import cleaning.twitter.opinion.export_user_opinion_data as eud

    root = tmp_path / "twitter_opinion_data"
    root.mkdir(parents=True)
    for topic, base in (
        (eud.TWITTER_LGBT_TOPIC, 0.8),
        (eud.TWITTER_ENV_TOPIC, -0.6),
    ):
        df = pd.DataFrame(
            {
                "2018": [base, base + 0.1, np.nan, 0.0],
                "2019": [base - 0.1, np.nan, base + 0.2, 0.0],
                "2020": [np.nan, base + 0.3, base - 0.2, 0.0],
            },
            index=pd.Index([1001, 1002, 1003, 9999], name="user_id"),
        )
        df.to_parquet(root / f"merged-{topic}.parquet")
    return root


def test_export_picks_average_column_when_present(fake_weibo_root, tmp_path):
    """If the upstream parquet has an 'average' column, the export must use it."""
    import cleaning.twitter.opinion.export_user_opinion_data as eud

    out = eud.weibo(str(fake_weibo_root), output_dir=str(tmp_path / "out"))
    df = pd.read_parquet(out)
    assert set(df.columns) == {"user_id", "LGBT", "Environment"}
    # The 'average' column for LGBT topic stored base+0.1 = 0.6 for u0.
    row_u0 = df[df["user_id"] == "u0"].iloc[0]
    assert row_u0["LGBT"] == pytest.approx(0.6)
    assert row_u0["Environment"] == pytest.approx(-0.2)


def test_export_clips_to_negative_two_to_two(fake_weibo_root, tmp_path):
    """Out-of-range values in the upstream data must be clipped to [-2, 2]."""
    import cleaning.twitter.opinion.export_user_opinion_data as eud

    out = eud.weibo(str(fake_weibo_root), output_dir=str(tmp_path / "out"))
    df = pd.read_parquet(out)
    assert df["LGBT"].between(-2, 2).all()
    assert df["Environment"].between(-2, 2).all()


def test_export_falls_back_to_year_columns_when_no_average(fake_twitter_root, tmp_path):
    """Twitter fixture has only year columns; export must average them."""
    import cleaning.twitter.opinion.export_user_opinion_data as eud

    out = eud.twitter(str(fake_twitter_root), output_dir=str(tmp_path / "out"))
    df = pd.read_parquet(out)
    # User 1001 had LGBT (sxo) [0.8, 0.7, NaN] -> mean = 0.75.
    row = df[df["user_id"] == 1001].iloc[0]
    assert row["LGBT"] == pytest.approx(0.75)
    # User 1001 had Environment (clc) [-0.6, -0.7, NaN] -> mean = -0.65.
    assert row["Environment"] == pytest.approx(-0.65)


def test_export_twitter_reads_flat_layout(fake_twitter_root, tmp_path):
    """The new twitter layout is <root>/merged-<topic>.parquet, not <root>/<mode>/..."""
    import cleaning.twitter.opinion.export_user_opinion_data as eud

    out = Path(eud.twitter(str(fake_twitter_root), output_dir=str(tmp_path / "out")))
    df = pd.read_parquet(out)
    # All 4 fixture users have at least one of LGBT/Environment non-null.
    assert set(df["user_id"]) == {1001, 1002, 1003, 9999}


def test_export_eu_nentwitter_writes_expected_filename(fake_twitter_root, tmp_path):
    """The eu_nentwitter mode writes user_opinion_eu_nentwitter_lgbt_env.parquet."""
    import cleaning.twitter.opinion.export_user_opinion_data as eud

    out = Path(eud.eu_nentwitter(str(fake_twitter_root), output_dir=str(tmp_path / "out")))
    assert out.name == "user_opinion_eu_nentwitter_lgbt_env.parquet"
    assert out.exists() and out.stat().st_size > 0
    df = pd.read_parquet(out)
    assert {"user_id", "LGBT", "Environment"}.issubset(df.columns)


def test_eu_nentwitter_is_a_verify_target():
    """The eu_nentwitter src must be recognized by the verify helpers."""
    import cleaning.twitter.opinion.export_user_opinion_data as eud

    assert "eu_nentwitter" in eud._VERIFY_TARGET_TIDS


def test_export_twitter_applies_user_ids_allowlist(fake_twitter_root, tmp_path):
    """When a user_ids_file is provided, only listed users appear in the output."""
    import cleaning.twitter.opinion.export_user_opinion_data as eud
    import json as _json

    allowlist_path = tmp_path / "us_user_ids.json"
    _json.dump([1001, 1002], allowlist_path.open("w"))
    out = Path(
        eud.twitter(
            str(fake_twitter_root),
            user_ids_file=str(allowlist_path),
            output_dir=str(tmp_path / "out"),
        )
    )
    df = pd.read_parquet(out)
    assert set(df["user_id"]) == {1001, 1002}, (
        f"allowlist filter failed, got users {set(df['user_id'])}"
    )


def test_verify_weibo_matches_network_analysis_reference():
    """Recomputed |r| from the exported parquet must match data/network_analysis_weibo.csv."""
    import cleaning.twitter.opinion.export_user_opinion_data as eud

    from common.paths import OPINION_DIR
    parquet = OPINION_DIR / "user_opinion_weibo_lgbt_env.parquet"
    if not parquet.exists():
        pytest.skip("weibo export not present")
    result = eud.verify("weibo", tolerance=0.001)
    assert result["status"] == "OK"
    assert result["delta_r"] < 0.001
    assert result["n_ours"] == result["n_ref"]


def test_verify_raises_when_parquet_missing(tmp_path):
    """verify must error clearly when the exported parquet is missing."""
    import cleaning.twitter.opinion.export_user_opinion_data as eud

    with pytest.raises(FileNotFoundError):
        eud.verify("twitter", data_dir=str(tmp_path))


def test_export_drops_rows_with_neither_value(tmp_path):
    """Users with both LGBT and Environment missing must not appear in the output."""
    import cleaning.twitter.opinion.export_user_opinion_data as eud

    lgbt = pd.Series(
        {"u_keep": 0.3, "u_drop": np.nan, "u_lgbt_only": 0.5},
        name="LGBT",
    )
    env = pd.Series(
        {"u_keep": -0.2, "u_drop": np.nan, "u_env_only": -0.4},
        name="Environment",
    )
    df = eud._assemble_pair(lgbt, env)
    assert "u_drop" not in set(df["user_id"])
    assert {"u_keep", "u_lgbt_only", "u_env_only"} == set(df["user_id"])


def test_export_writes_parquet_with_expected_filename(fake_weibo_root, tmp_path):
    """Output parquet path must match data/user_opinion_<src>_lgbt_env.parquet."""
    import cleaning.twitter.opinion.export_user_opinion_data as eud

    out_dir = tmp_path / "out"
    out = eud.weibo(str(fake_weibo_root), output_dir=str(out_dir))
    assert Path(out).name == "user_opinion_weibo_lgbt_env.parquet"
    assert Path(out).parent == out_dir
    assert Path(out).exists() and Path(out).stat().st_size > 0


# -- prepare_survey_individual_data ------------------------------------------


@pytest.mark.parametrize(
    "src, expected_topics",
    [
        ("anes", {"Abortion", "Gun", "Climate", "LGBT", "VACC", "DeathPenalty", "Media", "MinimumWage", "UBI"}),
        ("wvs", {"Corrup", "GenderEqual", "Marriage", "Childbearing", "LGBT", "Environment", "Econ", "Work", "Foreign"}),
        ("evs_resample2", {"LGBT", "Abortion", "DeathPenalty", "SocialMedia", "Egalitarian", "Environment", "Prostitution", "HealthCare", "UnemplyAid"}),
        ("en_evs", {"LGBT", "Abortion", "DeathPenalty", "SocialMedia", "Egalitarian", "Environment", "Prostitution", "HealthCare", "UnemplyAid"}),
    ],
)
def test_individual_opinion_parquet_schema(src, expected_topics):
    """Each prepared parquet must have respondent_id + year + the source's 9 topics."""
    path = OPINION_DIR / f"individual_opinion_{src}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not prepared; run prepare_survey_individual_data.py first")
    df = pd.read_parquet(path)
    assert "respondent_id" in df.columns
    assert "year" in df.columns
    topic_cols = set(df.columns) - {"respondent_id", "year"}
    assert topic_cols == expected_topics, (
        f"{src} parquet has topics {topic_cols}, expected {expected_topics}"
    )
    assert len(df) > 0, f"{src} parquet is empty"


def test_en_evs_individual_opinion_is_filtered_to_great_britain():
    """en_evs prep must restrict to Great Britain (smaller than full evs_resample2)."""
    en_path = OPINION_DIR / "individual_opinion_en_evs.parquet"
    eu_path = OPINION_DIR / "individual_opinion_evs_resample2.parquet"
    if not en_path.exists() or not eu_path.exists():
        pytest.skip("prep parquets not present; run prepare_survey_individual_data.py")
    n_en = len(pd.read_parquet(en_path))
    n_eu = len(pd.read_parquet(eu_path))
    assert 0 < n_en < n_eu, (
        f"en_evs ({n_en}) should be smaller than evs_resample2 ({n_eu})"
    )


# -- v4 data loaders ---------------------------------------------------------


SURVEY_SOURCES = ["anes", "wvs", "evs_resample2", "en_evs"]
SOCIAL_SOURCES = ["twitter", "weibo", "eutwitter"]
ALL_PAIRWISE_SOURCES = SURVEY_SOURCES + SOCIAL_SOURCES


@pytest.mark.parametrize("src", ALL_PAIRWISE_SOURCES)
def test_load_pairwise_correlation_returns_expected_columns(src):
    """Loader must return tid1, tid2, year, correlation (abs), pearson, intersection."""
    import plotting.figures as v4

    df = v4.load_pairwise_correlation(src)
    required = {"tid1", "tid2", "year", "correlation", "pearson", "intersection"}
    missing = required - set(df.columns)
    assert not missing, f"{src}: missing columns {missing}"
    assert len(df) > 0, f"{src}: returned empty frame"


@pytest.mark.parametrize("src", ALL_PAIRWISE_SOURCES)
def test_load_pairwise_correlation_is_non_negative(src):
    """The 'correlation' column is |r|, so always >= 0."""
    import plotting.figures as v4

    df = v4.load_pairwise_correlation(src)
    assert (df["correlation"] >= 0).all(), f"{src}: negative |r| found"
    assert (df["correlation"] <= 1.0001).all(), f"{src}: |r| > 1 found"


@pytest.mark.parametrize("src", ALL_PAIRWISE_SOURCES)
def test_load_pairwise_correlation_restricts_to_topics_of_interest(src):
    """Only the source's 9 topics should appear in tid1/tid2."""
    import plotting.figures as v4

    df = v4.load_pairwise_correlation(src)
    expected = set(v4.RESTRICTED_TOPICS[src])
    found_tids = set(df["tid1"]) | set(df["tid2"])
    extra = found_tids - expected
    assert not extra, f"{src}: unexpected topic ids {extra}"


def test_load_pairwise_correlation_average_year_only_by_default():
    """Default year filter is 'average' so consumers get one row per pair."""
    import plotting.figures as v4

    df = v4.load_pairwise_correlation("anes")
    assert (df["year"] == "average").all() or set(df["year"]) == {"average"}


@pytest.mark.parametrize("src", ALL_PAIRWISE_SOURCES)
def test_load_semantic_similarity_returns_expected_columns(src):
    """Semantic loader returns: tid1, tid2, topic_combination, similarity."""
    import plotting.figures as v4

    df = v4.load_semantic_similarity(src)
    required = {"tid1", "tid2", "topic_combination", "similarity"}
    missing = required - set(df.columns)
    assert not missing, f"{src}: missing {missing}"
    assert len(df) > 0


@pytest.mark.parametrize("src", ALL_PAIRWISE_SOURCES)
def test_load_semantic_similarity_in_unit_interval(src):
    """Cosine similarity for any pair is bounded in roughly [-1, 1]."""
    import plotting.figures as v4

    df = v4.load_semantic_similarity(src)
    s = df["similarity"].dropna()
    assert (s.between(-1.0001, 1.0001)).all(), (
        f"{src}: similarity out of [-1,1]: min={s.min()} max={s.max()}"
    )


@pytest.mark.parametrize("src", SURVEY_SOURCES)
def test_load_semantic_similarity_dictionary_returns_a_single_similarity_column(src):
    """Regression: dictionary embedding must not leave two 'similarity' columns.

    The survey topic_distance CSVs ship both 'similarity' (GPT) and
    'dictionary_similarity'. When we ask for dictionary embedding, the loader
    used to rename the wrong column without dropping the other, producing a
    duplicated 'similarity' column that broke ax.scatter downstream.
    """
    import plotting.figures as v4

    df = v4.load_semantic_similarity(src, embedding_type="dictionary")
    sim_view = df["similarity"]
    # Must be a 1-D Series, not a 2-column DataFrame from a duplicate-named column.
    assert sim_view.ndim == 1, (
        f"{src}: 'similarity' column was duplicated (shape={getattr(sim_view, 'shape', None)})"
    )


@pytest.mark.parametrize("src", ["anes", "wvs", "evs_resample2", "twitter", "weibo", "eutwitter"])
def test_load_spectral_summary_has_pr_and_erank(src):
    """Spectral loader returns dict with PR and eRank lists of floats."""
    import plotting.figures as v4

    summary = v4.load_spectral_summary(src)
    assert "PR" in summary and "eRank" in summary
    assert len(summary["PR"]) >= 1
    assert len(summary["eRank"]) >= 1
    assert all(isinstance(x, float) for x in summary["PR"])
    assert all(isinstance(x, float) for x in summary["eRank"])


@pytest.mark.parametrize("src", SURVEY_SOURCES)
def test_load_individual_opinion_has_topic_columns_and_respondent_id(src):
    """Individual-opinion loader returns respondent_id + the 9 topic columns."""
    import plotting.figures as v4

    df = v4.load_individual_opinion(src)
    assert "respondent_id" in df.columns
    topics = set(v4.RESTRICTED_TOPICS[src])
    assert topics.issubset(set(df.columns)), (
        f"{src}: missing topics {topics - set(df.columns)}"
    )
    assert len(df) > 0


@pytest.mark.parametrize("src", SOCIAL_SOURCES)
def test_load_user_opinion_pair_has_lgbt_and_env_when_available(src):
    """Social media user-level loader: user_id + LGBT + Environment when the file exists."""
    import plotting.figures as v4

    path = OPINION_DIR / f"user_opinion_{src}_lgbt_env.parquet"
    if not path.exists():
        pytest.skip(f"{path} not yet exported")
    df = v4.load_user_opinion_pair(src)
    assert {"user_id", "LGBT", "Environment"}.issubset(df.columns)
    assert len(df) > 0


# -- v4 statistical helpers --------------------------------------------------


def test_compute_abs_pearson_returns_absolute_value_and_n():
    """compute_abs_pearson(x, y) drops NaN pairs and returns (|r|, n_used)."""
    import plotting.figures as v4

    x = pd.Series([1, 2, 3, 4, 5, np.nan])
    y = pd.Series([2, 4, 6, 8, 10, 99])  # perfectly correlated where both defined
    abs_r, n = v4.compute_abs_pearson(x, y)
    assert abs_r == pytest.approx(1.0)
    assert n == 5


def test_compute_abs_pearson_returns_zero_correlation_for_independent():
    """Independent random data should give |r| close to 0."""
    import plotting.figures as v4

    rng = np.random.default_rng(42)
    x = rng.normal(size=2000)
    y = rng.normal(size=2000)
    abs_r, n = v4.compute_abs_pearson(x, y)
    assert abs_r < 0.1
    assert n == 2000


def test_bootstrap_mean_ci_contains_true_mean_for_normal_data():
    """A 95% bootstrap CI should cover the true mean for moderate samples."""
    import plotting.figures as v4

    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.5, scale=0.1, size=500)
    mean, lo, hi = v4.bootstrap_mean_ci(values, n_boot=1000, ci=0.95, rng=rng)
    assert lo <= mean <= hi
    # CI for n=500, sigma=0.1 has half-width ~ 0.009; should easily cover 0.5.
    assert lo <= 0.5 <= hi


def test_bootstrap_mean_ci_handles_empty_input():
    """Empty input must return nan, nan, nan rather than crashing."""
    import plotting.figures as v4

    mean, lo, hi = v4.bootstrap_mean_ci(np.array([]), n_boot=100, ci=0.95)
    assert np.isnan(mean) and np.isnan(lo) and np.isnan(hi)


def test_pca_loadings_2d_recovers_diagonal_direction():
    """For y = x + small noise, PC1 should point along the diagonal."""
    import plotting.figures as v4

    rng = np.random.default_rng(7)
    x = rng.normal(size=2000)
    noise = rng.normal(scale=0.05, size=2000)
    X = np.column_stack([x, x + noise])
    result = v4.pca_loadings(X)
    # eigenvalues sorted descending; first should dominate
    assert result["eigenvalues"][0] > 5 * result["eigenvalues"][1]
    # PC1 components both positive and ~ equal magnitude (sign may flip).
    pc1 = result["eigenvectors"][:, 0]
    assert np.sign(pc1[0]) == np.sign(pc1[1])
    assert abs(abs(pc1[0]) - abs(pc1[1])) < 0.05
    # explained variance ratio sums to 1 and PC1 carries most of it.
    assert result["explained_variance_ratio"][0] > 0.9
    assert sum(result["explained_variance_ratio"]) == pytest.approx(1.0)


def test_pca_loadings_listwise_deletes_rows_with_nan():
    """Rows containing NaN must be dropped before covariance computation."""
    import plotting.figures as v4

    X = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [np.nan, 4.0],  # dropped
            [4.0, 4.0],
        ]
    )
    result = v4.pca_loadings(X)
    # only 4 valid rows; PC1 direction is along (1,1) regardless.
    assert result["n_samples_used"] == 4
    pc1 = result["eigenvectors"][:, 0]
    assert np.sign(pc1[0]) == np.sign(pc1[1])


def test_participation_ratio_of_uniform_eigenvalues_equals_dimension():
    """If all eigenvalues equal, PR == number of eigenvalues."""
    import plotting.figures as v4

    assert v4.participation_ratio(np.ones(9)) == pytest.approx(9.0)


def test_exponential_rank_of_one_dominant_mode_is_one():
    """If only one eigenvalue is non-zero, eRank == 1."""
    import plotting.figures as v4

    lam = np.array([10.0, 0.0, 0.0])
    assert v4.exponential_rank(lam) == pytest.approx(1.0, abs=1e-6)


def test_participation_ratio_matches_dimension_data_for_anes():
    """PR computed from anes individual data must match dimension_data within 5%."""
    import plotting.figures as v4

    df = v4.load_individual_opinion("anes")
    topics = v4.RESTRICTED_TOPICS["anes"]
    X = df[list(topics)].to_numpy(dtype=float)
    result = v4.pca_loadings(X)
    pr = v4.participation_ratio(result["eigenvalues"])
    reference = v4.load_spectral_summary("anes")["PR"][0]
    # Listwise PCA vs upstream pairwise-cov differ slightly; allow 10% tolerance.
    assert pr == pytest.approx(reference, rel=0.10), (
        f"PR mismatch: ours={pr:.3f}, reference={reference:.3f}"
    )


def test_bootstrap_lowess_ci_returns_arrays_of_same_length():
    """LOWESS bootstrap returns four arrays (x_grid, mean, lo, hi) of the same shape."""
    import plotting.figures as v4

    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, size=200)
    y = 0.5 + 0.3 * x + rng.normal(scale=0.05, size=200)
    x_grid, mean, lo, hi = v4.bootstrap_lowess_ci(
        x, y, frac=0.4, n_bootstrap=50, confidence_level=0.95
    )
    assert x_grid is not None
    assert len(x_grid) == len(mean) == len(lo) == len(hi)
    # lo <= mean <= hi pointwise
    assert (lo <= mean).all()
    assert (mean <= hi).all()


# -- v4 Figure 1: case study (LGBT x Env across contexts) -------------------


@pytest.mark.parametrize(
    "src, expected_xy",
    [
        ("anes", ("LGBT", "Climate")),
        ("wvs", ("LGBT", "Environment")),
        ("evs_resample2", ("LGBT", "Environment")),
        ("en_evs", ("LGBT", "Environment")),
    ],
)
def test_get_case_study_pair_survey_returns_lgbt_and_env_columns(src, expected_xy):
    """Offline case-study extractor picks the LGBT and Env-like column per source."""
    import plotting.figures as v4

    df = v4.get_case_study_pair(src)
    assert set(df.columns) >= {"x", "y"}
    assert df.attrs["x_topic"] == expected_xy[0]
    assert df.attrs["y_topic"] == expected_xy[1]
    assert len(df) > 0
    assert df["x"].notna().any() and df["y"].notna().any()


def test_case_study_correlations_match_reported_values_for_surveys():
    """Computed |r| from individual data should match the spec's anchor numbers.

    Spec: US |r|=0.574, CN |r|=0.006, EU |r|=0.303 for the offline case.
    We compare loosely (within 0.03) since the spec is rounded to 3dp.
    """
    import plotting.figures as v4

    pairs = {"anes": 0.574, "wvs": 0.006, "evs_resample2": 0.303}
    for src, expected in pairs.items():
        df = v4.get_case_study_pair(src)
        abs_r, _ = v4.compute_abs_pearson(df["x"], df["y"])
        assert abs_r == pytest.approx(expected, abs=0.05), (
            f"{src}: |r| expected ~{expected:.3f}, got {abs_r:.3f}"
        )


def test_make_fig1_writes_pdf_with_expected_naming(tmp_figure_folder):
    """make_fig1 must save a non-empty PDF named YYYYMMDD_<task>_results_fig1.pdf."""
    import plotting.figures as v4

    plotter = v4.Plotter(
        figure_folder=str(tmp_figure_folder),
        date_prefix="20260513",
    )
    out_path = plotter.make_fig1(task_name="main")
    out_path = Path(out_path)
    assert out_path.exists() and out_path.stat().st_size > 0
    assert out_path.name == "20260513_main_results_fig1.pdf"


def test_make_fig1_dash_plot_has_no_offline_online_connector(tmp_figure_folder):
    """No diagonal connecting lines that span offline tick to online tick.

    Tiny leader lines (label staggering) are fine; what we forbid is a line
    that joins one dash at the Offline tick to another at the Online tick.
    """
    import plotting.figures as v4
    import matplotlib.pyplot as plt

    plotter = v4.Plotter(
        figure_folder=str(tmp_figure_folder), date_prefix="20260513"
    )
    fig = plotter.build_fig1()
    dash_ax = fig.axes[-1]
    # An offline-online connector spans most of the tick gap (0..1 in data
    # units). A 0.025-wide label-leader does not.
    cross_tick = [
        line for line in dash_ax.get_lines()
        if len(line.get_xdata()) == 2
        and abs(line.get_xdata()[1] - line.get_xdata()[0]) > 0.6
        and line.get_ydata()[0] != line.get_ydata()[1]
    ]
    assert cross_tick == [], "dash plot must not draw offline-online connectors"
    plt.close(fig)


def test_make_fig1_row1_each_panel_has_scatter_and_fitted_line(tmp_figure_folder):
    """Row1 offline panels each have at least one scatter collection and one line."""
    import plotting.figures as v4
    import matplotlib.pyplot as plt

    plotter = v4.Plotter(
        figure_folder=str(tmp_figure_folder), date_prefix="20260513"
    )
    fig = plotter.build_fig1()
    for ax in fig.axes[:3]:
        assert len(ax.collections) >= 1, "missing scatter in offline panel"
        assert len(ax.get_lines()) >= 1, "missing fitted line in offline panel"
    plt.close(fig)


# -- v4 Figure 2: generalization across all pairs ----------------------------


def test_make_fig2_writes_pdf_with_expected_naming(tmp_figure_folder):
    """make_fig2 writes a non-empty PDF named YYYYMMDD_<task>_results_fig2.pdf."""
    import plotting.figures as v4

    plotter = v4.Plotter(
        figure_folder=str(tmp_figure_folder), date_prefix="20260513",
        n_bootstrap=100,  # speed up bootstrap CIs in tests
    )
    out_path = Path(plotter.make_fig2(task_name="main"))
    assert out_path.exists() and out_path.stat().st_size > 0
    assert out_path.name == "20260513_main_results_fig2.pdf"


def test_make_fig2_loess_panel_has_no_pair_callouts(tmp_figure_folder):
    """LOWESS panels stay clean — callouts live in the dedicated examples row."""
    import plotting.figures as v4
    import matplotlib.pyplot as plt

    plotter = v4.Plotter(
        figure_folder=str(tmp_figure_folder), date_prefix="20260513",
        n_bootstrap=50,
    )
    fig = plotter.build_fig2()
    for ax in fig.axes[3:5]:
        annotations = [t for t in ax.texts if "×" in t.get_text()]
        assert annotations == [], (
            f"LOWESS panel must have no pair callouts; got {[t.get_text() for t in annotations]}"
        )
    plt.close(fig)


def test_word_overlap_rows_identical_words_match_at_top():
    """Identical strings have cos=1.0, so greedy matching surfaces them first."""
    import plotting.figures as v4
    import numpy as np

    # Three words, identical vectors so cos = 1.0 for matching pairs.
    embeddings = {
        "shared1": np.array([1.0, 0.0, 0.0]),
        "shared2": np.array([0.0, 1.0, 0.0]),
        "alpha":   np.array([0.5, 0.5, 0.0]),
        "beta":    np.array([0.6, 0.4, 0.0]),
    }
    rows = v4._compute_word_overlap_rows(
        [("alpha", 0.05), ("shared1", 0.04), ("shared2", 0.03)],
        [("beta", 0.05), ("shared1", 0.04), ("shared2", 0.03)],
        embeddings=embeddings,
    )
    pairs = [(r.left, r.right) for r in rows if r.kind == "matched"]
    # Two shared words sit at the top (in either order, both at sim = 1.0).
    assert {pairs[0], pairs[1]} == {("shared1", "shared1"), ("shared2", "shared2")}
    # alpha vs beta is the lower-similarity matched pair.
    assert pairs[2] == ("alpha", "beta")


def test_word_overlap_rows_match_pairs_by_descending_similarity():
    """Greedy 1:1 ranks candidates by cosine descending and consumes them in order."""
    import plotting.figures as v4
    import numpy as np

    embeddings = {
        "apple":   np.array([1.0, 0.0, 0.0]),
        "fruit":   np.array([0.99, 0.01, 0.0]),  # ~1.00 with apple
        "car":     np.array([0.0, 1.0, 0.0]),
        "vehicle": np.array([0.01, 0.99, 0.0]),  # ~1.00 with car
        "stone":   np.array([0.0, 0.0, 1.0]),
        "iron":    np.array([0.0, 0.0, 0.5]),    # 1.00 with stone
    }
    rows = v4._compute_word_overlap_rows(
        [("apple", 0.10), ("car", 0.08), ("stone", 0.06)],
        [("fruit", 0.07), ("vehicle", 0.05), ("iron", 0.03)],
        embeddings=embeddings,
    )
    matched = [(r.left, r.right) for r in rows if r.kind == "matched"]
    assert ("apple", "fruit") in matched
    assert ("car", "vehicle") in matched
    assert ("stone", "iron") in matched
    # Each row carries similarity, and rows are sorted descending.
    sims = [r.similarity for r in rows if r.kind == "matched"]
    assert sims == sorted(sims, reverse=True)


def test_word_overlap_rows_carry_weights_for_font_size():
    """Each row preserves the source TF-IDF weight per side for downstream sizing."""
    import plotting.figures as v4
    import numpy as np

    embeddings = {"x": np.array([1.0, 0.0]), "y": np.array([1.0, 0.0])}
    rows = v4._compute_word_overlap_rows(
        [("x", 0.07)], [("y", 0.02)], embeddings=embeddings,
    )
    matched = [r for r in rows if r.kind == "matched"]
    assert matched[0].left_weight == pytest.approx(0.07)
    assert matched[0].right_weight == pytest.approx(0.02)


def test_make_fig2_row2_word_overlap_matched_pairs_share_row(tmp_figure_folder):
    """Every matched (left, right) pair renders at the same y in its panel half."""
    import plotting.figures as v4
    import matplotlib.pyplot as plt

    plotter = v4.Plotter(
        figure_folder=str(tmp_figure_folder), date_prefix="20260513",
        n_bootstrap=50,
    )
    fig = plotter.build_fig2()
    row2_ax = fig.axes[2]

    high = v4._select_us_twitter_extremes()[1]
    wa = v4._load_twitter_top_words_with_weights(high[0])
    wb = v4._load_twitter_top_words_with_weights(high[1])
    rows = v4._compute_word_overlap_rows(wa, wb)
    matched_rows = [r for r in rows if r.kind == "matched" and r.left and r.right]
    assert matched_rows, "precondition: high pair must have at least one matched row"

    # Texts in the right half of the panel are the high pair; their y for the
    # left-side word and the right-side word must agree exactly.
    right_half_texts = [t for t in row2_ax.texts if t.get_position()[0] > 0.5]
    text_xy = [(t.get_text(), t.get_position()) for t in right_half_texts]

    for row in matched_rows:
        left_ys = {round(p[1], 4) for txt, p in text_xy if txt == row.left}
        right_ys = {round(p[1], 4) for txt, p in text_xy if txt == row.right}
        if not left_ys or not right_ys:
            continue
        common = left_ys & right_ys
        assert common, (
            f"matched pair ({row.left!r}, {row.right!r}) should share a y; "
            f"left_ys={left_ys}, right_ys={right_ys}"
        )


def test_make_fig2_row2_word_overlap_word_size_scales_with_tf_idf(tmp_figure_folder):
    """Higher-TF-IDF words render at larger font size than lower-TF-IDF words."""
    import plotting.figures as v4
    import matplotlib.pyplot as plt

    plotter = v4.Plotter(
        figure_folder=str(tmp_figure_folder), date_prefix="20260513",
        n_bootstrap=50,
    )
    fig = plotter.build_fig2()
    row2_ax = fig.axes[2]

    # Use the high pair: the topic's #1 TF-IDF word and the topic's #last
    # word both render; #1 should be visibly larger.
    high = v4._select_us_twitter_extremes()[1]
    weighted = v4._load_twitter_top_words_with_weights(high[0])
    top_word, top_weight = weighted[0]
    last_word, last_weight = weighted[-1]
    assert top_weight > last_weight, "precondition: top word should outrank the last"

    texts = {t.get_text(): t for t in row2_ax.texts}
    if top_word in texts and last_word in texts:
        assert texts[top_word].get_fontsize() > texts[last_word].get_fontsize(), (
            f"{top_word!r} (weight={top_weight:.4f}) should render larger than "
            f"{last_word!r} (weight={last_weight:.4f})"
        )


def test_offline_online_topic_map_covers_all_three_countries():
    """The researcher-supplied offline↔online topic map must cover US, CN, EU."""
    import plotting.figures as v4

    assert set(v4.OFFLINE_ONLINE_TOPIC_MAP) == {"US", "CN", "EU"}
    for country, mapping in v4.OFFLINE_ONLINE_TOPIC_MAP.items():
        assert len(mapping) == 9, f"{country}: expected 9 topic pairs, got {len(mapping)}"


def test_offline_online_topic_map_codomain_matches_restricted_topics():
    """Every online side of the mapping must appear in the platform's restricted set."""
    import plotting.figures as v4

    survey_for_country = {"US": "anes", "CN": "wvs", "EU": "evs_resample2"}
    platform_for_country = {"US": "twitter", "CN": "weibo", "EU": "eutwitter"}
    for country, mapping in v4.OFFLINE_ONLINE_TOPIC_MAP.items():
        offline_topics = set(v4.RESTRICTED_TOPICS[survey_for_country[country]])
        online_topics = set(v4.RESTRICTED_TOPICS[platform_for_country[country]])
        assert set(mapping.keys()) == offline_topics, (
            f"{country}: mapping keys mismatch survey topics"
        )
        assert set(mapping.values()) == online_topics, (
            f"{country}: mapping values mismatch online topics"
        )


def test_pretty_topic_maps_codes_and_ints():
    """_pretty_topic should handle survey strings, twitter codes, and weibo ints."""
    import plotting.figures as v4

    assert v4._pretty_topic("LGBT") == "LGBT"
    assert v4._pretty_topic("DeathPenalty") == "Death Pen"
    assert v4._pretty_topic("sxo") == "LGBT"
    assert v4._pretty_topic("clc") == "Climate"
    assert v4._pretty_topic(10) == "LGBT"
    assert v4._pretty_topic(13) == "Env"


def test_make_fig2_has_no_lgbt_env_star_markers(tmp_figure_folder):
    """The star highlight has been removed per design feedback."""
    import plotting.figures as v4
    import matplotlib.pyplot as plt

    plotter = v4.Plotter(
        figure_folder=str(tmp_figure_folder), date_prefix="20260513",
        n_bootstrap=50,
    )
    fig = plotter.build_fig2()
    for ax in fig.axes:
        starred = [c for c in ax.collections if getattr(c, "_is_lgbt_env_marker", False)]
        assert starred == [], "all LGBT*Env star markers should be removed"
    plt.close(fig)


# -- v4 Figure 3: spectral, PCA arrows, loadings heatmap --------------------


def test_topic_family_ordering_is_consistent_per_country():
    """Each country has its 9 topics partitioned into 3 families (Table 1)."""
    import plotting.figures as v4

    for src in ("anes", "wvs", "evs_resample2"):
        ordering = v4.topic_family_order(src)
        expected_topics = set(v4.RESTRICTED_TOPICS[src])
        assert set(ordering) == expected_topics, (
            f"{src}: family ordering missing topics {expected_topics - set(ordering)}"
        )
        assert len(ordering) == 9


def test_make_fig3_writes_pdf_with_expected_naming(tmp_figure_folder):
    """make_fig3 writes a non-empty PDF named YYYYMMDD_<task>_results_fig3.pdf."""
    import plotting.figures as v4

    plotter = v4.Plotter(
        figure_folder=str(tmp_figure_folder), date_prefix="20260513",
    )
    out_path = Path(plotter.make_fig3(task_name="main"))
    assert out_path.exists() and out_path.stat().st_size > 0
    assert out_path.name == "20260513_main_results_fig3.pdf"


def test_make_fig3_row1_all_panels_are_3d(tmp_figure_folder):
    """Every Row-1 panel uses the same 3D coordinate system."""
    import plotting.figures as v4
    import matplotlib.pyplot as plt

    plotter = v4.Plotter(
        figure_folder=str(tmp_figure_folder), date_prefix="20260513",
    )
    fig = plotter.build_fig3()
    for ax in fig.axes[:3]:
        assert ax.name == "3d", f"Row 1 panel should be 3D, got {ax.name}"
    plt.close(fig)


def test_make_fig3_row1_plate_panel_draws_supporting_plane(tmp_figure_folder):
    """The 'plate' panel must overlay a translucent plane (a Poly3DCollection)."""
    import plotting.figures as v4
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    plotter = v4.Plotter(
        figure_folder=str(tmp_figure_folder), date_prefix="20260513",
    )
    fig = plotter.build_fig3()
    plate_ax = fig.axes[1]  # second toy panel is the plate
    poly3d = [c for c in plate_ax.collections if isinstance(c, Poly3DCollection)]
    assert poly3d, "plate panel must draw a 3D plane (Poly3DCollection)"
    plt.close(fig)


def test_make_fig3_row1_line_panel_draws_axis_segment(tmp_figure_folder):
    """The 'line' panel must overlay a 3D segment showing the supporting line."""
    import plotting.figures as v4
    import matplotlib.pyplot as plt

    plotter = v4.Plotter(
        figure_folder=str(tmp_figure_folder), date_prefix="20260513",
    )
    fig = plotter.build_fig3()
    line_ax = fig.axes[2]  # third toy panel is the line
    # Among the Line3D objects, at least one should span the full x-range
    # at constant y=z=0.
    bound = plotter._TOY_AXIS_BOUND
    long_segments = [
        line for line in line_ax.get_lines()
        if len(line.get_xdata()) == 2
        and abs(line.get_xdata()[1] - line.get_xdata()[0]) >= 1.5 * bound
    ]
    assert long_segments, "line panel must include a long supporting segment"
    plt.close(fig)


def test_make_fig3_toy_pr_values_match_intent(tmp_figure_folder):
    """Toy clouds should produce PR ≈ 3, ≈2, ≈1 across the row."""
    import plotting.figures as v4

    plotter = v4.Plotter(figure_folder=str(tmp_figure_folder))
    expected_targets = [3.0, 2.0, 1.0]
    for (label, eigs, _rendering), expected in zip(
        plotter._TOY_CLOUDS, expected_targets
    ):
        pr_population = v4.participation_ratio(np.asarray(eigs))
        assert abs(pr_population - expected) <= 0.2, (
            f"{label}: population PR {pr_population:.2f} far from intended {expected}"
        )


def test_make_fig3_row3_y_axis_does_not_start_at_zero(tmp_figure_folder):
    """PR / eRank panels should autoscale around the data, not force y=0."""
    import plotting.figures as v4
    import matplotlib.pyplot as plt

    plotter = v4.Plotter(
        figure_folder=str(tmp_figure_folder), date_prefix="20260513",
    )
    fig = plotter.build_fig3()
    for ax in fig.axes[6:8]:
        ymin, _ = ax.get_ylim()
        assert ymin > 0.5, (
            f"PR/eRank panel y-axis should not start near zero, got ymin={ymin}"
        )
    plt.close(fig)


def test_make_fig3_dash_plot_legend_has_no_medium_handles(tmp_figure_folder):
    """Per design feedback: the dash legend shows only country colors."""
    import plotting.figures as v4
    import matplotlib.pyplot as plt

    plotter = v4.Plotter(
        figure_folder=str(tmp_figure_folder), date_prefix="20260513",
    )
    fig = plotter.build_fig3()
    for ax in fig.axes[6:8]:
        leg = ax.get_legend()
        if leg is None:
            continue
        labels = [t.get_text() for t in leg.get_texts()]
        assert "Survey" not in labels and "Social media" not in labels, (
            f"legend should not list Survey/Social media (axis tells us), got {labels}"
        )
        assert set(labels) == {"US", "CN", "EU"}
    plt.close(fig)


# -- orchestration -----------------------------------------------------------


def test_task_list_covers_main_plus_robustness_variants():
    """main + LOWESS supplement + media + noputback + england + word2vec + 4-region eu_nen."""
    import plotting.figures as v4

    names = [t["name"] for t in v4.TASKS]
    assert "main" in names
    assert "robust_1_media" in names
    assert "robust_2_noputback" in names
    assert "robust_3_england" in names
    assert "supp_lowess" in names
    assert "robust_5_word2vec" in names
    assert "robust_6_eu_nen" in names


def test_run_task_loess_variant_only_produces_fig2(tmp_figure_folder):
    """The LOWESS (nonlinear) alternative should regenerate Fig 2 only."""
    import plotting.figures as v4

    paths = v4.run_task(
        "supp_lowess",
        figure_folder=str(tmp_figure_folder),
        date_prefix="20260513",
        n_bootstrap=20,
    )
    assert list(paths) == [2]
    p = Path(paths[2])
    assert p.exists()
    assert p.name == "20260513_supp_lowess_results_fig2.pdf"


# -- v4 N-region (4-region eu_nen variant) ----------------------------------


def _four_region_plotter(folder):
    """Build the Plotter used by the eu_nen 4-region robustness task."""
    import plotting.figures as v4

    return v4.Plotter(
        regions=[
            v4.RegionSpec("US", "anes", "twitter"),
            v4.RegionSpec("CN", "wvs", "weibo"),
            v4.RegionSpec("EN", "en_evs", "entwitter"),
            v4.RegionSpec("EU_NEN", "eu_nen_evs", "eu_nentwitter"),
        ],
        figure_folder=str(folder), date_prefix="20260513",
        n_bootstrap=20,
    )


def test_plotter_accepts_regions_list_with_four_entries(tmp_figure_folder):
    """Plotter(regions=[...]) overrides the default 3-region setup."""
    plotter = _four_region_plotter(tmp_figure_folder)
    codes = [r.code for r in plotter.regions]
    assert codes == ["US", "CN", "EN", "EU_NEN"]


def test_fig3_with_four_regions_dash_plot_has_eight_dashes(tmp_figure_folder):
    """PR / eRank panels each carry 4 regions × 2 media = 8 dashes."""
    import matplotlib.pyplot as plt
    plotter = _four_region_plotter(tmp_figure_folder)
    fig = plotter.build_fig3()
    for ax in fig.axes[6:8]:
        horizontal_lines = [
            line for line in ax.get_lines()
            if len(line.get_xdata()) == 2 and line.get_ydata()[0] == line.get_ydata()[1]
        ]
        assert len(horizontal_lines) == 8, (
            f"4-region PR/eRank panel should have 8 dashes, got {len(horizontal_lines)}"
        )
    plt.close(fig)


def test_prepare_word_overlap_writes_json_with_expected_schema(tmp_path):
    """prepare_word_overlap.compute writes the JSON shape that plotting.figures reads."""
    from plotting import prepare_word_overlap as prep
    import json

    out_path = tmp_path / "word_overlap_rows.json"
    written = prep.compute(output_path=str(out_path), top_k=10)
    assert Path(written).exists()
    payload = json.load(open(written))
    assert payload["source"] == "twitter"
    assert payload["embedding_type"] == "gpt"
    assert payload["top_k"] == 10
    for key in ("low", "high"):
        entry = payload["pairs"][key]
        assert {"topic_a", "topic_b", "topic_similarity", "words_a", "words_b", "rows"} <= entry.keys()
        assert len(entry["rows"]) > 0
        first = entry["rows"][0]
        assert {"left", "right", "left_weight", "right_weight", "similarity", "kind"} <= first.keys()


def test_word_overlap_panel_uses_cached_artifact_when_present(tmp_path, monkeypatch):
    """When the cache JSON exists, the panel reads from it (no live cosine sweep)."""
    import plotting.figures as v4
    import matplotlib.pyplot as plt
    import json

    cache_path = tmp_path / "word_overlap_rows.json"
    # Hand-rolled cache with sentinel words so we can detect it on the panel.
    payload = {
        "source": "twitter", "embedding_type": "gpt", "top_k": v4._WORD_OVERLAP_TOP_K,
        "pairs": {
            "low": {
                "topic_a": "vac", "topic_b": "soc", "topic_similarity": 0.28,
                "words_a": [{"word": "SENTINEL_LEFT", "tf_idf": 0.05}],
                "words_b": [{"word": "SENTINEL_RIGHT", "tf_idf": 0.04}],
                "rows": [{
                    "left": "SENTINEL_LEFT", "right": "SENTINEL_RIGHT",
                    "left_weight": 0.05, "right_weight": 0.04,
                    "similarity": 0.50, "kind": "matched",
                }],
            },
            "high": {
                "topic_a": "ubi", "topic_b": "minwage", "topic_similarity": 0.71,
                "words_a": [{"word": "SENTINEL_H_LEFT", "tf_idf": 0.06}],
                "words_b": [{"word": "SENTINEL_H_RIGHT", "tf_idf": 0.05}],
                "rows": [{
                    "left": "SENTINEL_H_LEFT", "right": "SENTINEL_H_RIGHT",
                    "left_weight": 0.06, "right_weight": 0.05,
                    "similarity": 0.70, "kind": "matched",
                }],
            },
        },
    }
    cache_path.write_text(json.dumps(payload))
    monkeypatch.setattr(v4, "WORD_OVERLAP_CACHE_PATH", cache_path)
    # Sabotage the live-compute fallback so we can prove we read the cache.
    def _boom(*_args, **_kwargs):
        raise AssertionError("live compute path should not be invoked")
    monkeypatch.setattr(v4, "_select_us_twitter_extremes", _boom)

    fig, ax = plt.subplots()
    v4.Plotter()._word_overlap_panel(ax)
    texts = {t.get_text() for t in ax.texts}
    assert "SENTINEL_LEFT" in texts
    assert "SENTINEL_H_RIGHT" in texts
    plt.close(fig)


def test_word_overlap_panel_falls_back_to_live_compute_when_cache_missing(
    tmp_path, monkeypatch,
):
    """With the cache JSON absent, the panel still renders via the live path."""
    import plotting.figures as v4
    import matplotlib.pyplot as plt

    monkeypatch.setattr(v4, "WORD_OVERLAP_CACHE_PATH", tmp_path / "missing.json")
    fig, ax = plt.subplots()
    v4.Plotter()._word_overlap_panel(ax)
    assert len(ax.texts) > 0, "panel must still render words via the live path"
    plt.close(fig)


def test_fig2_non_pedagogical_omits_word_overlap_row(tmp_figure_folder):
    """Robustness Fig 2 (pedagogical=False) has 4 axes and no word-overlap panel."""
    import plotting.figures as v4
    import matplotlib.pyplot as plt

    plotter = v4.Plotter(
        figure_folder=str(tmp_figure_folder), date_prefix="20260513",
        n_bootstrap=50, pedagogical=False,
    )
    fig = plotter.build_fig2()
    assert len(fig.axes) == 4, (
        f"non-pedagogical Fig 2 should have 4 axes (2 violin + 2 LOESS), got {len(fig.axes)}"
    )
    # No 'Low similarity' / 'High similarity' headers from the word-overlap panel.
    all_text = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "Low similarity" not in all_text and "High similarity" not in all_text
    plt.close(fig)


def test_fig3_non_pedagogical_keeps_only_pr_erank(tmp_figure_folder):
    """Robustness Fig 3 (pedagogical=False) has 2 axes and no 3D toy-cloud panels."""
    import plotting.figures as v4
    import matplotlib.pyplot as plt

    plotter = v4.Plotter(
        figure_folder=str(tmp_figure_folder), date_prefix="20260513",
        pedagogical=False,
    )
    fig = plotter.build_fig3()
    assert len(fig.axes) == 2, (
        f"non-pedagogical Fig 3 should have 2 axes (PR + eRank), got {len(fig.axes)}"
    )
    assert all(ax.name != "3d" for ax in fig.axes), "no 3D toy-cloud panels expected"
    # Both panels should still carry the real dash plot (8 or 6 dashes).
    for ax in fig.axes:
        horizontal = [
            line for line in ax.get_lines()
            if len(line.get_xdata()) == 2 and line.get_ydata()[0] == line.get_ydata()[1]
        ]
        assert len(horizontal) >= 6
    plt.close(fig)


def test_run_task_sets_pedagogical_only_for_main(tmp_figure_folder, monkeypatch):
    """run_task builds a pedagogical Plotter for 'main' and a plain one otherwise."""
    import plotting.figures as v4

    captured = {}
    real_init = v4.Plotter.__init__

    def spy_init(self, *args, **kwargs):
        captured["pedagogical"] = kwargs.get("pedagogical")
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(v4.Plotter, "__init__", spy_init)

    v4.run_task("main", figure_folder=str(tmp_figure_folder),
                date_prefix="20260513", n_bootstrap=10)
    assert captured["pedagogical"] is True

    v4.run_task("robust_3_england", figure_folder=str(tmp_figure_folder),
                date_prefix="20260513", n_bootstrap=10)
    assert captured["pedagogical"] is False



def test_annual_task_uses_2021_spectra_for_social_media():
    """robust_10_annual: Weibo spectral value is its 2021 matrix, surveys unchanged."""
    import json

    import plotting.figures as v4
    from common.paths import dimension_results_dir

    plotter = v4.Plotter(**v4.task_plotter_kwargs("robust_10_annual"))
    assert plotter.spectral_year("weibo") == "2021"
    assert plotter.spectral_year("anes") is None
    means = plotter._spectral_means("PR")
    with open(dimension_results_dir("cov") / "weibo-2021.csr-none.json") as f:
        assert np.isclose(means["weibo"], json.load(f)["PR"])
    main = v4.Plotter(**v4.task_plotter_kwargs("main"))._spectral_means("PR")
    assert np.isclose(means["anes"], main["anes"])
    assert np.isclose(means["twitter"], main["twitter"])  # Twitter/X is 2021 already


def test_weibo_user_tables(tmp_path):
    from cleaning.descriptive_stats import weibo_user_tables

    d = tmp_path / "7"
    d.mkdir()
    pd.DataFrame({"2020": [0.5, np.nan, -1.0], "2020_count": [2, np.nan, 1],
                  "2021": [np.nan, 1.0, 0.0], "2021_count": [np.nan, 3, 4],
                  "average": [0.5, 1.0, -0.2]}).to_parquet(d / "avg_opinion.parquet")
    out = weibo_user_tables(str(tmp_path))
    tot = out["weibo_users_by_topic"].set_index("topic").loc["7"]
    assert tot["users"] == 3 and tot["relevant_posts"] == 10
    yr = out["weibo_users_by_topic_year"].set_index("year")
    assert yr.loc[2020, "users"] == 2 and yr.loc[2021, "relevant_posts"] == 7
