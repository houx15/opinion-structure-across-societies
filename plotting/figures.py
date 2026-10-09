"""Results-section figures (Fig 2, Fig 3) and their robustness variants.

This module only *plots*; ``paper_stats.py`` writes the numbers behind each
figure. Inputs (see data/README.md):

    data/correlation/network_analysis_<src>.csv        pairwise topic correlations
    data/embedding/<survey>_topic_distance.csv          survey topic similarities
    data/embedding/dynamic_embedding_<social>_<gpt|dictionary>.csv
    data/dimension/results/cov/<stem>-none-summary.json PR / eRank (+ -bootstrap.json; corr/ for the robustness check)
    data/opinions/individual_opinion_<survey>.parquet
    data/opinions/user_opinion_<social>_lgbt_env.parquet
    data/tf_idf/tf_idf_twitter/, data/embedding/word_overlap_rows.json  (Fig 2 word panel)

Usage (from the repository root):
    python -m plotting.figures main                 # main + LOWESS supplement
    python -m plotting.figures task robust_3_england
    python -m plotting.figures all                  # every task in TASKS
Figures go to outputs/figures/<date>_<task>_results_<fig>.pdf, statistics to
outputs/stats/<date>_<task>_results_stats.txt.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
from statsmodels.nonparametric.smoothers_lowess import lowess

from common.topics import (
    CN_SURVEY_TOPICS,
    EU_SURVEY_TOPICS,
    EUTWITTER_TOPICS,
    TWITTER_TOPICS,
    US_SURVEY_TOPICS,
    WEIBO_TOPICS,
)


# -- Topic sets (mirrors plot_prod_v3) ---------------------------------------

# The nine analysed topics per source are defined in common/topics.py:
# US/CN/EU_SURVEY_TOPICS, TWITTER_TOPICS, EUTWITTER_TOPICS, WEIBO_TOPICS.


RESTRICTED_TOPICS: Dict[str, Sequence[Union[str, int]]] = {
    # surveys
    "anes": US_SURVEY_TOPICS,
    "anes_media": US_SURVEY_TOPICS,
    "wvs": CN_SURVEY_TOPICS,
    "wvs_media": CN_SURVEY_TOPICS,
    "evs": EU_SURVEY_TOPICS,
    "evs_media": EU_SURVEY_TOPICS,
    "evs_resample2": EU_SURVEY_TOPICS,
    "evs_media_resample2": EU_SURVEY_TOPICS,
    "en_evs": EU_SURVEY_TOPICS,
    "en_evs_media": EU_SURVEY_TOPICS,
    "eu_nen_evs": EU_SURVEY_TOPICS,
    # social media
    "twitter": TWITTER_TOPICS,
    "weibo": WEIBO_TOPICS,
    "eutwitter": EUTWITTER_TOPICS,
    "entwitter": EUTWITTER_TOPICS,
    "eu_nentwitter": EUTWITTER_TOPICS,
}


from common.paths import (
    CORRELATION_DIR,
    dimension_results_dir,
    EMBEDDING_DIR,
    FIGURE_DIR,
    OPINION_DIR,
    STATS_DIR,
    TF_IDF_DIR,
)


SOCIAL_SOURCES = {"twitter", "weibo", "eutwitter", "entwitter", "eu_nentwitter"}


# For each source, the (LGBT, environment-like) topic pair anchoring the case
# study. Surveys use the source's native column names; social-media sources
# carry the normalized LGBT / Environment columns produced by
# export_user_opinion_data.py.
CASE_STUDY_TOPICS: Dict[str, Tuple[str, str]] = {
    "anes": ("LGBT", "Climate"),
    "anes_media": ("LGBT", "Climate"),
    "wvs": ("LGBT", "Environment"),
    "wvs_media": ("LGBT", "Environment"),
    "evs": ("LGBT", "Environment"),
    "evs_media": ("LGBT", "Environment"),
    "evs_resample2": ("LGBT", "Environment"),
    "evs_media_resample2": ("LGBT", "Environment"),
    "en_evs": ("LGBT", "Environment"),
    "en_evs_media": ("LGBT", "Environment"),
    "eu_nen_evs": ("LGBT", "Environment"),
    "twitter": ("LGBT", "Environment"),
    "weibo": ("LGBT", "Environment"),
    "eutwitter": ("LGBT", "Environment"),
    "entwitter": ("LGBT", "Environment"),
    "eu_nentwitter": ("LGBT", "Environment"),
}

# Topic family ordering per country (cultural-moral -> economic-welfare ->
# governance-regulatory). Matches Table 1 in docs/results_figures_spec.md and
# drives the row ordering in Fig 3 Row 3's PC1 loadings heatmap.
_TOPIC_FAMILY_ORDER: Dict[str, List[str]] = {
    "anes": [
        "Abortion", "LGBT", "DeathPenalty",                       # cultural-moral
        "MinimumWage", "UBI",                                     # economic-welfare
        "Gun", "Climate", "VACC", "Media",                        # governance-regulatory
    ],
    "wvs": [
        "LGBT", "GenderEqual", "Marriage", "Childbearing",        # cultural-moral
        "Econ", "Work",                                           # economic-welfare
        "Corrup", "Environment", "Foreign",                       # governance-regulatory
    ],
    "evs_resample2": [
        "LGBT", "Abortion", "DeathPenalty", "Prostitution",       # cultural-moral
        "Egalitarian", "HealthCare", "UnemplyAid",                # economic-welfare
        "Environment", "SocialMedia",                             # governance-regulatory
    ],
}
# All EU-survey variants share the topic ordering of evs_resample2.
_TOPIC_FAMILY_ORDER["en_evs"] = _TOPIC_FAMILY_ORDER["evs_resample2"]
_TOPIC_FAMILY_ORDER["evs"] = _TOPIC_FAMILY_ORDER["evs_resample2"]
_TOPIC_FAMILY_ORDER["anes_media"] = _TOPIC_FAMILY_ORDER["anes"]
_TOPIC_FAMILY_ORDER["wvs_media"] = _TOPIC_FAMILY_ORDER["wvs"]
_TOPIC_FAMILY_ORDER["evs_media"] = _TOPIC_FAMILY_ORDER["evs_resample2"]
_TOPIC_FAMILY_ORDER["evs_media_resample2"] = _TOPIC_FAMILY_ORDER["evs_resample2"]
_TOPIC_FAMILY_ORDER["en_evs_media"] = _TOPIC_FAMILY_ORDER["en_evs"]
_TOPIC_FAMILY_ORDER["eu_nen_evs"] = _TOPIC_FAMILY_ORDER["evs_resample2"]


def topic_family_order(src: str) -> List[str]:
    """Return the 9 topic columns for ``src`` ordered by issue family (Table 1)."""
    return list(_TOPIC_FAMILY_ORDER[src])


# Offline ↔ online topic correspondence per country, as supplied by the
# researcher. Each entry maps an offline (survey) topic id to its online
# (social-media) counterpart. The mapping is positional: each row of the
# country's 9 topics is paired with the same row of the social-media
# platform's 9 topics. Not every pairing is a clean conceptual match — EU is
# the loosest, where some EVS topics (Prostitution, HealthCare, UnemplyAid,
# Egalitarian) don't have an exact Twitter equivalent and are mapped to the
# closest Twitter slot. Keep this in mind before treating offline and online
# rows of the same name as identical concepts.
OFFLINE_ONLINE_TOPIC_MAP: Dict[str, Dict[Union[str, int], Union[str, int]]] = {
    "US": {
        "Abortion": "abo",
        "Gun": "gun",
        "Climate": "clc",
        "LGBT": "sxo",
        "VACC": "vac",
        "DeathPenalty": "dpp",
        "Media": "soc",
        "MinimumWage": "minwage",
        "UBI": "ubi",
    },
    "CN": {
        "Corrup": 7,
        "GenderEqual": 9,
        "Marriage": 11,
        "Childbearing": 12,
        "LGBT": 10,
        "Environment": 13,
        "Econ": 15,
        "Work": 14,
        "Foreign": 0,
    },
    "EU": {
        # Conceptual bridges (HealthCare ↔ vac, UnemplyAid ↔ minwage,
        # Egalitarian ↔ ubi) are loose mappings, not exact equivalences.
        "Abortion": "abo",
        "Prostitution": "swe",
        "Environment": "clc",
        "LGBT": "sxo",
        "HealthCare": "vac",
        "DeathPenalty": "dpp",
        "SocialMedia": "soc",
        "UnemplyAid": "minwage",
        "Egalitarian": "ubi",
    },
}


# Short, reader-friendly names for every topic id we plot. Used by the Fig 2
# LOESS callouts so labels read "Abortion x Death Pen" instead of raw codes.
_TOPIC_LABEL: Dict[Union[str, int], str] = {
    # US (ANES) — column names are already readable; only abbreviate the long ones.
    "Abortion": "Abortion", "Gun": "Gun", "Climate": "Climate", "LGBT": "LGBT",
    "VACC": "Vaccine", "DeathPenalty": "Death Pen", "Media": "Media",
    "MinimumWage": "Min Wage", "UBI": "UBI",
    # CN (WVS)
    "Corrup": "Corrupt", "GenderEqual": "Gender Eq", "Marriage": "Marriage",
    "Childbearing": "Child", "Environment": "Env", "Econ": "Economy",
    "Work": "Work", "Foreign": "Foreign",
    # EU (EVS) — overlap with the survey block above.
    "SocialMedia": "Social Med", "Egalitarian": "Egal",
    "Prostitution": "Prostitution", "HealthCare": "Health", "UnemplyAid": "Unempl",
    # Twitter codes (US set + eu-only "swe")
    "abo": "Abortion", "gun": "Gun", "clc": "Climate", "sxo": "LGBT",
    "vac": "Vaccine", "dpp": "Death Pen", "soc": "Social Med",
    "minwage": "Min Wage", "ubi": "UBI",
    "swe": "Prostitution",
    # Weibo numeric topic ids (see data/new_topic_opinion.csv).
    0: "US", 7: "Corrupt", 9: "Gender Eq", 10: "LGBT", 11: "Marriage",
    12: "Child", 13: "Env", 14: "Work", 15: "Economy",
}


def _pretty_topic(tid) -> str:
    """Map an internal topic id (str or int) to a short readable label."""
    if tid in _TOPIC_LABEL:
        return _TOPIC_LABEL[tid]
    try:
        # weibo tids are int-like; allow int(tid) fallback for strs that hold "11".
        return _TOPIC_LABEL.get(int(tid), str(tid))
    except (TypeError, ValueError):
        return str(tid)


# Topic IDs (in the network_analysis CSVs) corresponding to the case-study pair.
# Social sources store native codes; surveys store the column name directly.
_CASE_STUDY_TIDS: Dict[str, frozenset] = {
    "anes": frozenset({"LGBT", "Climate"}),
    "anes_media": frozenset({"LGBT", "Climate"}),
    "wvs": frozenset({"LGBT", "Environment"}),
    "wvs_media": frozenset({"LGBT", "Environment"}),
    "evs": frozenset({"LGBT", "Environment"}),
    "evs_media": frozenset({"LGBT", "Environment"}),
    "evs_resample2": frozenset({"LGBT", "Environment"}),
    "evs_media_resample2": frozenset({"LGBT", "Environment"}),
    "en_evs": frozenset({"LGBT", "Environment"}),
    "en_evs_media": frozenset({"LGBT", "Environment"}),
    "eu_nen_evs": frozenset({"LGBT", "Environment"}),
    "twitter": frozenset({"sxo", "clc"}),
    "eutwitter": frozenset({"sxo", "clc"}),
    "entwitter": frozenset({"sxo", "clc"}),
    "eu_nentwitter": frozenset({"sxo", "clc"}),
    "weibo": frozenset({10, 13}),
}


# Maps a plotting-side src name to the data/dimension/results summary filename stem.
# Surveys use their own name; twitter platforms split by region.
_SPECTRAL_STEM_OVERRIDE: Dict[str, str] = {
    "twitter": "twitter-us",
    "eutwitter": "twitter-eu",
    "entwitter": "twitter-en",
    "eu_nentwitter": "twitter-eu_nen",
}


# -- Data loaders ------------------------------------------------------------


def load_pairwise_correlation(src: str, year: str = "average") -> pd.DataFrame:
    """Load topic-pair correlations for one source.

    Returns a DataFrame with the columns the figures need:
        tid1, tid2, topic_combination, year,
        intersection, pearson (signed), correlation (|r|)

    Restricts to the source's nine topics of interest, drops NaN correlations,
    and defaults to the pooled "average" year so consumers get one row per pair.
    Pass year=None to keep every year.
    """
    csv_path = CORRELATION_DIR / f"network_analysis_{src}.csv"
    raw = pd.read_csv(csv_path)

    # Some social-media CSVs only have Topic1/Topic2; copy into tid1/tid2 for
    # uniform downstream handling. These files store the topic *code* (e.g. "sxo").
    if "tid1" not in raw.columns:
        raw["tid1"] = raw["Topic1"]
    if "tid2" not in raw.columns:
        raw["tid2"] = raw["Topic2"]

    # Restrict to the topics of interest. Weibo stores tids as ints — the
    # restricted_topics list is also int there, so the .isin works directly.
    topics = list(RESTRICTED_TOPICS[src])
    mask = raw["tid1"].isin(topics) & raw["tid2"].isin(topics)
    df = raw[mask].copy()
    df = df.dropna(subset=["Correlation (Pearson)"])

    df["pearson"] = df["Correlation (Pearson)"].astype(float)
    df["correlation"] = df["pearson"].abs()
    df["intersection"] = df["Intersection"].astype(int)
    df["topic_combination"] = df.apply(
        lambda r: "_".join(sorted([str(r["tid1"]), str(r["tid2"])])), axis=1
    )

    if year is not None:
        df = df[df["year"].astype(str) == year]

    keep = [
        "tid1", "tid2", "topic_combination", "year",
        "intersection", "pearson", "correlation",
    ]
    return df[keep].reset_index(drop=True)


# Survey topic-distance column per embedding_type (semantic_similarity
# .process_survey_embedding): capped top-5 keywords (main), the uncapped
# selected keywords, hand-curated keywords, and the earlier full-question
# embeddings.
SURVEY_SIMILARITY_COLUMN: Dict[str, str] = {
    "gpt": "similarity_top5",
    "dictionary": "dictionary_similarity_top5",
    "gpt_selected": "similarity",
    "gpt_curated": "similarity_curated",
    "gpt_question": "similarity_question",
}


def load_semantic_similarity(src: str, embedding_type: str = "gpt") -> pd.DataFrame:
    """Load a single cosine similarity per topic pair for one source.

    Returns DataFrame with columns: tid1, tid2, topic_combination, similarity.

    For surveys, the static `embedding/<src>_topic_distance.csv` stores one
    similarity per pair and per variant (see SURVEY_SIMILARITY_COLUMN). For
    social media, we use the pooled "merged" year from
    `embedding/dynamic_embedding_<src>_<gpt|dictionary>.csv` so the value is
    comparable across pairs; the survey-only variants use the gpt file.
    """
    topics = list(RESTRICTED_TOPICS[src])

    if src in SOCIAL_SOURCES:
        social_type = "dictionary" if embedding_type == "dictionary" else "gpt"
        path = EMBEDDING_DIR / f"dynamic_embedding_{src}_{social_type}.csv"
        raw = pd.read_csv(path)
        raw = raw[raw["year"].astype(str).isin(["merged", "average"])].copy()
        raw["similarity"] = raw["similarity"].astype(float)
    else:
        path = EMBEDDING_DIR / f"{src}_topic_distance.csv"
        raw = pd.read_csv(path)
        sim_col = SURVEY_SIMILARITY_COLUMN[embedding_type]
        raw = raw[["topic_id1", "topic_id2", sim_col]].rename(columns={sim_col: "similarity"})

    raw["tid1"] = raw["topic_id1"]
    raw["tid2"] = raw["topic_id2"]
    # Weibo IDs may be ints; coerce when topics is int-typed so .isin matches.
    if topics and isinstance(topics[0], int):
        raw["tid1"] = pd.to_numeric(raw["tid1"], errors="coerce").astype("Int64")
        raw["tid2"] = pd.to_numeric(raw["tid2"], errors="coerce").astype("Int64")
    mask = raw["tid1"].isin(topics) & raw["tid2"].isin(topics)
    df = raw[mask].copy()
    df["topic_combination"] = df.apply(
        lambda r: "_".join(sorted([str(r["tid1"]), str(r["tid2"])])), axis=1
    )
    return df[["tid1", "tid2", "topic_combination", "similarity"]].reset_index(drop=True)


def spectral_summary_path(src: str, matrix: str = "cov", year: Optional[str] = None) -> Path:
    """<stem>-none-summary.json (all yearly matrices) or, with ``year``, that
    year's <stem>-<year>.csr-none.json."""
    stem = _SPECTRAL_STEM_OVERRIDE.get(src, src)
    if year is None:
        return dimension_results_dir(matrix) / f"{stem}-none-summary.json"
    return dimension_results_dir(matrix) / f"{stem}-{year}.csr-none.json"


def load_spectral_summary(src: str, matrix: str = "cov", year: Optional[str] = None) -> Dict[str, List[float]]:
    """Load PR / eRank / srank arrays for one source from data/dimension/results.

    Returns a dict ``{"PR": [...], "eRank": [...], "srank": [...]}`` with
    floats (so callers can compute means or plot directly). ``matrix`` picks
    data/dimension/results/cov (main) or results/corr (robustness check);
    ``year`` keeps only that year's matrix (annual robustness check).
    """
    with open(spectral_summary_path(src, matrix, year)) as f:
        raw = json.load(f)
    keys = ("PR", "eRank", "srank")
    return {key: [float(v) for v in np.atleast_1d(raw[key])] for key in keys if key in raw}


def load_spectral_bootstrap(src: str, matrix: str = "cov", year: Optional[str] = None) -> Optional[Dict[str, float]]:
    """Load spectral_bootstrap.py's interval file for one source, or None.

    Keys used here: ``PR``, ``PR_lo``, ``PR_hi`` and the same for ``eRank``.
    With ``year``: <stem>-none-<year>-bootstrap.json, or the main file when
    that already covers exactly this one year (Twitter/X: 2021 only).
    """
    stem = _SPECTRAL_STEM_OVERRIDE.get(src, src)
    folder = dimension_results_dir(matrix)
    if year is not None:
        path = folder / f"{stem}-none-{year}-bootstrap.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    path = folder / f"{stem}-none-bootstrap.json"
    if not path.exists():
        return None
    with open(path) as f:
        boot = json.load(f)
    if year is not None and boot.get("files") != [f"{year}.csr.npz"]:
        return None
    return boot


def load_individual_opinion(src: str) -> pd.DataFrame:
    """Load survey respondent-level opinions from the prepared parquet.

    Produced by ``prepare_survey_individual_data.py``. Columns:
        respondent_id, year, plus the source's 9 topic columns.
    """
    path = OPINION_DIR / f"individual_opinion_{src}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run prepare_survey_individual_data.py first."
        )
    return pd.read_parquet(path)


def load_user_opinion_pair(src: str) -> pd.DataFrame:
    """Load social-media user-level opinions for the LGBT x Environment pair.

    Produced by ``export_user_opinion_data.py`` on the upstream host.
    Columns: user_id, LGBT, Environment.
    """
    path = OPINION_DIR / f"user_opinion_{src}_lgbt_env.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run export_user_opinion_data.py {src} on the upstream host."
        )
    return pd.read_parquet(path)


# -- Statistical helpers -----------------------------------------------------


def compute_abs_pearson(x, y) -> Tuple[float, int]:
    """Return (|r|, n_used) after dropping pairs with NaN in either side."""
    s = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(s) < 2:
        return float("nan"), len(s)
    r, _ = pearsonr(s["x"].to_numpy(), s["y"].to_numpy())
    return abs(float(r)), int(len(s))


def bootstrap_mean_ci(
    values,
    n_boot: int = 1000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of a 1-D sample.

    Returns (mean, lo, hi). For an empty input, returns (nan, nan, nan).
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng or np.random.default_rng()
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = sample.mean()
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(arr.mean()), float(lo), float(hi)


def pca_loadings(X: np.ndarray) -> Dict[str, np.ndarray]:
    """Listwise-deletion PCA on an (n_samples, n_features) matrix.

    Returns a dict with:
        eigenvalues:               descending
        eigenvectors:              column k is the k-th principal axis (descending)
        explained_variance_ratio:  descending, sums to 1
        n_samples_used:            rows kept after listwise deletion
        mean:                      column means of the kept rows (for plotting arrows)
    """
    X = np.asarray(X, dtype=float)
    mask = np.all(np.isfinite(X), axis=1)
    Xc = X[mask]
    n_used = Xc.shape[0]
    mean = Xc.mean(axis=0)
    centered = Xc - mean
    # rowvar=False -> each column is a variable
    cov = np.cov(centered, rowvar=False, ddof=1)
    cov = 0.5 * (cov + cov.T)  # numerical symmetry
    lam, V = np.linalg.eigh(cov)  # ascending
    order = np.argsort(lam)[::-1]
    lam = lam[order]
    V = V[:, order]
    total = lam.sum()
    evr = lam / total if total > 0 else np.zeros_like(lam)
    return {
        "eigenvalues": lam,
        "eigenvectors": V,
        "explained_variance_ratio": evr,
        "n_samples_used": n_used,
        "mean": mean,
    }


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """PR = (sum lambda)^2 / sum(lambda^2), restricted to positive eigenvalues."""
    lam = np.asarray(eigenvalues, dtype=float)
    lam = lam[lam > 0]
    if lam.size == 0:
        return float("nan")
    s1 = lam.sum()
    s2 = (lam ** 2).sum()
    return float((s1 ** 2) / s2) if s2 > 0 else float("nan")


def exponential_rank(eigenvalues: np.ndarray) -> float:
    """eRank = exp(H(p)), where p_i = lambda_i / sum(lambda) (positive lambdas only)."""
    lam = np.asarray(eigenvalues, dtype=float)
    lam = lam[lam > 0]
    if lam.size == 0:
        return float("nan")
    p = lam / lam.sum()
    H = -float(np.sum(p * np.log(p + 1e-12)))
    return float(np.exp(H))


def get_case_study_pair(src: str) -> pd.DataFrame:
    """Return individual- or user-level (x, y) opinions for the LGBT x Env case study.

    The returned DataFrame has columns ['x', 'y'] (so plot code doesn't have to
    know the source-specific topic name) and ``df.attrs`` carries the original
    topic names plus the source label.
    """
    x_topic, y_topic = CASE_STUDY_TOPICS[src]
    if src in SOCIAL_SOURCES:
        df = load_user_opinion_pair(src)
        out = df[["LGBT", "Environment"]].rename(columns={"LGBT": "x", "Environment": "y"})
    else:
        df = load_individual_opinion(src)
        out = df[[x_topic, y_topic]].rename(columns={x_topic: "x", y_topic: "y"})
    out = out.dropna(how="all").reset_index(drop=True)
    out.attrs["x_topic"] = x_topic
    out.attrs["y_topic"] = y_topic
    out.attrs["src"] = src
    return out


def lookup_case_study_correlation(src: str) -> float:
    """|r| for the LGBT x Env pair from the pre-aggregated network_analysis CSV.

    This is the dash-plot fallback when user-level social-media data isn't
    exported locally yet. Always available because the CSVs ship with the repo.
    """
    target = _CASE_STUDY_TIDS[src]
    df = load_pairwise_correlation(src)
    matched = df[df.apply(lambda r: frozenset({r["tid1"], r["tid2"]}) == target, axis=1)]
    if matched.empty:
        return float("nan")
    return float(matched["correlation"].iloc[0])


def signed_r_from_pairwise(corr_df: pd.DataFrame, t1, t2) -> float:
    """Look up the signed Pearson r for a single (t1, t2) pair in a pairwise CSV."""
    target = frozenset({str(t1), str(t2)})
    mask = corr_df.apply(
        lambda r: frozenset({str(r["tid1"]), str(r["tid2"])}) == target, axis=1,
    )
    matched = corr_df[mask]
    if matched.empty:
        return float("nan")
    return float(matched["pearson"].iloc[0])


def synth_ellipse_from_signed_r(
    signed_r: float,
    *,
    n: int = 800,
    seed: int = 0,
    gamma: float = 1.0,
) -> Dict[str, object]:
    """Synthetic standardized cloud + correlation ellipse for a given signed r.

    Sampled from the 2D Gaussian with correlation matrix ``[[1, r], [r, 1]]``
    so the cloud is unit-variance on each axis. The ellipse is then derived
    from the *theoretical* correlation matrix (not the noisy sample) so its
    orientation is exactly +45° for ``r >= 0`` and -45° for ``r < 0`` even
    when ``|r|`` is near zero. ``gamma`` optionally stretches both the
    scattered points and the ellipse along the same principal axes
    (``T = V·Λ^((γ-1)/2)·V^T``); the default ``gamma=1.0`` draws the true
    correlation ellipse with no exaggeration.
    """
    r = float(max(min(signed_r, 0.999), -0.999))
    rng = np.random.default_rng(seed)
    cov0 = np.array([[1.0, r], [r, 1.0]])
    pts = rng.multivariate_normal([0.0, 0.0], cov0, size=n)
    pts = (pts - pts.mean(axis=0)) / (pts.std(axis=0) + 1e-12)

    angle_deg = 45.0 if r >= 0 else -45.0
    rad = np.radians(angle_deg)
    V = np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad),  np.cos(rad)],
    ])
    w = np.array([1.0 + abs(r), 1.0 - abs(r)])
    if gamma != 1.0:
        transform = V @ np.diag(w ** ((gamma - 1.0) / 2.0)) @ V.T
        pts = pts @ transform
        w = w ** gamma
    return {
        "pts": pts,
        "angle": angle_deg,
        "smaj": float(np.sqrt(w[0])),
        "smin": float(np.sqrt(w[1])),
    }


def bootstrap_lowess_ci(
    x_data,
    y_data,
    frac: float = 0.4,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Bootstrap a LOWESS curve to obtain a pointwise confidence band.

    Ported from plot_prod_v3.bootstrap_lowess_confidence_interval but with a
    seeded RNG so tests are reproducible. Returns (x_grid, mean, lo, hi) or
    (None, None, None, None) on degenerate input.
    """
    rng = rng or np.random.default_rng()
    x_arr = np.asarray(x_data)
    y_arr = np.asarray(y_data)
    n = len(x_arr)
    if n < 4:
        return None, None, None, None

    curves = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            curve = lowess(y_arr[idx], x_arr[idx], frac=frac)
        except Exception:
            continue
        if len(curve) > 1:
            curves.append(curve)

    if not curves:
        return None, None, None, None

    all_x = np.concatenate([c[:, 0] for c in curves])
    x_grid = np.linspace(all_x.min(), all_x.max(), 100)
    interp = np.array(
        [np.interp(x_grid, c[:, 0], c[:, 1]) for c in curves]
    )
    alpha = 1.0 - confidence_level
    lo = np.percentile(interp, (alpha / 2) * 100, axis=0)
    hi = np.percentile(interp, (1 - alpha / 2) * 100, axis=0)
    mean = interp.mean(axis=0)
    return x_grid, mean, lo, hi


# -- Plotter -----------------------------------------------------------------


# Colors and styles mirror plot_prod_v3.
_WEIBO_COLOR = "#ff7333"        # red — China
_TWITTER_COLOR = "#207de6"      # deeper blue — United States (text readability)
_EUTWITTER_COLOR = "#54457F"    # purple — Europe (full / continental)
_ENTWITTER_COLOR = "#2A9D8F"    # teal — English-speaking Europe (when split from EU_NEN)
_SURVEY_LINESTYLE = "solid"
_MEDIA_LINESTYLE = "dashed"


# Defaults for the country-color lookup. The 4-region task adds EN as teal
# alongside the existing US / CN / EU palette.
_DEFAULT_COUNTRY_COLOR: Dict[str, str] = {
    "US": _TWITTER_COLOR,
    "CN": _WEIBO_COLOR,
    "EU": _EUTWITTER_COLOR,
    "EN": _ENTWITTER_COLOR,
    "EU_NEN": _EUTWITTER_COLOR,
}

# Visual amplification for the Fig 1 case-study ellipses. Each standardized
# cloud is stretched along its own principal axes by this power so the
# alignment differences (thin vs round) read clearly — the natural correlation
# ellipse (γ=1) compresses them. The displayed |r| stays the true value; only
# the shape is exaggerated. γ=1 disables it.
_ELLIPSE_EXAGGERATION = 2.5


@dataclass(frozen=True)
class RegionSpec:
    """One country/region slot in the Plotter.

    Each spec carries the offline (survey) and online (social media) sources
    plus a stable country code used for color lookup and axis ticks. The 3
    standard regions (US / CN / EU) are built automatically by Plotter when no
    explicit ``regions`` list is supplied; the 4-region eu_nen robustness task
    passes its own list.
    """

    code: str
    survey: str
    social: str
    color: Optional[str] = None
    survey_display: Optional[str] = None
    social_display: Optional[str] = None

    def resolved_color(self) -> str:
        return self.color or _DEFAULT_COUNTRY_COLOR.get(self.code, "#333333")


# -- Word-overlap (Fig 2 row 2) ----------------------------------------------

# Top-K words per topic. Mirrors semantic_similarity.process_embedding
# which uses head(10) by TF-IDF when building topic centroids; using the same
# K here keeps the panel honest about which words actually drive each topic's
# embedding centroid in that computation.
_WORD_OVERLAP_TOP_K = 10
# Font family for the word list (the rest of the figure uses Times New Roman).
_WORD_OVERLAP_FONT_FAMILY = "Arial"
# Matched word pairs above this cosine similarity are highlighted; everything
# else renders in plain dark gray.
_WORD_OVERLAP_HIGHLIGHT_THRESHOLD = 0.40
_WORD_OVERLAP_HIGHLIGHT_COLOR = "#d62728"   # red — highlighted (sim > threshold)
_WORD_OVERLAP_PLAIN_COLOR = "#333333"       # dark gray — everything else
# Font size range. Word size is interpolated linearly across each panel's
# observed TF-IDF range, so a high-TF-IDF word renders ~2x the area of the
# low end.
_WORD_OVERLAP_FONT_SIZE_MIN = 10.0
_WORD_OVERLAP_FONT_SIZE_MAX = 18.0


def _load_twitter_top_words(topic_id: str, k: int = _WORD_OVERLAP_TOP_K) -> List[str]:
    """Top-K words for a US Twitter topic ranked by TF-IDF (pooled 'merged' year).

    Returns only the words; for word + weight pairs use
    :func:`_load_twitter_top_words_with_weights`. Ranking matches
    ``semantic_similarity.process_embedding`` (top-10 by TF-IDF), so
    the panel illustrates the same set of words that drive the published
    topic-similarity numbers.
    """
    return [w for w, _ in _load_twitter_top_words_with_weights(topic_id, k=k)]


def _load_twitter_top_words_with_weights(
    topic_id: str, k: int = _WORD_OVERLAP_TOP_K,
) -> List[Tuple[str, float]]:
    """Top-K (word, TF-IDF weight) tuples for a US Twitter topic, sorted desc."""
    path = TF_IDF_DIR / "tf_idf_twitter" / f"tf-idf-{topic_id}-merged.parquet"
    df = pd.read_parquet(path)
    top = df.sort_values("TF-IDF", ascending=False).head(k)
    return [(str(w), float(s)) for w, s in zip(top["word"], top["TF-IDF"])]


def _select_us_twitter_extremes() -> Tuple[Tuple[str, str, float], Tuple[str, str, float]]:
    """Pick the lowest- and highest-similarity topic pair from US Twitter.

    Returns ((low_t1, low_t2, low_sim), (high_t1, high_t2, high_sim)). The
    selection always uses ``embedding/dynamic_embedding_twitter_gpt.csv``
    restricted to the pooled 'merged'/'average' row per pair, so callers get
    one value per pair regardless of the per-year breakdown.
    """
    path = EMBEDDING_DIR / "dynamic_embedding_twitter_gpt.csv"
    df = pd.read_csv(path)
    merged = df[df["year"].astype(str).isin(["merged", "average"])].copy()
    if merged.empty:
        raise ValueError(f"no merged/average rows in {path}")
    merged = merged.dropna(subset=["similarity"]).sort_values("similarity")
    low = merged.iloc[0]
    high = merged.iloc[-1]
    return (
        (str(low["topic_id1"]), str(low["topic_id2"]), float(low["similarity"])),
        (str(high["topic_id1"]), str(high["topic_id2"]), float(high["similarity"])),
    )


_WORD_EMBEDDING_CACHE_PATH = EMBEDDING_DIR / "cached_embedding.pkl"
_WORD_EMBEDDING_CACHE: Optional[Mapping[str, np.ndarray]] = None

# Optional precomputed artifact written by prepare_word_overlap.py. When this
# file is present, the row-2 panel reads it instead of recomputing the greedy
# match + cosine matrix on every figure render.
WORD_OVERLAP_CACHE_PATH = EMBEDDING_DIR / "word_overlap_rows.json"


def _load_cached_word_overlap(
    path: Optional[Path] = None,
) -> Optional[Dict]:
    """Load the precomputed row-2 payload written by ``prepare_word_overlap.py``.

    Returns ``None`` when the file is missing — callers should then fall back
    to a live compute. The schema is the one written by
    :func:`prepare_word_overlap._build_payload`::

        {
            "source": str, "embedding_type": str, "top_k": int,
            "pairs": {
                "low" | "high": {
                    "topic_a": str, "topic_b": str, "topic_similarity": float,
                    "words_a": [{"word": str, "tf_idf": float}, ...],
                    "words_b": [...],
                    "rows": [WordOverlapRow as dict, ...],
                },
            },
        }
    """
    path = path or WORD_OVERLAP_CACHE_PATH
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_word_embeddings() -> Mapping[str, np.ndarray]:
    """Load the shared GPT word-embedding cache (memoized at module level).

    The cache is the same pickle that ``analysis/semantic/semantic_similarity.py``
    populates: a ``{word: list[float]}`` mapping with 1536-d vectors. We
    convert each value to a numpy array on first read so the cosine-similarity
    inner loop in :func:`_compute_word_overlap_rows` stays in numpy.
    """
    global _WORD_EMBEDDING_CACHE
    if _WORD_EMBEDDING_CACHE is None:
        with open(_WORD_EMBEDDING_CACHE_PATH, "rb") as f:
            raw = pickle.load(f)
        _WORD_EMBEDDING_CACHE = {w: np.asarray(v, dtype=np.float64) for w, v in raw.items()}
    return _WORD_EMBEDDING_CACHE


@dataclass(frozen=True)
class WordOverlapRow:
    """One row in the word-overlap panel — a pair of words plus their weights.

    Each row is either:
    - ``kind == "matched"``: ``left`` and ``right`` are paired by GPT-embedding
      cosine similarity (recorded in ``similarity``). Identical strings simply
      surface as a matched row with ``similarity == 1.0``.
    - ``kind == "unmatched"``: one side lacked an embedding in the cache, so
      its word falls through alone. The other column is ``None``.
    """

    left: Optional[str]
    right: Optional[str]
    left_weight: float = 0.0
    right_weight: float = 0.0
    similarity: float = 0.0
    kind: str = "matched"


def _normalize_word_input(
    words: Sequence[Union[str, Tuple[str, float]]],
) -> List[Tuple[str, float]]:
    """Accept either ['word', ...] or [('word', weight), ...] inputs.

    Bare strings get weight 0.0 so existing callers (and tests) keep working.
    """
    out: List[Tuple[str, float]] = []
    for item in words:
        if isinstance(item, str):
            out.append((item, 0.0))
        else:
            w, weight = item
            out.append((str(w), float(weight)))
    return out


def _compute_word_overlap_rows(
    words_a: Sequence[Union[str, Tuple[str, float]]],
    words_b: Sequence[Union[str, Tuple[str, float]]],
    embeddings: Optional[Mapping[str, np.ndarray]] = None,
) -> List[WordOverlapRow]:
    """Greedy 1:1 word pairing by GPT cosine similarity, no threshold.

    Each word in ``words_a`` is matched to at most one word in ``words_b``.
    Pairs are taken in descending similarity (so identical strings, which
    yield cosine 1.0, surface at the top). Pairs whose embeddings are
    missing fall through as ``kind="unmatched"`` rows at the bottom.

    Inputs may be plain strings or (word, weight) tuples. Weights are
    preserved on each row for downstream font-size scaling.
    """
    pairs_a = _normalize_word_input(words_a)
    pairs_b = _normalize_word_input(words_b)

    if embeddings is None:
        try:
            embeddings = _load_word_embeddings()
        except FileNotFoundError:
            embeddings = {}

    candidates: List[Tuple[float, str, float, str, float]] = []
    for a, wa in pairs_a:
        va = embeddings.get(a)
        if va is None:
            continue
        na = float(np.linalg.norm(va))
        if na == 0.0:
            continue
        for b, wb in pairs_b:
            vb = embeddings.get(b)
            if vb is None:
                continue
            nb = float(np.linalg.norm(vb))
            if nb == 0.0:
                continue
            sim = float(np.dot(va, vb) / (na * nb))
            candidates.append((sim, a, wa, b, wb))
    candidates.sort(reverse=True)

    used_a: set = set()
    used_b: set = set()
    rows: List[WordOverlapRow] = []
    for sim, a, wa, b, wb in candidates:
        if a in used_a or b in used_b:
            continue
        used_a.add(a)
        used_b.add(b)
        rows.append(WordOverlapRow(
            left=a, right=b, left_weight=wa, right_weight=wb,
            similarity=sim, kind="matched",
        ))

    unmatched_a = [(a, w) for a, w in pairs_a if a not in used_a]
    unmatched_b = [(b, w) for b, w in pairs_b if b not in used_b]
    for i in range(max(len(unmatched_a), len(unmatched_b))):
        la, lw = unmatched_a[i] if i < len(unmatched_a) else (None, 0.0)
        rb, rw = unmatched_b[i] if i < len(unmatched_b) else (None, 0.0)
        rows.append(WordOverlapRow(
            left=la, right=rb, left_weight=lw, right_weight=rw,
            similarity=0.0, kind="unmatched",
        ))

    return rows


def _render_one_word_venn(
    ax,
    *,
    cx: float,
    entry: Tuple[Union[str, int], Union[str, int], float],
    header: str,
    embeddings: Dict,
    top_k: int,
) -> None:
    """Draw one 2-circle word Venn centred at ``cx`` on a [0, 1] data canvas.

    Shared by ``draw_word_venns`` (two side-by-side at cx 0.46 / 1.54 on a
    [0, 2] axis) and the per-panel Fig 2 row-1 use (cx 0.5 on a [0, 1] axis).
    """
    from matplotlib.patches import Circle, FancyBboxPatch

    t1, t2, sim = entry
    cy, fs = 0.50, 7.0

    def _stack(words, x, color, *, ncols=1, step=0.074, col_gap=0.185):
        words = [w for w in words if w]
        if not words:
            return
        nrows = int(np.ceil(len(words) / ncols))
        y0 = cy + (nrows - 1) / 2 * step
        for idx, w in enumerate(words):
            c, r = divmod(idx, nrows)
            xc = x + (c - (ncols - 1) / 2) * col_gap
            ax.text(xc, y0 - r * step, w, ha="center", va="center", fontsize=fs,
                    color=color, family=_WORD_OVERLAP_FONT_FAMILY)

    wa = _load_twitter_top_words_with_weights(t1, k=top_k)
    wb = _load_twitter_top_words_with_weights(t2, k=top_k)
    rows = _compute_word_overlap_rows(wa, wb, embeddings=embeddings)
    shared = [
        (rw.left, rw.right) for rw in rows
        if rw.kind == "matched" and rw.similarity > _WORD_OVERLAP_HIGHLIGHT_THRESHOLD
    ]
    used_l = {l for l, _ in shared}
    used_r = {rr for _, rr in shared}
    uniq_l = [w for w, _ in wa if w not in used_l]
    uniq_r = [w for w, _ in wb if w not in used_r]

    if shared:
        r, dx = 0.27, 0.145
    else:
        r, dx = 0.225, 0.245
    xL, xR = cx - dx, cx + dx

    ax.add_patch(FancyBboxPatch(
        (cx - 0.46, 0.16), 0.92, 0.74,
        boxstyle="round,pad=0,rounding_size=0.04",
        fill=False, edgecolor="#dddddd", linewidth=1.0,
    ))
    ax.text(cx, 0.95, header, ha="center", va="top", fontsize=10,
            fontweight="bold", color="#333333", family=_WORD_OVERLAP_FONT_FAMILY)

    for xc in (xL, xR):
        ax.add_patch(Circle((xc, cy), r, facecolor="#cccccc", alpha=0.16,
                            edgecolor="#888888", linewidth=1.0))
    ax.text(xL, cy + r + 0.02, _pretty_topic(t1), ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#222222",
            family=_WORD_OVERLAP_FONT_FAMILY)
    ax.text(xR, cy + r + 0.02, _pretty_topic(t2), ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#222222",
            family=_WORD_OVERLAP_FONT_FAMILY)
    ax.text(cx, 0.205, f"sim = {sim:.3f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#444444",
            family=_WORD_OVERLAP_FONT_FAMILY)

    if shared:
        _stack(uniq_l, xL - 0.46 * r, _WORD_OVERLAP_PLAIN_COLOR, step=0.082)
        _stack(uniq_r, xR + 0.46 * r, _WORD_OVERLAP_PLAIN_COLOR, step=0.082)
        _stack([f"{l}–{rr}" for l, rr in shared], cx,
               _WORD_OVERLAP_HIGHLIGHT_COLOR, step=0.082)
    else:
        _stack(uniq_l, xL, _WORD_OVERLAP_PLAIN_COLOR, ncols=2)
        _stack(uniq_r, xR, _WORD_OVERLAP_PLAIN_COLOR, ncols=2)


def draw_single_word_venn(
    ax,
    *,
    which: str = "low",
    top_k: int = _WORD_OVERLAP_TOP_K,
    panel_label: Optional[str] = None,
) -> None:
    """One word Venn (``"low"`` or ``"high"``) centred on a [0, 1] x [0, 1] axis."""
    low, high = _select_us_twitter_extremes()
    entry = low if which == "low" else high
    header = "Low similarity" if which == "low" else "High similarity"
    try:
        embeddings = _load_word_embeddings()
    except FileNotFoundError:
        embeddings = {}

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    _render_one_word_venn(
        ax, cx=0.5, entry=entry, header=header,
        embeddings=embeddings, top_k=top_k,
    )
    if panel_label:
        ax.set_title(panel_label, fontsize=12, fontweight="bold", pad=8)


def draw_word_venns(ax, *, top_k: int = _WORD_OVERLAP_TOP_K,
                    panel_label: Optional[str] = None) -> None:
    """Two 2-circle word Venns illustrating semantic similarity (US Twitter).

    For the lowest- and highest-similarity topic pairs, each Venn shows the
    pair's top-K TF-IDF words: the overlap holds the words matched across the
    two topics with embedding cosine > the highlight threshold (0.40), while
    each lobe lists the words unique to one topic. A low-similarity pair has an
    (almost) empty overlap; a high-similarity pair fills it.
    """
    low, high = _select_us_twitter_extremes()
    try:
        embeddings = _load_word_embeddings()
    except FileNotFoundError:
        embeddings = {}

    # 2:1 equal-aspect canvas so the circles render round (two Venns side by side).
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    for entry, cx, header in [
        (low, 0.46, "Low similarity"),
        (high, 1.54, "High similarity"),
    ]:
        _render_one_word_venn(
            ax, cx=cx, entry=entry, header=header,
            embeddings=embeddings, top_k=top_k,
        )

    if panel_label:
        ax.set_title(panel_label, fontsize=12, fontweight="bold", pad=8)


class Plotter:
    """Builds Fig 1, Fig 2, Fig 3 for one analysis variant (main + robustness)."""

    def __init__(
        self,
        us_survey: str = "anes",
        cn_survey: str = "wvs",
        eu_survey: str = "evs_resample2",
        twitter_src: str = "twitter",
        weibo_src: str = "weibo",
        eutwitter_src: str = "eutwitter",
        embedding_type: str = "gpt",
        figure_folder: Union[str, Path] = FIGURE_DIR,
        date_prefix: Optional[str] = None,
        loess_frac: float = 0.4,
        n_bootstrap: int = 1000,
        regions: Optional[Sequence[RegionSpec]] = None,
        pedagogical: bool = True,
        trend: str = "linear",
        spectral_matrix: str = "cov",
        social_year: str = "average",
    ):
        self.embedding_type = embedding_type
        # Fig 3 spectrum: pairwise-available covariance ("cov", main) or the
        # same matrix rescaled to correlations ("corr", robustness check).
        dimension_results_dir(spectral_matrix)  # validates the name
        self.spectral_matrix = spectral_matrix
        # Social-media opinions: "average" (main; pairwise correlations pool
        # each user's posts over all years, Fig 3 uses every yearly matrix)
        # or one year, e.g. "2021" (annual robustness: correlations and
        # spectra from that year only). Surveys are single cross-sections.
        self.social_year = str(social_year)
        self.loess_frac = loess_frac
        # Fig 2 |r|-vs-similarity trend: "linear" (OLS line, first-order only,
        # easier for a general audience) or "lowess" (bootstrapped LOWESS).
        if trend not in ("linear", "lowess"):
            raise ValueError(f"trend must be 'linear' or 'lowess', got {trend!r}")
        self.trend = trend
        self.n_bootstrap = n_bootstrap
        # When False, drop the explanatory panels (Fig 2 word-overlap row,
        # Fig 3 toy-cloud + scree rows) and keep only the real-data panels.
        # Robustness checks set this False; the main figure keeps them.
        self.pedagogical = pedagogical
        self.figure_folder = Path(figure_folder)
        self.figure_folder.mkdir(parents=True, exist_ok=True)
        self.date_prefix = date_prefix or datetime.now().strftime("%Y%m%d")

        if regions is None:
            eu_is_english_survey = "en" in eu_survey
            regions = [
                RegionSpec(
                    code="US", survey=us_survey, social=twitter_src,
                    survey_display="ANES (US)", social_display="Twitter (US)",
                ),
                RegionSpec(
                    code="CN", survey=cn_survey, social=weibo_src,
                    survey_display="WVS (CN)", social_display="Weibo (CN)",
                ),
                RegionSpec(
                    code="EU", survey=eu_survey, social=eutwitter_src,
                    survey_display=(
                        "EVS (EN)" if eu_is_english_survey else "EVS (EU)"
                    ),
                    social_display=(
                        "Twitter (EN)" if eutwitter_src != "eutwitter"
                        else "Twitter (EU)"
                    ),
                ),
            ]
        self.regions: List[RegionSpec] = list(regions)

        self.country_color: Dict[str, str] = {
            r.code: r.resolved_color() for r in self.regions
        }
        self.source_country: Dict[str, str] = {}
        self.source_medium: Dict[str, str] = {}
        self.source_display: Dict[str, str] = {}
        for r in self.regions:
            self.source_country[r.survey] = r.code
            self.source_country[r.social] = r.code
            self.source_medium[r.survey] = "Offline"
            self.source_medium[r.social] = "Online"
            self.source_display[r.survey] = r.survey_display or r.survey
            self.source_display[r.social] = r.social_display or r.social

        # Back-compat single-source attributes (look up by code). Custom
        # region lists without a US/CN/EU slot leave the matching attribute as
        # whatever the constructor kwarg supplied.
        by_code = {r.code: r for r in self.regions}
        self.us_survey = by_code["US"].survey if "US" in by_code else us_survey
        self.cn_survey = by_code["CN"].survey if "CN" in by_code else cn_survey
        self.eu_survey = by_code["EU"].survey if "EU" in by_code else eu_survey
        self.twitter = by_code["US"].social if "US" in by_code else twitter_src
        self.weibo = by_code["CN"].social if "CN" in by_code else weibo_src
        self.eutwitter = by_code["EU"].social if "EU" in by_code else eutwitter_src

    # -- output paths --------------------------------------------------------

    def _output_path(self, task_name: str, fig_name: str) -> Path:
        return (
            self.figure_folder
            / f"{self.date_prefix}_{task_name}_results_{fig_name}.pdf"
        )

    # -- per-panel helpers ---------------------------------------------------

    # Shared x-axis labels and linestyles for every dash plot. Survey sources
    # render as solid dashes, social-media sources as actually-dashed dashes.
    _OFFLINE_LABEL = "Offline\n(Surveys)"
    _ONLINE_LABEL = "Online\n(Social Media)"
    _MEDIA_DASH_PATTERN = (0, (4, 2))

    def _label_panel(self, ax, letter: str, *, x: float = -0.12, y: float = 1.06) -> None:
        """Drop an uppercase italic panel label ('A') at the top-left.

        PNAS submission style (callers pass lowercase letters). Position is
        in axes coordinates so it survives bbox_inches='tight' on save. 3D
        axes need ``text2D`` because their ``text`` signature requires a
        z-coordinate.
        """
        text_kwargs = dict(
            fontsize=14, fontweight="bold", fontstyle="italic", va="bottom", ha="left",
        )
        label = letter.upper()
        if getattr(ax, "name", "") == "3d":
            ax.text2D(x, y, label, transform=ax.transAxes, **text_kwargs)
        else:
            ax.text(x, y, label, transform=ax.transAxes, **text_kwargs)

    def _decimals_for(self, src: str) -> int:
        """Decimal places for |r| labels — 3 across offline and online for
        consistency. (The spec originally used 2 for social media; that's
        been unified per design feedback.)
        """
        return 3

    def _scatter_with_fit(
        self, ax, df: pd.DataFrame, src: str
    ) -> None:
        """Scatter cloud + OLS fit + |r|/N annotation in a single panel.

        Survey opinions are 1..5 integers, so points stack up on the grid; we
        add a small uniform jitter (only for display) and use a very low alpha
        so the fitted line dominates. Social-media values are continuous —
        still rendered with a low alpha because the cloud has hundreds of
        thousands of points.
        """
        valid = df.dropna(subset=["x", "y"]).copy()
        if len(valid) < 2:
            ax.text(
                0.5, 0.5, "data not available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="#888888",
            )
            ax.set_xticks([]); ax.set_yticks([])
            return

        color = self.country_color[self.source_country[src]]
        is_survey = src not in SOCIAL_SOURCES
        if is_survey:
            # Gaussian jitter (sigma=0.08) is the conventional academic choice
            # for Likert-style data — softer than uniform, concentrates points
            # near each integer cell.
            rng = np.random.default_rng(0)
            jx = rng.normal(scale=0.08, size=len(valid))
            jy = rng.normal(scale=0.08, size=len(valid))
            x_plot = valid["x"].to_numpy() + jx
            y_plot = valid["y"].to_numpy() + jy
            alpha = 0.025
        else:
            # Online clouds have hundreds of thousands of points; render them
            # ~3x lighter than the survey scatter so the fitted line dominates.
            x_plot = valid["x"].to_numpy()
            y_plot = valid["y"].to_numpy()
            alpha = 0.008

        ax.scatter(
            x_plot, y_plot,
            s=6, alpha=alpha, color=color, edgecolors="none", rasterized=True,
        )
        slope, intercept = np.polyfit(valid["x"], valid["y"], 1)
        x_line = np.linspace(valid["x"].min(), valid["x"].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=2.8)

        abs_r, n = compute_abs_pearson(valid["x"], valid["y"])
        decimals = self._decimals_for(src)
        ax.text(
            0.04, 0.96,
            f"|r| = {abs_r:.{decimals}f}\nN = {n:,}",
            transform=ax.transAxes, ha="left", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85),
        )
        ax.set_title(self.source_display[src], fontsize=12, fontweight="bold")
        ax.set_xlabel(df.attrs.get("x_topic", "LGBT"), fontsize=10)
        ax.set_ylabel(df.attrs.get("y_topic", "Environment"), fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def _case_study_signed_r(self, src: str) -> float:
        """Signed Pearson r for the case-study pair (headline alignment number).

        Falls back to the aggregate CSV's |r| (assumed positive) when the
        individual-level data isn't exported locally.
        """
        try:
            df = get_case_study_pair(src)
            valid = df.dropna(subset=["x", "y"])
            if len(valid) < 3:
                raise ValueError("too few points")
            rr = float(np.corrcoef(valid["x"], valid["y"])[0, 1])
            if np.isnan(rr):
                raise ValueError("nan correlation")
            return rr
        except (FileNotFoundError, KeyError, ValueError):
            return lookup_case_study_correlation(src)

    def _case_study_ellipse(self, src: str) -> Optional[Dict[str, object]]:
        """Standardized scatter + its correlation ellipse.

        Each topic is mean-centered and scaled to unit variance, so the cloud's
        covariance *is* the correlation matrix ``[[1, r], [r, 1]]``. The ellipse
        then encodes alignment alone: its **eccentricity tracks |r|** (thin =
        strongly aligned, round = unaligned) and it orients at ~45° (135° for
        r<0). Standardizing also makes every country comparable in size,
        regardless of each topic's or platform's native scale.

        Returns a dict with ``pts`` (the standardized points), ``angle``
        (degrees), and ``smaj``/``smin`` (the √eigenvalues). ``None`` when
        individual data isn't exported locally.
        """
        try:
            df = get_case_study_pair(src)
            valid = df.dropna(subset=["x", "y"])
            if len(valid) < 5:
                return None
            xy = valid[["x", "y"]].to_numpy(dtype=float)
            xy = xy - xy.mean(axis=0)
            std = xy.std(axis=0)
            std[std == 0] = 1.0
            xy = xy / std            # standardize → covariance is the corr matrix
            cov = np.cov(xy.T)
            w, vec = np.linalg.eigh(cov)
            w = np.clip(w, 1e-12, None)

            gamma = _ELLIPSE_EXAGGERATION
            if gamma != 1.0:
                # Stretch the cloud along its principal axes so the new
                # covariance is V·Λ^γ·V^T — amplifies anisotropy for both the
                # points and the fitted ellipse, keeping them consistent.
                transform = vec @ np.diag(w ** ((gamma - 1.0) / 2.0)) @ vec.T
                xy = xy @ transform
                w = w ** gamma

            order = np.argsort(w)[::-1]            # descending: major first
            major = vec[:, order[0]]
            angle = float(np.degrees(np.arctan2(major[1], major[0])))
            return {
                "pts": xy,
                "angle": angle,
                "smaj": float(np.sqrt(w[order[0]])),
                "smin": float(np.sqrt(w[order[1]])),
            }
        except (FileNotFoundError, KeyError, ValueError):
            return None

    def _correlation_ellipse_panel(
        self, ax, sources: Sequence[str], *, panel_label: str,
    ) -> None:
        """Overlay each country's standardized scatter cloud with its correlation ellipse.

        For every country we draw a faint, subsampled, standardized scatter of
        the actual LGBT × Environment opinions and fit the 2σ correlation
        ellipse on top — so each ellipse visibly hugs its own cloud (the
        toy-figure relationship). Because each topic is scaled to unit variance,
        the **ellipse width tracks |r| alone**: thin = strongly aligned, round =
        unaligned. A dot marks each end of the major (alignment) axis. All
        clouds share one origin and one scale, so offline the three differ
        markedly (US thin → CN round) while online they collapse toward a common
        shape.
        """
        from matplotlib.patches import Ellipse
        from matplotlib.lines import Line2D

        n_sigma = 2.0       # confidence-ellipse radius (≈86% of a 2D Gaussian)
        n_show = 500        # points drawn per cloud (subsampled for clarity)
        rng = np.random.default_rng(0)
        handles = []
        max_extent = 1e-6
        for src in sources:
            e = self._case_study_ellipse(src)
            if e is None:
                continue
            angle, smaj, smin, pts = e["angle"], e["smaj"], e["smin"], e["pts"]
            code = self.source_country[src]
            color = self.country_color[code]
            ar = min(abs(self._case_study_signed_r(src)), 0.999)

            idx = rng.choice(len(pts), min(len(pts), n_show), replace=False)
            disp = pts[idx].copy()
            if src not in SOCIAL_SOURCES:
                # Survey opinions are 1..5 integers, so the centered points fall
                # on a lattice; jitter them (display only) into a smooth cloud.
                disp = disp + rng.normal(scale=0.10, size=disp.shape)
            ax.scatter(disp[:, 0], disp[:, 1], s=7, color=color, alpha=0.18,
                       edgecolors="none", zorder=1, rasterized=True)
            ax.add_patch(Ellipse(
                (0.0, 0.0), width=2 * n_sigma * smaj, height=2 * n_sigma * smin,
                angle=angle, facecolor="none", edgecolor=color, linewidth=2.4,
                zorder=3,
            ))
            # Dots at the two ends of the major (principal/regression) axis.
            rad = np.radians(angle)
            ex, ey = n_sigma * smaj * np.cos(rad), n_sigma * smaj * np.sin(rad)
            ax.scatter([ex, -ex], [ey, -ey], s=44, color=color,
                       edgecolors="white", linewidths=0.8, zorder=4)
            handles.append(Line2D(
                [0], [0], color=color, lw=2.6, label=f"{code}   |r| = {ar:.3f}",
            ))
            max_extent = max(max_extent, n_sigma * smaj)

        lim = max_extent * 1.3
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axhline(0, color="#dddddd", lw=0.8, zorder=0)
        ax.axvline(0, color="#dddddd", lw=0.8, zorder=0)
        ax.set_xlabel("LGBT", fontsize=10)
        ax.set_ylabel("Environment / Climate", fontsize=10)
        ax.set_title(panel_label, fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(handles=handles, loc="upper right", fontsize=9,
                  frameon=True, framealpha=0.85)

    def _case_study_correlations(self) -> Dict[str, float]:
        """Compute or look up |r| for each (region x medium) cell."""
        out: Dict[str, float] = {}
        for r in self.regions:
            for src in (r.survey, r.social):
                try:
                    df = get_case_study_pair(src)
                    abs_r, _ = compute_abs_pearson(df["x"], df["y"])
                    if np.isnan(abs_r):
                        raise ValueError("nan correlation")
                    out[src] = abs_r
                except (FileNotFoundError, KeyError, ValueError):
                    # Fallback to the aggregate CSV (always available).
                    out[src] = lookup_case_study_correlation(src)
        return out

    # -- Tutorial pair selection (Fig 2 row 1) -------------------------------

    def _us_eu_concept_pairs(self) -> List[Dict[str, object]]:
        """All US/EU topic pairs that share a common Twitter social code.

        Each entry maps one social code to the offline topic name on each
        side (US ANES column vs. EU EVS column). Pairs of such concepts are
        what the tutorial row chooses between.
        """
        us_off_by_social = {v: k for k, v in OFFLINE_ONLINE_TOPIC_MAP["US"].items()}
        eu_off_by_social = {v: k for k, v in OFFLINE_ONLINE_TOPIC_MAP["EU"].items()}
        shared = sorted(set(us_off_by_social) & set(eu_off_by_social))
        pairs: List[Dict[str, object]] = []
        for i, sc1 in enumerate(shared):
            for sc2 in shared[i + 1:]:
                pairs.append({
                    "social_codes": (sc1, sc2),
                    "us_offline": (us_off_by_social[sc1], us_off_by_social[sc2]),
                    "eu_offline": (eu_off_by_social[sc1], eu_off_by_social[sc2]),
                })
        return pairs

    def _select_tutorial_pair(self) -> Optional[Dict[str, object]]:
        """Pick the topic pair best demonstrating "different offline, similar online".

        Loads the four pairwise-correlation CSVs (US survey, EU survey, US
        Twitter, EU Twitter), computes signed r for every concept-aligned
        pair, and ranks by ``Δoff / (Δon + 0.05)`` — a ratio that rewards a
        big offline gap while penalising any online gap, so the headline
        narrative ("online |r| matches, offline does not") reads cleanly.
        Pairs with any missing correlation are skipped. Returns ``None`` if
        no pair has complete data.
        """
        try:
            corr_us_off = load_pairwise_correlation(self.us_survey)
            corr_eu_off = load_pairwise_correlation(self.eu_survey)
            corr_us_on = load_pairwise_correlation(self.twitter)
            corr_eu_on = load_pairwise_correlation(self.eutwitter)
        except FileNotFoundError:
            return None

        candidates: List[Dict[str, object]] = []
        for entry in self._us_eu_concept_pairs():
            sc1, sc2 = entry["social_codes"]
            usa, usb = entry["us_offline"]
            eua, eub = entry["eu_offline"]
            us_off = signed_r_from_pairwise(corr_us_off, usa, usb)
            eu_off = signed_r_from_pairwise(corr_eu_off, eua, eub)
            us_on = signed_r_from_pairwise(corr_us_on, sc1, sc2)
            eu_on = signed_r_from_pairwise(corr_eu_on, sc1, sc2)
            if any(not np.isfinite(v) for v in (us_off, eu_off, us_on, eu_on)):
                continue
            delta_off = abs(abs(us_off) - abs(eu_off))
            delta_on = abs(abs(us_on) - abs(eu_on))
            score = delta_off / (delta_on + 0.05)
            candidates.append({
                **entry,
                "us_off": us_off, "eu_off": eu_off,
                "us_on": us_on, "eu_on": eu_on,
                "delta_off": delta_off, "delta_on": delta_on,
                "score": score,
            })
        if not candidates:
            return None
        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates[0]

    def _tutorial_regression_panel(
        self,
        ax,
        *,
        kind: str,
        pair: Dict[str, object],
        panel_label: str,
    ) -> None:
        """Two-region (US, EU) standardized scatter + OLS regression line.

        ``kind`` is ``"offline"`` (survey r's) or ``"online"`` (social r's).
        For each region, we synthesize a 2D Gaussian sample whose theoretical
        correlation matches the looked-up signed r, draw the standardized
        cloud, and overlay its regression line. Because both topics are
        standardized to unit variance, the line slope equals r — so the line
        directly visualises the correlation strength: steep diagonal for
        strong alignment, flat for none.
        """
        n_show = 400
        x_lim = 3.0

        if kind == "offline":
            rows = [
                ("US", float(pair["us_off"])),
                ("EU", float(pair["eu_off"])),
            ]
        elif kind == "online":
            rows = [
                ("US", float(pair["us_on"])),
                ("EU", float(pair["eu_on"])),
            ]
        else:
            raise ValueError(f"unknown kind: {kind}")

        handles: List[Line2D] = []
        for code, signed_r in rows:
            color = self.country_color[code]
            seed = hash((code, kind, pair["social_codes"])) & 0xFFFF
            r = float(max(min(signed_r, 0.999), -0.999))
            rng = np.random.default_rng(seed)
            cov = np.array([[1.0, r], [r, 1.0]])
            pts = rng.multivariate_normal([0.0, 0.0], cov, size=n_show)
            pts = (pts - pts.mean(axis=0)) / (pts.std(axis=0) + 1e-12)

            ax.scatter(pts[:, 0], pts[:, 1], s=8, color=color, alpha=0.18,
                       edgecolors="none", zorder=1, rasterized=True)
            x_line = np.linspace(-x_lim + 0.2, x_lim - 0.2, 50)
            ax.plot(x_line, r * x_line, color=color, linewidth=2.8, zorder=3)
            handles.append(Line2D(
                [0], [0], color=color, lw=2.6,
                label=f"{code}   |r| = {abs(signed_r):.3f}",
            ))

        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(-x_lim, x_lim)
        ax.set_aspect("equal")
        ax.axhline(0, color="#dddddd", lw=0.8, zorder=0)
        ax.axvline(0, color="#dddddd", lw=0.8, zorder=0)

        usa, usb = pair["us_offline"]
        eua, eub = pair["eu_offline"]
        x_label = usa if usa == eua else f"{usa} / {eua}"
        y_label = usb if usb == eub else f"{usb} / {eub}"
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_title(panel_label, fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(handles=handles, loc="upper right", fontsize=9,
                  frameon=True, framealpha=0.85)

    def _aligned_dash_plot(
        self,
        ax,
        values: Dict[str, float],
        *,
        y_label: str,
        title: Optional[str] = None,
        ylim: Optional[Tuple[float, float]] = None,
        decimals_fn: Optional[Callable[[str], int]] = None,
        autoscale_y: bool = False,
        invert_y: bool = False,
        connect: bool = False,
        range_span: bool = False,
        direction_notes: Optional[Tuple[str, str]] = None,
        intervals: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        """Shared dash-plot motif: one dash per (region × medium) cell at 2 ticks.

        Survey sources render solid; social-media sources render with a dashed
        pattern. Countries are distinguished by color, and ``values[src]``
        carries the y-value.

        ``connect`` joins each country's offline and online dashes with a thin
        segment (two observations of one society, not a trajectory); offline
        labels then sit left of their dash so the segment stays clear.
        ``range_span`` shades the cross-country range within each medium.
        ``direction_notes`` = (top, bottom) writes small cues just right of the
        y-axis at its two ends, so an inverted axis reads without the caption.
        ``intervals`` maps a source to its (lo, hi) bootstrap interval, drawn
        as a vertical bar through the dash centre; sources without one are
        drawn as plain dashes.
        """
        offline_sources = [r.survey for r in self.regions]
        online_sources = [r.social for r in self.regions]
        countries = [r.code for r in self.regions]
        tick_offline, tick_online = 0.0, 1.0
        dash_half_width = 0.16
        decimals_fn = decimals_fn or self._decimals_for

        # Minimum vertical space between two adjacent labels at the same tick,
        # in data units. Scales with the panel's y-range so the gap is always
        # roughly one line-height regardless of metric.
        v_max = max(values.values()) if values else 1.0
        v_min = min(values.values()) if values else 0.0
        y_span = max(v_max - v_min, 1e-6)
        min_label_gap = 0.07 * y_span

        if connect:
            for r in self.regions:
                ax.plot(
                    [tick_offline + dash_half_width, tick_online - dash_half_width],
                    [values[r.survey], values[r.social]],
                    color=self.country_color[r.code], linewidth=0.8, alpha=0.6,
                    zorder=1,
                )

        for tick, tick_sources in [
            (tick_offline, offline_sources),
            (tick_online, online_sources),
        ]:
            if range_span:
                ys = [values[s] for s in tick_sources]
                ax.fill_between(
                    [tick - dash_half_width - 0.04, tick + dash_half_width + 0.04],
                    min(ys), max(ys), color="0.88", linewidth=0, zorder=0,
                )
            # Offline labels go left of the dash when a connector leaves the
            # right edge; otherwise every label sits to the right.
            side = -1 if connect and tick == tick_offline else 1
            # Draw every dash at its actual y.
            for src in tick_sources:
                color = self.country_color[self.source_country[src]]
                y = values[src]
                is_offline = self.source_medium[src] == "Offline"
                ls = "solid" if is_offline else self._MEDIA_DASH_PATTERN
                ax.plot(
                    [tick - dash_half_width, tick + dash_half_width],
                    [y, y],
                    color=color, linewidth=3.5, linestyle=ls,
                    solid_capstyle="butt",
                )
                if intervals and src in intervals:
                    lo, hi = intervals[src]
                    ax.errorbar(
                        tick, y, yerr=[[y - lo], [hi - y]], fmt="none",
                        ecolor=color, elinewidth=1.2, capsize=3, zorder=3,
                    )

            # Stagger the labels so they don't overlap when two values are
            # close. Sort ascending; bump each subsequent label up by
            # min_label_gap if it crowds its predecessor.
            ordered = sorted(tick_sources, key=lambda s: values[s])
            label_positions = []
            prev = -float("inf")
            for src in ordered:
                actual = values[src]
                placed = max(actual, prev + min_label_gap)
                label_positions.append((src, actual, placed))
                prev = placed

            for src, actual, placed in label_positions:
                color = self.country_color[self.source_country[src]]
                decimals = decimals_fn(src)
                # Thin leader when the label is pushed away from its dash.
                edge = tick + side * dash_half_width
                if abs(placed - actual) > 1e-3:
                    ax.plot(
                        [edge, edge + side * 0.025],
                        [actual, placed],
                        color=color, linewidth=0.7, alpha=0.6,
                    )
                ax.text(
                    edge + side * 0.035, placed,
                    f"{actual:.{decimals}f}",
                    fontsize=9, va="center", ha="left" if side > 0 else "right",
                    color=color,
                )

        ax.set_xticks([tick_offline, tick_online])
        ax.set_xticklabels(
            [self._OFFLINE_LABEL, self._ONLINE_LABEL],
            fontsize=11, fontweight="bold",
        )
        ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
        ax.set_xlim(-0.5, 1.6)
        v_max = max(values.values()) if values else 1.0
        v_min = min(values.values()) if values else 0.0
        if ylim is not None:
            ax.set_ylim(*ylim)
        elif autoscale_y:
            # Tight axis around data, with a 10% margin on each side. Use this
            # for metrics like PR / eRank where 0 is far from the data.
            span = max(v_max - v_min, 1e-3)
            ax.set_ylim(v_min - 0.1 * span, v_max + 0.1 * span)
        else:
            ax.set_ylim(0, max(0.7, v_max * 1.18))
        # Dimensionality metrics (PR / eRank): higher = higher dimension =
        # *lower* alignment, so invert the axis to keep "up = more aligned"
        # consistent with the |r| panels.
        if invert_y:
            ax.set_ylim(ax.get_ylim()[::-1])
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if direction_notes:
            top_note, bottom_note = direction_notes
            for y, text, va in ((0.99, top_note, "top"), (0.01, bottom_note, "bottom")):
                ax.text(
                    0.015, y, text, transform=ax.transAxes,
                    ha="left", va=va, fontsize=7.5, color="0.4", linespacing=1.1,
                )

        # Legend shows only country colors — the x-axis tick labels already
        # carry the offline/online (and solid/dashed) distinction.
        handles = [
            Line2D([0], [0], color=self.country_color[c], lw=4, label=c)
            for c in countries
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.95)

    def _half_violin_panel(
        self,
        ax,
        sources: Sequence[str],
        *,
        panel_label: str,
    ) -> None:
        """Per-country half-violin (KDE pointing right) of all-pair |r|, with mean marker.

        Replaces the old strip plot + separate mean dash row. The mean is
        drawn as a short horizontal segment inside the violin at its mean |r|.
        """
        from scipy.stats import gaussian_kde

        countries = [r.code for r in self.regions]
        positions = {r.code: i for i, r in enumerate(self.regions)}
        width = 0.40

        for src in sources:
            country = self.source_country[src]
            color = self.country_color[country]
            x_center = positions[country]
            values = self.pair_correlations(src)["correlation"].to_numpy()
            values = values[np.isfinite(values)]
            if len(values) < 3:
                continue
            kde = gaussian_kde(values)
            y_grid = np.linspace(0.0, 0.7, 240)
            density = kde(y_grid)
            density = density / density.max() * width  # rescaled to panel width
            ax.fill_betweenx(
                y_grid, x_center, x_center + density,
                color=color, alpha=0.15, linewidth=0,
            )
            ax.plot(
                np.full_like(y_grid, x_center, dtype=float), y_grid,
                color=color, linewidth=0.5, alpha=0.45,
            )
            mean_v = float(values.mean())
            mean_density = float(density[np.argmin(np.abs(y_grid - mean_v))])
            ax.plot(
                [x_center, x_center + mean_density],
                [mean_v, mean_v],
                color=color, linewidth=2.4,
            )
            decimals = self._decimals_for(src)
            ax.text(
                x_center + mean_density + 0.04, mean_v,
                f"{mean_v:.{decimals}f}",
                color=color, fontsize=9, va="center",
            )

        ax.set_xticks([positions[c] for c in countries])
        ax.set_xticklabels(countries, fontsize=12, fontweight="bold")
        ax.set_ylabel("|r|", fontsize=12, fontweight="bold")
        ax.set_title(panel_label, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 0.7)
        ax.set_xlim(-0.5, len(self.regions) - 0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def _word_overlap_panel(
        self,
        ax,
        *,
        panel_label: Optional[str] = None,
        top_k: int = _WORD_OVERLAP_TOP_K,
    ) -> None:
        """Anchor 'cosine similarity = 0.X' in concrete word-pair geometry.

        Always uses US Twitter (the lowest- and highest-similarity topic pair
        from ``embedding/dynamic_embedding_twitter_gpt.csv``). For each pair,
        the top-K words by TF-IDF in each topic are paired 1:1 by GPT-embedding
        cosine similarity (greedy, sorted descending). Row encoding:

        - **Row position**: rows are ordered by similarity descending, so the
          closest word pair sits at the top.
        - **Color**: each row is colored by its pair's cosine similarity via a
          YlOrRd colormap clipped to ``[VMIN, VMAX]``. Both columns share the
          row color so the eye links the matched words.
        - **Font size**: linearly scaled by each word's own TF-IDF weight, so
          the most-distinctive words appear largest.

        The high-similarity pair therefore shows a stack of saturated-red,
        large-font matched rows; the low-similarity pair shows pale-yellow,
        weaker matches.
        """
        # Prefer the precomputed artifact (cheap JSON read) over the live
        # compute (~80 MB pickle + cosine sweep). prepare_word_overlap.py
        # writes the artifact; fall back to live if it's missing or its
        # top_k differs from the requested one.
        cached = _load_cached_word_overlap()
        pair_blocks: List[Dict] = []
        all_weights: List[float] = []
        if cached is not None and cached.get("top_k") == top_k:
            for label_key, label_txt in (
                ("low", "Low similarity"),
                ("high", "High similarity"),
            ):
                entry = cached["pairs"][label_key]
                rows = [
                    WordOverlapRow(
                        left=r.get("left"),
                        right=r.get("right"),
                        left_weight=float(r.get("left_weight", 0.0)),
                        right_weight=float(r.get("right_weight", 0.0)),
                        similarity=float(r.get("similarity", 0.0)),
                        kind=r.get("kind", "matched"),
                    )
                    for r in entry["rows"]
                ]
                pair_blocks.append({
                    "label": label_txt, "t1": entry["topic_a"],
                    "t2": entry["topic_b"],
                    "sim": float(entry["topic_similarity"]), "rows": rows,
                })
                all_weights.extend([w["tf_idf"] for w in entry["words_a"]])
                all_weights.extend([w["tf_idf"] for w in entry["words_b"]])
        else:
            low, high = _select_us_twitter_extremes()
            embeddings = _load_word_embeddings()
            for label_txt, (t1, t2, sim) in (
                ("Low similarity", low),
                ("High similarity", high),
            ):
                wa = _load_twitter_top_words_with_weights(t1, k=top_k)
                wb = _load_twitter_top_words_with_weights(t2, k=top_k)
                rows = _compute_word_overlap_rows(wa, wb, embeddings=embeddings)
                pair_blocks.append({
                    "label": label_txt, "t1": t1, "t2": t2, "sim": sim, "rows": rows,
                })
                all_weights.extend([w for _, w in wa] + [w for _, w in wb])

        weight_lo, weight_hi = min(all_weights), max(all_weights)
        font_span = _WORD_OVERLAP_FONT_SIZE_MAX - _WORD_OVERLAP_FONT_SIZE_MIN

        def fontsize_for(weight: float) -> float:
            if weight_hi == weight_lo:
                return (_WORD_OVERLAP_FONT_SIZE_MIN + _WORD_OVERLAP_FONT_SIZE_MAX) / 2
            t = (weight - weight_lo) / (weight_hi - weight_lo)
            return _WORD_OVERLAP_FONT_SIZE_MIN + t * font_span

        def color_for(row: WordOverlapRow) -> str:
            # Highlight only matched pairs above the similarity threshold;
            # everything else (weaker matches, unmatched) renders dark.
            if row.kind == "matched" and row.similarity > _WORD_OVERLAP_HIGHLIGHT_THRESHOLD:
                return _WORD_OVERLAP_HIGHLIGHT_COLOR
            return _WORD_OVERLAP_PLAIN_COLOR

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        x_cols = [0.10, 0.32, 0.62, 0.84]
        y_pair_header = 0.965
        y_topic_header = 0.905
        y_word_start = 0.84
        # Reserve a little room at the bottom for the one-line legend.
        y_legend_top = 0.06
        max_rows = max(len(block["rows"]) for block in pair_blocks)
        y_word_step = (y_word_start - y_legend_top) / max(max_rows, 1)

        ax.text(
            (x_cols[0] + x_cols[1]) / 2, y_pair_header,
            f"Low similarity (sim = {pair_blocks[0]['sim']:.3f})",
            ha="center", va="top", fontsize=12, fontweight="bold",
            fontfamily=_WORD_OVERLAP_FONT_FAMILY,
        )
        ax.text(
            (x_cols[2] + x_cols[3]) / 2, y_pair_header,
            f"High similarity (sim = {pair_blocks[1]['sim']:.3f})",
            ha="center", va="top", fontsize=12, fontweight="bold",
            fontfamily=_WORD_OVERLAP_FONT_FAMILY,
        )
        # Vertical separator between the two pairs.
        ax.plot(
            [0.5, 0.5], [0.02, y_pair_header - 0.005],
            color="#cccccc", linewidth=0.8, zorder=0,
        )

        for i, block in enumerate(pair_blocks):
            x_left, x_right = x_cols[2 * i], x_cols[2 * i + 1]

            ax.text(
                x_left, y_topic_header, _pretty_topic(block["t1"]),
                ha="center", va="top", fontsize=11, fontweight="bold",
                color="#222222", fontfamily=_WORD_OVERLAP_FONT_FAMILY,
            )
            ax.text(
                x_right, y_topic_header, _pretty_topic(block["t2"]),
                ha="center", va="top", fontsize=11, fontweight="bold",
                color="#222222", fontfamily=_WORD_OVERLAP_FONT_FAMILY,
            )

            for row_idx, row in enumerate(block["rows"]):
                y = y_word_start - row_idx * y_word_step
                color = color_for(row)
                highlighted = color == _WORD_OVERLAP_HIGHLIGHT_COLOR

                if highlighted and row.left is not None and row.right is not None:
                    # Thin connector only for highlighted pairs.
                    ax.plot(
                        [x_left + 0.04, x_right - 0.04], [y, y],
                        color=color, alpha=0.35, linewidth=0.8, zorder=0,
                    )

                if row.left is not None:
                    ax.text(
                        x_left, y, row.left, ha="center", va="center",
                        fontsize=fontsize_for(row.left_weight),
                        color=color,
                        fontweight="bold" if highlighted else "normal",
                        fontfamily=_WORD_OVERLAP_FONT_FAMILY,
                    )
                if row.right is not None:
                    ax.text(
                        x_right, y, row.right, ha="center", va="center",
                        fontsize=fontsize_for(row.right_weight),
                        color=color,
                        fontweight="bold" if highlighted else "normal",
                        fontfamily=_WORD_OVERLAP_FONT_FAMILY,
                    )

        # One-line legend describing the encoding.
        ax.text(
            0.5, 0.02,
            f"highlighted = embedding similarity > {_WORD_OVERLAP_HIGHLIGHT_THRESHOLD:.2f}"
            "   ·   size = TF-IDF weight",
            ha="center", va="bottom", fontsize=10, color="#444444",
            fontfamily=_WORD_OVERLAP_FONT_FAMILY,
        )

        if panel_label:
            ax.set_title(panel_label, fontsize=12, fontweight="bold", pad=8)

    def _semantic_loess_panel(
        self,
        ax,
        sources: Sequence[str],
        *,
        line_style,
        panel_label: str,
    ) -> None:
        """|r| vs semantic similarity panel.

        Just the light background scatter + per-country trend (OLS line by
        default, bootstrapped LOWESS when ``trend="lowess"``). Concrete
        example pairs live in the dedicated examples row above so this panel
        stays uncluttered.
        """
        rng = np.random.default_rng(7)
        for src in sources:
            color = self.country_color[self.source_country[src]]
            corr = self.pair_correlations(src)
            sem = load_semantic_similarity(src, embedding_type=self.embedding_type)
            merged = (
                corr.merge(
                    sem[["topic_combination", "similarity"]],
                    on="topic_combination",
                    how="left",
                )
                .dropna(subset=["similarity"])
                .drop_duplicates(subset=["topic_combination"])
            )
            if merged.empty:
                continue
            ax.scatter(
                merged["similarity"].to_numpy(),
                merged["correlation"].to_numpy(),
                color=color, alpha=0.18, s=18, edgecolors="none",
            )
            x = merged["similarity"].to_numpy()
            y = merged["correlation"].to_numpy()
            if self.trend == "linear":
                x_grid, mean_curve = None, None
                if len(x) >= 2 and np.ptp(x) > 0:
                    slope, intercept = np.polyfit(x, y, 1)
                    x_grid = np.linspace(x.min(), x.max(), 100)
                    mean_curve = intercept + slope * x_grid
            else:
                x_grid, mean_curve, _lo, _hi = bootstrap_lowess_ci(
                    x, y,
                    frac=self.loess_frac,
                    n_bootstrap=self.n_bootstrap,
                    rng=rng,
                )
            if x_grid is not None:
                ax.plot(x_grid, mean_curve, color=color, linewidth=3, linestyle=line_style)

        ax.set_xlabel("Semantic similarity", fontsize=12)
        ax.set_ylabel("|r|", fontsize=12, fontweight="bold")
        ax.set_title(panel_label, fontsize=12, fontweight="bold")
        # Let matplotlib autoscale x to the actual similarity range — the
        # row 2 examples panel sits at [0, 1] but locking row 3 to that range
        # wastes white space where the data doesn't reach.
        ax.set_ylim(0, 0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # -- Figure 1 ------------------------------------------------------------

    def build_fig1(self) -> Figure:
        """Construct Figure 1 (case study) and return the matplotlib Figure.

        Two panels overlay each country's real scatter cloud with its covariance
        ellipse for the LGBT × Environment pair — offline (left) and online
        (right). Offline the three ellipses differ in shape and orientation;
        online they collapse toward a common shape, making "divergent offline →
        convergent online" legible at a glance.
        """
        plt.rcParams["font.family"] = "Times New Roman"
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(1, 2, wspace=0.22)

        offline_sources = [r.survey for r in self.regions]
        online_sources = [r.social for r in self.regions]

        ax_off = fig.add_subplot(gs[0, 0])
        ax_on = fig.add_subplot(gs[0, 1])
        self._correlation_ellipse_panel(
            ax_off, offline_sources, panel_label="Offline (Surveys)",
        )
        self._correlation_ellipse_panel(
            ax_on, online_sources, panel_label="Online (Social Media)",
        )
        self._label_panel(ax_off, "a")
        self._label_panel(ax_on, "b")
        return fig

    def make_fig1(self, task_name: str = "main") -> Path:
        """Build and save Figure 1; returns the saved PDF path."""
        fig = self.build_fig1()
        path = self._output_path(task_name, "fig1")
        fig.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        return path

    def build_fig2(self) -> Figure:
        """Construct Figure 2 — tutorial row + generalization row.

        With ``pedagogical`` (the default, used by the main figure), 2 rows x
        4 columns:
            Row 1 (tutorial): standardized correlation ellipse for the
                selected US/EU pair, offline and online (a, b); plus two word
                Venns anchoring 'semantic similarity = 0.X' in concrete
                US-Twitter TF-IDF keywords (c = low, d = high).
            Row 2 (generalization): per-country half-violin of pair |r| with
                mean marker (e, f); |r| vs semantic similarity LOWESS curves
                (g, h).

        Robustness checks (``pedagogical=False``) drop the tutorial row,
        leaving the four generalization panels (a-d) laid out 2 rows x 2
        cols: row 1 = half-violins (a offline, b online), row 2 = LOWESS
        curves (c offline, d online). The wider per-panel footprint keeps
        country labels and dashed-line legends legible.
        """
        plt.rcParams["font.family"] = "Times New Roman"
        offline_sources = [r.survey for r in self.regions]
        online_sources = [r.social for r in self.regions]

        if self.pedagogical:
            fig = plt.figure(figsize=(17, 9.0))
            gs = fig.add_gridspec(
                2, 4, height_ratios=[1.0, 0.82], hspace=0.32, wspace=0.32,
            )
            bottom_coords = [(1, 0), (1, 1), (1, 2), (1, 3)]
            bottom_labels = ("e", "f", "g", "h")

            # Row 1: tutorial — two scatter+regression panels + two word Venns.
            # The regression panels use the algorithmically-selected pair
            # (most similar online, most different offline between US and EU).
            pair = self._select_tutorial_pair()
            ax_reg_off = fig.add_subplot(gs[0, 0])
            ax_reg_on = fig.add_subplot(gs[0, 1])
            if pair is not None:
                self._tutorial_regression_panel(
                    ax_reg_off, kind="offline", pair=pair,
                    panel_label="Offline (Surveys)",
                )
                self._tutorial_regression_panel(
                    ax_reg_on, kind="online", pair=pair,
                    panel_label="Online (Social Media)",
                )
            else:
                for axx in (ax_reg_off, ax_reg_on):
                    axx.text(0.5, 0.5, "pairwise data missing",
                             transform=axx.transAxes, ha="center", va="center",
                             fontsize=11, color="#888888")
                    axx.set_xticks([])
                    axx.set_yticks([])

            ax_venn_low = fig.add_subplot(gs[0, 2])
            ax_venn_high = fig.add_subplot(gs[0, 3])
            draw_single_word_venn(ax_venn_low, which="low")
            draw_single_word_venn(ax_venn_high, which="high")

            self._label_panel(ax_reg_off, "a")
            self._label_panel(ax_reg_on, "b")
            self._label_panel(ax_venn_low, "c", x=-0.04, y=1.02)
            self._label_panel(ax_venn_high, "d", x=-0.04, y=1.02)
        else:
            fig = plt.figure(figsize=(12.5, 9.0))
            gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)
            bottom_coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
            bottom_labels = ("a", "b", "c", "d")

        # Distributions (half-violin) on row 1; LOWESS curves on row 2 (or
        # the bottom row of the 2x4 grid in pedagogical mode).
        ax_v_off = fig.add_subplot(gs[bottom_coords[0]])
        ax_v_on = fig.add_subplot(gs[bottom_coords[1]])
        ax_l_off = fig.add_subplot(gs[bottom_coords[2]])
        ax_l_on = fig.add_subplot(gs[bottom_coords[3]])
        self._half_violin_panel(
            ax_v_off, offline_sources, panel_label="Offline (Surveys)",
        )
        self._half_violin_panel(
            ax_v_on, online_sources, panel_label="Online (Social Media)",
        )
        self._semantic_loess_panel(
            ax_l_off, offline_sources, line_style="solid",
            panel_label="Offline (Surveys)",
        )
        self._semantic_loess_panel(
            ax_l_on, online_sources, line_style=self._MEDIA_DASH_PATTERN,
            panel_label="Online (Social Media)",
        )
        self._label_panel(ax_v_off, bottom_labels[0])
        self._label_panel(ax_v_on, bottom_labels[1])
        self._label_panel(ax_l_off, bottom_labels[2])
        self._label_panel(ax_l_on, bottom_labels[3])
        return fig

    def make_fig2(self, task_name: str = "main") -> Path:
        """Build and save Figure 2; returns the saved PDF path."""
        fig = self.build_fig2()
        path = self._output_path(task_name, "fig2")
        fig.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        return path

    # -- Figure 3 ------------------------------------------------------------

    def correlation_year(self, src: str) -> str:
        """Year of the pairwise correlations used for ``src`` (see social_year)."""
        return self.social_year if src in SOCIAL_SOURCES else "average"

    def spectral_year(self, src: str) -> Optional[str]:
        """Year of the spectral matrix used for ``src`` (None = all years)."""
        if src in SOCIAL_SOURCES and self.social_year != "average":
            return self.social_year
        return None

    def pair_correlations(self, src: str) -> pd.DataFrame:
        """load_pairwise_correlation for ``src`` at this plotter's year."""
        return load_pairwise_correlation(src, year=self.correlation_year(src))

    def _spectral_means(self, metric: str, normalize: bool = False) -> Dict[str, float]:
        """Return {src: mean(metric)} across both offline and online sources.

        With ``normalize`` the value is divided by the source's number of
        topics (the covariance matrix dimension), so 1 = every topic an
        independent direction and 1/K = a single shared direction.
        """
        if metric not in ("PR", "eRank"):
            raise ValueError(f"unknown spectral metric: {metric}")
        out: Dict[str, float] = {}
        for r in self.regions:
            for src in (r.survey, r.social):
                entries = load_spectral_summary(src, self.spectral_matrix, self.spectral_year(src))[metric]
                scale = len(RESTRICTED_TOPICS[src]) if normalize else 1
                out[src] = float(np.mean(entries)) / scale
        return out

    # Toy opinion clouds anchoring Fig 3's pedagogy. All three rendered in
    # the same 3D coordinate system so the reader sees the *collapse* of an
    # initially isotropic cloud onto a plane and then a line; (b) and (c)
    # also draw the supporting subspace (plane / line) to make the
    # condensation visually explicit.
    _TOY_CLOUDS: Tuple[Tuple[str, Tuple[float, float, float], str], ...] = (
        ("3D (isotropic)", (1.0, 1.0, 1.0), "isotropic"),
        ("2D (plate)", (1.0, 1.0, 0.005), "plate"),
        ("1D (line)", (1.0, 0.005, 0.005), "line"),
    )
    # Sequential colormap clearly outside the country palette (US blue / CN
    # red / EU purple). YlGn goes yellow -> dark green.
    _TOY_CMAP = "YlGn"
    _TOY_SAMPLE_SIZE = 500
    _TOY_AXIS_BOUND = 2.6

    def _sample_toy_cloud(
        self, eigenvalues: Tuple[float, float, float], seed: int
    ) -> np.ndarray:
        """Sample a 3D point cloud whose covariance has the given eigenvalues."""
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(self._TOY_SAMPLE_SIZE, 3))
        return X * np.sqrt(np.asarray(eigenvalues))[None, :]

    def _toy_cloud_panel(
        self,
        ax,
        eigenvalues: Tuple[float, float, float],
        *,
        title: str,
        seed: int,
        rendering: str,
    ) -> None:
        """Render one toy opinion cloud in the same 3D coordinate system.

        For ``plate`` and ``line``, the supporting subspace (a translucent
        plane at z=0 / a thick segment along the x-axis) is drawn first so
        the visible 'condensation' of points onto a lower-dimensional set is
        unambiguous.
        """
        X = self._sample_toy_cloud(eigenvalues, seed=seed)
        bound = self._TOY_AXIS_BOUND
        distances = np.linalg.norm(X, axis=1)

        # Draw the supporting subspace first so points float on top.
        if rendering == "plate":
            xx, yy = np.meshgrid([-bound, bound], [-bound, bound])
            zz = np.zeros_like(xx)
            ax.plot_surface(
                xx, yy, zz,
                color="0.65", alpha=0.18, edgecolor="none", shade=False,
                zorder=0,
            )
        elif rendering == "line":
            ax.plot(
                [-bound, bound], [0, 0], [0, 0],
                color="0.45", linewidth=3.0, alpha=0.7, zorder=0,
            )

        ax.scatter(
            X[:, 0], X[:, 1], X[:, 2],
            c=distances, cmap=self._TOY_CMAP,
            s=12, alpha=0.85, edgecolors="none", depthshade=False,
        )

        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        ax.set_zlim(-bound, bound)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=22, azim=35)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel("opinion 1", fontsize=9, labelpad=-10)
        ax.set_ylabel("opinion 2", fontsize=9, labelpad=-10)
        ax.set_zlabel("opinion 3", fontsize=9, labelpad=-10)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)

    def _toy_scree_panel(
        self, ax, eigenvalues: Tuple[float, float, float], *, seed: int,
    ) -> None:
        """Bar chart of normalized eigenvalues for one toy cloud.

        Computed on the *same* sample the row above shows so finite-N
        fluctuations match what the reader sees and the annotated PR is the
        number you'd get by hand.
        """
        X = self._sample_toy_cloud(eigenvalues, seed=seed)
        result = pca_loadings(X)
        lam = result["eigenvalues"]
        # Floor at zero so tiny negative numerical artefacts (e.g. -1e-17)
        # don't render as a downward bar.
        lam = np.maximum(lam, 0.0)
        normalized = lam / lam.sum() if lam.sum() > 0 else lam
        pr = participation_ratio(lam)
        erank = exponential_rank(lam)

        positions = np.arange(1, len(normalized) + 1)
        # Color each bar with the same YlGn ramp keyed by rank, so the visual
        # link between cloud color and scree color is preserved.
        cmap = plt.get_cmap(self._TOY_CMAP)
        bar_colors = [cmap(0.85 - 0.30 * (k - 1)) for k in positions]
        ax.bar(
            positions, normalized,
            color=bar_colors, edgecolor="white", linewidth=1.2,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels([f"$\\lambda_{k}$" for k in positions], fontsize=10)
        ax.set_ylabel("variance fraction", fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.text(
            0.96, 0.96,
            f"PR ≈ {pr:.3f}\neRank ≈ {erank:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Panel (a) titles for the three toy clouds, weakest to strongest bundling.
    _TOY_BUNDLING_TITLES = ("Unbundled", "Partly bundled", "Strongly bundled")
    # (top, bottom) cues on the inverted PR / eRank axes of Fig 3 (b, c).
    _SPECTRAL_DIRECTION_NOTES = (
        "lower rank\nmore collapsed", "higher rank\nless collapsed",
    )

    def _spectral_dash_panels(self, pr_ax, erank_ax) -> None:
        """PR and eRank (effective number of dimensions, out of K topics) by
        society and medium (Fig 3 B, C).

        Each panel has its own y-range, fitted to that metric's values and
        intervals: eRank >= PR by construction (exponentials of the order-1
        and order-2 Renyi entropies of the spectrum), so the two are compared
        in pattern, not level, and a shared axis would invite the latter.
        """
        k = len(RESTRICTED_TOPICS[self.regions[0].survey])
        means = {
            metric: self._spectral_means(metric, normalize=False)
            for metric in ("PR", "eRank")
        }
        # Bootstrap intervals from spectral_bootstrap.py, where available.
        boots = {
            src: load_spectral_bootstrap(src, self.spectral_matrix, self.spectral_year(src))
            for r in self.regions for src in (r.survey, r.social)
        }
        intervals = {
            metric: {
                src: (b[f"{metric}_lo"], b[f"{metric}_hi"])
                for src, b in boots.items() if b is not None
            }
            for metric in means
        }
        def metric_ylim(metric: str) -> Tuple[float, float]:
            values = list(means[metric].values()) + [
                v for pair in intervals[metric].values() for v in pair
            ]
            lo, hi = min(values), max(values)
            pad = 0.08 * (hi - lo)
            return (np.floor((lo - pad) / 0.5) * 0.5, np.ceil((hi + pad) / 0.5) * 0.5)

        for ax, metric, y_label, title in (
            (pr_ax, "PR", f"Effective dimensions (of K = {k})", "Participation ratio"),
            (erank_ax, "eRank", f"Effective dimensions (of K = {k})", "Exponential rank"),
        ):
            self._aligned_dash_plot(
                ax, means[metric],
                y_label=y_label, title=title, ylim=metric_ylim(metric),
                decimals_fn=lambda src: 2, invert_y=True,
                connect=True, range_span=True,
                direction_notes=self._SPECTRAL_DIRECTION_NOTES,
                intervals=intervals[metric],
            )

    def build_fig3(self) -> Figure:
        """Construct Figure 3.

        With ``pedagogical`` (the default, used by the main figure):
            (a) three toy opinion clouds, unbundled -> strongly bundled, each
                with its normalized PR: bundling concentrates variation in
                fewer directions.
            (b, c) normalized PR and eRank (divided by the number of topics K)
                by society and medium; each society's two dashes joined by a
                thin segment, the cross-societal range per medium shaded.

        Robustness checks (``pedagogical=False``) keep only (b, c), labelled
        (a, b).
        """
        plt.rcParams["font.family"] = "Times New Roman"

        if not self.pedagogical:
            fig = plt.figure(figsize=(13, 5))
            gs = fig.add_gridspec(1, 2, wspace=0.38)
            pr_ax = fig.add_subplot(gs[0, 0])
            erank_ax = fig.add_subplot(gs[0, 1])
            self._spectral_dash_panels(pr_ax, erank_ax)
            self._label_panel(pr_ax, "a", x=-0.08)
            self._label_panel(erank_ax, "b", x=-0.08)
            return fig

        fig = plt.figure(figsize=(13, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.0], hspace=0.30)

        # (a) toy clouds in one shared 3D coordinate system; the plate / line
        # variants overlay their supporting subspace so the collapse is
        # explicit. The normalized PR is computed on the very sample shown.
        clouds_gs = gs[0].subgridspec(1, len(self._TOY_CLOUDS), wspace=0.05)
        for j, ((_, eigs, rendering), title) in enumerate(
            zip(self._TOY_CLOUDS, self._TOY_BUNDLING_TITLES)
        ):
            seed = 101 + j
            ax = fig.add_subplot(clouds_gs[0, j], projection="3d")
            self._toy_cloud_panel(ax, eigs, title=title, seed=seed, rendering=rendering)
            lam = np.maximum(pca_loadings(self._sample_toy_cloud(eigs, seed=seed))["eigenvalues"], 0.0)
            ax.text2D(
                0.5, -0.04,
                f"PR = {participation_ratio(lam):.2f}   (K = {len(lam)} opinions)",
                transform=ax.transAxes, ha="center", va="top", fontsize=11,
            )
            if j == 0:
                self._label_panel(ax, "a", x=-0.05, y=1.04)

        # (B, C) PR and eRank of the real data.
        dash_gs = gs[1].subgridspec(1, 2, wspace=0.38)
        pr_ax = fig.add_subplot(dash_gs[0, 0])
        erank_ax = fig.add_subplot(dash_gs[0, 1])
        self._spectral_dash_panels(pr_ax, erank_ax)
        self._label_panel(pr_ax, "b", x=-0.08)
        self._label_panel(erank_ax, "c", x=-0.08)
        return fig

    def make_fig3(self, task_name: str = "main") -> Path:
        """Build and save Figure 3; returns the saved PDF path."""
        fig = self.build_fig3()
        path = self._output_path(task_name, "fig3")
        fig.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        return path


# -- Task orchestration (mirrors plot_prod_v3.run_all_tasks) -----------------


TASKS: List[Dict] = [
    {
        "name": "main",
        "us_survey": "anes", "cn_survey": "wvs", "eu_survey": "evs_resample2",
        "eutwitter_src": "eutwitter", "embedding_type": "gpt",
        "figures": [2, 3],
    },
    {
        # Supplement / robustness: main data with LOWESS curves (span 0.4)
        # instead of the main figure's linear fits (nonlinear alternative).
        "name": "supp_lowess",
        "us_survey": "anes", "cn_survey": "wvs", "eu_survey": "evs_resample2",
        "eutwitter_src": "eutwitter", "embedding_type": "gpt",
        "figures": [2], "trend": "lowess",
    },
    {
        "name": "robust_1_media",
        "us_survey": "anes_media", "cn_survey": "wvs_media",
        "eu_survey": "evs_media_resample2",
        "eutwitter_src": "eutwitter", "embedding_type": "gpt",
        "figures": [2, 3],
    },
    {
        "name": "robust_2_noputback",
        "us_survey": "anes", "cn_survey": "wvs", "eu_survey": "evs",
        "eutwitter_src": "eutwitter", "embedding_type": "gpt",
        "figures": [2, 3],
    },
    {
        "name": "robust_3_england",
        "us_survey": "anes", "cn_survey": "wvs", "eu_survey": "en_evs",
        "eutwitter_src": "entwitter", "embedding_type": "gpt",
        "figures": [2, 3],
    },
    {
        "name": "robust_5_word2vec",
        "us_survey": "anes", "cn_survey": "wvs", "eu_survey": "evs_resample2",
        "eutwitter_src": "eutwitter", "embedding_type": "dictionary",
        "figures": [2],
    },
    {
        # Survey topic similarity from the hand-curated keyword lists.
        "name": "robust_7_survey_curated",
        "us_survey": "anes", "cn_survey": "wvs", "eu_survey": "evs_resample2",
        "eutwitter_src": "eutwitter", "embedding_type": "gpt_curated",
        "figures": [2],
    },
    {
        # Survey topic similarity from embeddings of the full question text.
        "name": "robust_8_survey_question",
        "us_survey": "anes", "cn_survey": "wvs", "eu_survey": "evs_resample2",
        "eutwitter_src": "eutwitter", "embedding_type": "gpt_question",
        "figures": [2],
    },
    {
        # 4-region eu_nen variant: side-by-side US / CN / EN / EU-NEN.
        "name": "robust_6_eu_nen",
        "regions": [
            RegionSpec(
                "US", "anes", "twitter",
                survey_display="ANES (US)", social_display="Twitter (US)",
            ),
            RegionSpec(
                "CN", "wvs", "weibo",
                survey_display="WVS (CN)", social_display="Weibo (CN)",
            ),
            RegionSpec(
                "EN", "en_evs", "entwitter",
                survey_display="EVS (EN)", social_display="Twitter (EN)",
            ),
            RegionSpec(
                "EU_NEN", "eu_nen_evs", "eu_nentwitter",
                survey_display="EVS (EU-NEN)", social_display="Twitter (EU-NEN)",
            ),
        ],
        "embedding_type": "gpt",
        "figures": [2, 3],
    },
    {
        # Fig 3 from the pairwise-available correlation matrix: every topic
        # weighted equally, whatever its variance.
        "name": "robust_9_correlation",
        "us_survey": "anes", "cn_survey": "wvs", "eu_survey": "evs_resample2",
        "eutwitter_src": "eutwitter", "embedding_type": "gpt",
        "figures": [3], "spectral_matrix": "corr",
    },
    {
        # Social-media opinions of a single year instead of pooled over
        # years. 2021 is the only year in which all nine Twitter/X topics are
        # observed (Weibo covers it too). Fig 2: correlations of 2021; Fig 3:
        # Weibo's 2021 matrix instead of the mean of 2016-2023 (Twitter/X is
        # 2021 in the main analysis already).
        "name": "robust_10_annual",
        "us_survey": "anes", "cn_survey": "wvs", "eu_survey": "evs_resample2",
        "eutwitter_src": "eutwitter", "embedding_type": "gpt",
        "figures": [2, 3], "social_year": "2021",
    },
    {
        # Survey topic similarity from the uncapped selected keywords (before
        # the anchor / fragment / top-5 rules of cap_survey_keywords).
        "name": "robust_11_survey_selected",
        "us_survey": "anes", "cn_survey": "wvs", "eu_survey": "evs_resample2",
        "eutwitter_src": "eutwitter", "embedding_type": "gpt_selected",
        "figures": [2],
    },
]


def _task_by_name(name: str) -> Dict:
    for task in TASKS:
        if task["name"] == name:
            return task
    raise KeyError(f"Unknown task: {name}. Known: {[t['name'] for t in TASKS]}")


def task_plotter_kwargs(
    name: str,
    figure_folder: Union[str, Path] = FIGURE_DIR,
    date_prefix: Optional[str] = None,
    n_bootstrap: int = 1000,
) -> Dict:
    """Plotter keyword arguments for one task in TASKS."""
    task = _task_by_name(name)
    plotter_kwargs: Dict = dict(
        embedding_type=task["embedding_type"],
        figure_folder=figure_folder,
        date_prefix=date_prefix,
        loess_frac=task.get("loess_frac", 0.4),
        n_bootstrap=n_bootstrap,
        # Explanatory panels (Fig 2 word-overlap, Fig 3 toy clouds + scree)
        # only belong in the main figure; robustness checks omit them.
        pedagogical=task.get("pedagogical", name == "main"),
        trend=task.get("trend", "linear"),
        spectral_matrix=task.get("spectral_matrix", "cov"),
        social_year=task.get("social_year", "average"),
    )
    if "regions" in task:
        plotter_kwargs["regions"] = task["regions"]
    else:
        plotter_kwargs.update(
            us_survey=task["us_survey"],
            cn_survey=task["cn_survey"],
            eu_survey=task["eu_survey"],
            twitter_src="twitter",
            weibo_src="weibo",
            eutwitter_src=task["eutwitter_src"],
        )
    return plotter_kwargs


def run_task(
    name: str,
    figure_folder: Union[str, Path] = FIGURE_DIR,
    date_prefix: Optional[str] = None,
    n_bootstrap: int = 1000,
    write_stats: bool = True,
) -> Dict[int, Path]:
    """Run one task by name and return {fig_num: output_path}.

    With ``write_stats`` a plain-text statistics report for the task's
    sources is written next to the figures (see paper_stats.py).
    """
    task = _task_by_name(name)
    plotter = Plotter(**task_plotter_kwargs(
        name, figure_folder=figure_folder, date_prefix=date_prefix,
        n_bootstrap=n_bootstrap,
    ))
    # Variant spectra (results/corr) are computed on the clusters; skip the
    # task with a message instead of failing a full run before they exist.
    if plotter.spectral_matrix != "cov":
        folder = dimension_results_dir(plotter.spectral_matrix)
        missing = [
            src for r in plotter.regions for src in (r.survey, r.social)
            if not (folder / f"{_SPECTRAL_STEM_OVERRIDE.get(src, src)}-none-summary.json").exists()
        ]
        if missing:
            print(f"[figures] skipped {name}: no {plotter.spectral_matrix} results in {folder} for {missing} "
                  f"(run dimension_pipeline calculate_all / spectral_bootstrap all --matrix {plotter.spectral_matrix})")
            return {}
    paths: Dict[int, Path] = {}
    for fig_num in task["figures"]:
        method = getattr(plotter, f"make_fig{fig_num}")
        paths[fig_num] = method(task_name=name)
    if write_stats:
        from plotting import paper_stats  # imports this module; deferred to avoid a cycle

        STATS_DIR.mkdir(parents=True, exist_ok=True)
        stats_path = STATS_DIR / plotter._output_path(name, "stats").with_suffix(".txt").name
        paper_stats.write_report(plotter, stats_path, header=f"(task: {name})")
        print(f"[figures] stats written: {stats_path}")
    return paths


# Paper figures (main text + LOWESS supplement); every other task is a
# robustness check, run by robustness/run_robustness.py.
MAIN_TASKS = ("main", "supp_lowess")


def task_names(group: str = "all") -> List[str]:
    """Task names of one group: "main", "robustness" or "all"."""
    if group == "main":
        return [t["name"] for t in TASKS if t["name"] in MAIN_TASKS]
    if group == "robustness":
        return [t["name"] for t in TASKS if t["name"] not in MAIN_TASKS]
    if group == "all":
        return [t["name"] for t in TASKS]
    raise ValueError(f"group must be main, robustness or all, got {group!r}")


def run_all_tasks(
    group: str = "all",
    figure_folder: Union[str, Path] = FIGURE_DIR,
    date_prefix: Optional[str] = None,
    n_bootstrap: int = 1000,
) -> Dict[str, Dict[int, Path]]:
    """Run every task of ``group`` and return {task_name: {fig_num: output_path}}."""
    out: Dict[str, Dict[int, Path]] = {}
    for task in (t for t in TASKS if t["name"] in task_names(group)):
        print(f"[figures] running task: {task['name']}")
        out[task["name"]] = run_task(
            task["name"],
            figure_folder=figure_folder,
            date_prefix=date_prefix,
            n_bootstrap=n_bootstrap,
        )
    return out


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "all": run_all_tasks,
            "main": lambda **kw: run_all_tasks("main", **kw),
            "task": run_task,
        }
    )
