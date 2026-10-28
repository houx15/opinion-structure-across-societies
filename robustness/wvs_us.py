"""Robustness check: the same survey instrument offline (task robust_12_wvs_us).

ROBUSTNESS ONLY. The main U.S. sources stay ANES and U.S. Twitter/X with gun
control; nothing here changes them.

Offline, the U.S. source is WVS wave 7 (USA, fieldwork April-May 2017, the
only WVS-7 U.S. survey) on items mirroring the British EVS topics, so the
United States, China (WVS) and Great Britain (EVS, which shares the WVS
core) are measured with one questionnaire family. Gun control has no WVS item, so the
check uses eight topics in every source (common.topics.TOPICS_8): this file's
nine items without Prostitution (source ``wvs_us8``), U.S. Twitter/X without
gun control, and Great Britain and China without the topic that has no counterpart
(prostitution / sex work; attitudes toward the U.S.). Spectra of the
eight-topic sources: robustness/topics8.py.

WVS-7 USA items (common/topics.WVS_US_TOPICS), coded as in the main analysis
(SI "Topic measures"; validated by rebuilding the stored China file exactly,
see ``validate_china``):

    Abortion      Q184 abortion justifiable            1-10 -> 1..5
    LGBT          Q182 homosexuality justifiable       1-10 -> 1..5
    DeathPenalty  Q195 death penalty justifiable       1-10 -> 5..1 (as EVS)
    Prostitution  Q183 prostitution justifiable        1-10 -> 1..5
    Egalitarian   Q108 government vs people responsible 1-10 -> 5..1 (EVS direction)
    UnemplyAid    Q244 state aid for the unemployed as essential to democracy 1-10 -> 1..5
    WHO           Q88 confidence in the WHO (substitute for EVS health-care system) 1,2,4,5
    Press         Q66 confidence in the press (substitute for EVS social media)     1,2,4,5
    Environment   Q111 environment vs growth (the China item) env 1, other 3, growth 5

Negative codes (don't know, no answer, missing) are missing; respondents
under 18 or with no topic answered are dropped; no weights. Direction does
not affect |r| or the eigenvalues.

Semantic similarity reuses existing keyword centroids: the EVS topic for the
identical items (WHO uses EVS health-care system), the WVS China topic for
Environment, and ANES news media for Press.

Steps (``python -m robustness.wvs_us local --raw <WVS-7 csv>``): build
data/survey/wvs_us.dta, its pairwise correlations and topic similarities (read
by wvs_us8, restricted to its eight topics), and the nine-topic matrix whose
eight-topic subset robustness/topics8.py analyses; then
``python -m robustness.topics8 spectral`` and
``python -m plotting.figures task robust_12_wvs_us``.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from common.paths import EMBEDDING_DIR, SURVEY_DIR
from common.topics import WVS_US_TOPICS

SRC = "wvs_us"
YEAR = 2017

# topic -> (WVS-7 column, coding)
ITEMS = {
    "Abortion": ("Q184", "ten"),
    "LGBT": ("Q182", "ten"),
    "DeathPenalty": ("Q195", "ten_reversed"),
    "Prostitution": ("Q183", "ten"),
    "Egalitarian": ("Q108", "ten_reversed"),
    "UnemplyAid": ("Q244", "ten"),
    "WHO": ("Q88", "confidence"),
    "Press": ("Q66", "confidence"),
    "Environment": ("Q111", "environment"),
}

# topic -> (survey keyword region, topic there) whose keyword centroid represents it
SIMILARITY_SOURCE = {
    "Abortion": ("europe", "Abortion"),
    "LGBT": ("europe", "LGBT"),
    "DeathPenalty": ("europe", "DeathPenalty"),
    "Prostitution": ("europe", "Prostitution"),
    "Egalitarian": ("europe", "Egalitarian"),
    "UnemplyAid": ("europe", "UnemplyAid"),
    "WHO": ("europe", "HealthCare"),
    "Press": ("us", "Media"),
    "Environment": ("china", "Environment"),
}


def ten_to_five(s: pd.Series, reverse: bool = False) -> pd.Series:
    """1-10 scale -> adjacent pairs 1-2, ..., 9-10 -> 1..5 (reversed: 5..1)."""
    out = np.ceil(s.where(s.between(1, 10)) / 2)
    return 6 - out if reverse else out


def recode(s: pd.Series, coding: str) -> pd.Series:
    if coding == "ten":
        return ten_to_five(s)
    if coding == "ten_reversed":
        return ten_to_five(s, reverse=True)
    if coding == "confidence":   # four categories spread over 1..5, as EVS confidence items
        return s.map({1: 1.0, 2: 2.0, 3: 4.0, 4: 5.0})
    if coding == "environment":  # 1 environment, 2 growth, 3 other -> 1, 5, 3 (as China)
        return s.map({1: 1.0, 2: 5.0, 3: 3.0})
    raise ValueError(coding)


def read_raw(raw: str, country: str, columns) -> pd.DataFrame:
    df = pd.read_csv(raw, usecols=["B_COUNTRY_ALPHA", "Q262", *columns], low_memory=False)
    return df[df["B_COUNTRY_ALPHA"] == country].reset_index(drop=True)


def build_topics(raw: str) -> pd.DataFrame:
    """Topic scores of WVS-7 U.S. respondents aged 18+ who answered any topic."""
    df = read_raw(raw, "USA", [col for col, _ in ITEMS.values()])
    df = df[df["Q262"] >= 18]
    topics = pd.DataFrame({t: recode(df[col], coding) for t, (col, coding) in ITEMS.items()})
    topics = topics[WVS_US_TOPICS].dropna(how="all").reset_index(drop=True)
    topics["year"] = YEAR
    return topics


def build(raw: str) -> Path:
    """Write data/survey/wvs_us.dta (microdata; not tracked in git)."""
    topics = build_topics(raw)
    path = SURVEY_DIR / f"{SRC}.dta"
    topics.to_stata(path, write_index=False)
    answered = topics[WVS_US_TOPICS].notna()
    print(f"[{SRC}] {len(topics)} respondents -> {path}; answered per topic: {answered.sum().to_dict()}")
    return path


def validate_china(raw: str) -> dict:
    """Rebuild China's stored topic scores from the same raw file with the same
    rules (share of identical values per topic; all must be 1.0)."""
    cn = read_raw(raw, "CHN", ["Q182", "Q112", "Q185", "Q186", "Q111"])
    stored = pd.read_stata(SURVEY_DIR / "wvs.dta", convert_categoricals=False)
    marriage = pd.concat([ten_to_five(cn["Q185"], True), ten_to_five(cn["Q186"], True)], axis=1).mean(axis=1)
    built = {
        "LGBT": ten_to_five(cn["Q182"], reverse=True),
        "Corrup": ten_to_five(cn["Q112"], reverse=True),
        "Marriage": np.floor(marriage + 0.5),
        "Environment": recode(cn["Q111"], "environment"),
    }
    return {t: float(((v == stored[t]) | (v.isna() & stored[t].isna())).mean()) for t, v in built.items()}


def similarity() -> Path:
    """Write embedding/wvs_us_topic_distance.csv from existing keyword centroids."""
    from analysis.semantic.semantic_similarity import (
        DynamicSemanticEmbedding, _cosine_similarity_np, survey_topic_centroids,
    )

    dse = DynamicSemanticEmbedding()
    region_centroids = {r: survey_topic_centroids(r, dse, "top5") for r in {r for r, _ in SIMILARITY_SOURCE.values()}}
    centroid = {t: region_centroids[r][topic] for t, (r, topic) in SIMILARITY_SOURCE.items()}
    rows = [
        {"topic_id1": a, "topic_id2": b, "similarity_top5": _cosine_similarity_np(centroid[a], centroid[b])}
        for a, b in itertools.combinations(WVS_US_TOPICS, 2)
    ]
    path = EMBEDDING_DIR / f"{SRC}_topic_distance.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[{SRC}] wrote {len(rows)} topic similarities -> {path}")
    return path


def local(raw: str) -> None:
    """Build the source, its correlations, similarities and nine-topic matrix."""
    from analysis.correlation.survey_correlation import process_survey
    from analysis.dimension import dimension_pipeline as dp
    from common.paths import CORRELATION_DIR

    print("China rebuild check:", validate_china(raw))
    path = build(raw)
    process_survey(path, CORRELATION_DIR / f"network_analysis_{SRC}.csv")
    similarity()
    out = dp.csr_folder(SRC)
    out.mkdir(parents=True, exist_ok=True)
    dp.convert_and_save_survey_data(str(path), out, WVS_US_TOPICS)


if __name__ == "__main__":
    import fire

    fire.Fire({"build": build, "validate_china": validate_china, "similarity": similarity, "local": local})
