"""Per-respondent survey opinions (cleaned) from the survey microdata.

Reads data/survey/<src>.dta and writes one parquet per source; plotting code
only reads the parquets, never the raw .dta.

Output schema (one file per source):
    Path:    data/opinions/individual_opinion_<src>.parquet
    Columns: respondent_id (int), year (int), <topic_1> ... <topic_9> (float)
    Rows:    one row per respondent (per year row in the original .dta).
             Topic columns are the source's restricted-topic set.

Sources mirror ``analysis/correlation/survey_correlation.py``:
    anes / anes_media          -> US (ANES)
    wvs  / wvs_media           -> CN (WVS)
    evs  / evs_media           -> EU (EVS)
    evs_resample2 / *_media    -> EU (EVS, resampled — the v3 main task)
    en_evs / en_evs_media      -> EU (EVS, England-only)
    eu_nen_evs                 -> EU (EVS, non-English-native: EU minus GB)

en_evs is filtered to country == "Great Britain" and eu_nen_evs to
country != "Great Britain", exactly as ``survey_correlation.process_survey``
does.

Usage:
    python -m cleaning.survey.prepare_survey_individual_data single --src anes
    python -m cleaning.survey.prepare_survey_individual_data all
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import fire
import pandas as pd


US_TOPICS: List[str] = [
    "Abortion", "Gun", "Climate", "LGBT", "VACC",
    "DeathPenalty", "Media", "MinimumWage", "UBI",
]
CN_TOPICS: List[str] = [
    "Corrup", "GenderEqual", "Marriage", "Childbearing", "LGBT",
    "Environment", "Econ", "Work", "Foreign",
]
EU_TOPICS: List[str] = [
    "LGBT", "Abortion", "DeathPenalty", "SocialMedia", "Egalitarian",
    "Environment", "Prostitution", "HealthCare", "UnemplyAid",
]

RESTRICTED_TOPICS: Dict[str, List[str]] = {
    "anes": US_TOPICS,
    "anes_media": US_TOPICS,
    "wvs": CN_TOPICS,
    "wvs_media": CN_TOPICS,
    "evs": EU_TOPICS,
    "evs_media": EU_TOPICS,
    "evs_resample2": EU_TOPICS,
    "evs_media_resample2": EU_TOPICS,
    "en_evs": EU_TOPICS,
    "en_evs_media": EU_TOPICS,
    "eu_nen_evs": EU_TOPICS,
}

ALL_SOURCES: List[str] = list(RESTRICTED_TOPICS.keys())

from common.paths import OPINION_DIR, SURVEY_DIR

DATA_DIR = SURVEY_DIR


def _load_and_clean(input_path: Path, src: str) -> pd.DataFrame:
    """Read a .dta, apply src-specific filtering, return a wide respondent frame."""
    df = pd.read_stata(input_path)

    if src.startswith("en_evs") or src.startswith("eu_nen_evs"):
        # Match survey_correlation.process_survey's country split: en_evs
        # keeps only Great Britain; eu_nen_evs keeps everything *but* Great
        # Britain (EU non-English-native).
        if "country" not in df.columns:
            raise ValueError(
                f"{input_path} is missing a 'country' column required for {src}"
            )
        if src.startswith("eu_nen_evs"):
            df = df[df["country"] != "Great Britain"].copy()
        else:
            df = df[df["country"] == "Great Britain"].copy()

    if "year" not in df.columns:
        df["year"] = 2020

    topics = RESTRICTED_TOPICS[src]
    missing = [t for t in topics if t not in df.columns]
    if missing:
        raise ValueError(
            f"{input_path} is missing required topic columns for {src}: {missing}"
        )

    keep_cols = ["year"] + topics
    df = df[keep_cols].copy()
    df = df.reset_index(drop=True).reset_index().rename(columns={"index": "respondent_id"})

    # Drop rows where *every* topic is missing — they cannot contribute to any pair.
    df = df.dropna(subset=topics, how="all").reset_index(drop=True)
    # Coerce year to int when possible so downstream code can do df['year'] == 2020.
    try:
        df["year"] = df["year"].astype(int)
    except (TypeError, ValueError):
        pass
    return df


def single(src: str, data_dir: str = str(DATA_DIR)) -> Path:
    """Prepare a single source's individual-opinion parquet."""
    if src not in RESTRICTED_TOPICS:
        raise ValueError(
            f"Unknown source: {src}. Known sources: {sorted(RESTRICTED_TOPICS)}"
        )
    dd = Path(data_dir)
    input_path = dd / f"{src}.dta"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input: {input_path}")
    df = _load_and_clean(input_path, src)
    OPINION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OPINION_DIR / f"individual_opinion_{src}.parquet"
    df.to_parquet(output_path, index=False)
    print(
        f"[{src}] wrote {output_path}  rows={len(df):,}  "
        f"topics={len(RESTRICTED_TOPICS[src])}  years={sorted(df['year'].unique())[:5]}..."
    )
    return output_path


def all(data_dir: str = str(DATA_DIR)) -> Dict[str, str]:
    """Prepare every source whose .dta file is present. Missing inputs are skipped."""
    written: Dict[str, str] = {}
    for src in ALL_SOURCES:
        path = Path(data_dir) / f"{src}.dta"
        if not path.exists():
            print(f"[{src}] skipped: {path} missing")
            continue
        written[src] = str(single(src, data_dir=data_dir))
    return written


if __name__ == "__main__":
    fire.Fire({"single": single, "all": all})
