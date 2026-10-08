"""Descriptive statistics of the datasets, the LLM labelling and the BERT models.

    python -m cleaning.descriptive_stats all                       # everything available
    python -m cleaning.descriptive_stats datasets                  # from data/ only
    python -m cleaning.descriptive_stats labelling --merged_dir PATH --platform weibo
    python -m cleaning.descriptive_stats bert --log_dir PATH

Sections
* datasets: respondents per survey and topic (data/opinions/individual_opinion_*),
  users per social-media source and year (data/dimension/results/*-loadings.json,
  plus per-topic coverage when the csr matrices are present), and the number of
  respondents/users behind each topic pair (data/correlation/network_analysis_*).
* labelling: per topic, the LLM-labelled sample (10,000 posts per topic): share
  of posts the models agree on (agreement_count of 3, 2, <2), share relevant,
  label distribution of the BERT training set (agreement_count >= 2) and
  pairwise agreement / Cohen's kappa between models.
  Input: <topic>_merged.parquet files (columns <model>_opinion, agreement_count,
  agreement_value; -99 = irrelevant).
* bert: evaluation metrics parsed from the fine-tuning logs
  (<log_dir>/<topic>/run-<i><task>/log.txt, lines "metric: value").

Writes outputs/stats/<date>_descriptive_stats.txt and the tables to
outputs/reports/<date>_descriptive_stats/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import fire
import numpy as np
import pandas as pd

from common.paths import (
    CORRELATION_DIR,
    CSR_DIR,
    DIMENSION_RESULTS_DIR,
    OPINION_DIR,
    STATS_DIR,
    date_prefix,
    display,
    report_dir,
)
from config import cfg

SURVEYS = [
    "anes", "anes_media", "wvs", "wvs_media", "evs", "evs_media",
    "evs_resample2", "evs_media_resample2", "en_evs", "en_evs_media", "eu_nen_evs",
]
SOCIAL_STEMS = {  # correlation-file name -> dimension stem
    "twitter": "twitter-us", "eutwitter": "twitter-eu", "entwitter": "twitter-en",
    "eu_nentwitter": "twitter-eu_nen", "weibo": "weibo",
}
MAIN_SOURCES = ["anes", "wvs", "evs_resample2", "twitter", "weibo", "eutwitter"]


def _table(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda x: f"{x:,.3f}")


# -- datasets -------------------------------------------------------------------


def survey_table() -> pd.DataFrame:
    rows = []
    for src in SURVEYS:
        path = OPINION_DIR / f"individual_opinion_{src}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        topics = [c for c in df.columns if c not in ("respondent_id", "year")]
        for t in topics:
            x = pd.to_numeric(df[t], errors="coerce")
            rows.append({
                "source": src, "topic": t, "respondents": len(df),
                "answered": int(x.notna().sum()), "missing_share": float(x.isna().mean()),
                "mean": float(x.mean()), "sd": float(x.std()),
                "min": float(x.min()), "max": float(x.max()),
            })
    return pd.DataFrame(rows)


def social_year_table() -> pd.DataFrame:
    rows = []
    for src, stem in SOCIAL_STEMS.items():
        for f in sorted(DIMENSION_RESULTS_DIR.glob(f"{stem}-*.csr-loadings.json")):
            year = f.name[len(stem) + 1:].split(".")[0]
            rows.append({"source": src, "year": year, "users": json.load(open(f))["N"]})
    return pd.DataFrame(rows)


def social_coverage_table() -> pd.DataFrame:
    """Per source/year/topic: users with an opinion, and users with >= 2 topics."""
    from scipy import sparse

    rows = []
    for src, stem in SOCIAL_STEMS.items():
        for f in sorted((CSR_DIR / stem).glob("*.csr.npz")):
            X = sparse.load_npz(f).tocsr()
            per_user = np.diff(X.indptr)
            per_topic = np.bincount(X.indices, minlength=X.shape[1])
            rows.append({
                "source": src, "year": f.name.split(".")[0], "users": X.shape[0],
                "users_2plus_topics": int((per_user >= 2).sum()),
                "mean_topics_per_user": float(per_user.mean()),
                "observed_share": float(X.nnz / (X.shape[0] * X.shape[1])),
                **{f"topic_{j}": int(n) for j, n in enumerate(per_topic)},
            })
    return pd.DataFrame(rows)


def pair_size_table() -> pd.DataFrame:
    rows = []
    for src in SURVEYS + list(SOCIAL_STEMS):
        path = CORRELATION_DIR / f"network_analysis_{src}.csv"
        if not path.exists():
            continue
        from plotting.figures import load_pairwise_correlation

        n = load_pairwise_correlation(src)["intersection"]
        rows.append({"source": src, "pairs": len(n), "min_n": int(n.min()),
                     "median_n": float(n.median()), "max_n": int(n.max())})
    return pd.DataFrame(rows)


# -- LLM labelling -------------------------------------------------------------------


def labelling_table(merged_dir: str, platform: str = "weibo") -> pd.DataFrame:
    rows = []
    for f in sorted(Path(merged_dir).glob("*_merged.parquet")):
        topic = f.name[: -len("_merged.parquet")]
        df = pd.read_parquet(f)
        models = [c for c in df.columns if c.endswith("_opinion")]
        agree = df["agreement_count"]
        train = df[agree >= 2]
        row = {
            "platform": platform, "topic": topic, "labelled": len(df),
            "models": "+".join(m[: -len("_opinion")] for m in models),
            "agree_all": float((agree == len(models)).mean()),
            "agree_2plus": float((agree >= 2).mean()),
            "training_rows": len(train),
            "relevant_share": float((train["agreement_value"] != -99).mean()) if len(train) else np.nan,
        }
        for v in (-2, -1, 0, 1, 2):
            row[f"label_{v}"] = int((train["agreement_value"] == v).sum())
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                pa, pb = df[a].fillna(-99), df[b].fillna(-99)
                key = f"{a[:-8]}_{b[:-8]}"
                row[f"agree_{key}"] = float((pa == pb).mean())
                row[f"kappa_{key}"] = _cohen_kappa(pa.to_numpy(), pb.to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)


def _cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    cats = np.union1d(a, b)
    po = float((a == b).mean())
    pe = float(sum((a == c).mean() * (b == c).mean() for c in cats))
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


# -- BERT ----------------------------------------------------------------------------

_METRIC = re.compile(r"^\s*([A-Za-z_]+):\s*([-+0-9.eE]+|nan)\s*$")


def bert_table(log_dir: str) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(log_dir).glob("*/run-*/log.txt")):
        run = f.parent.name                        # run-0regression / run-0-binary
        m = re.match(r"run-(\d+)-?(\w+)", run)
        row = {"topic": f.parent.parent.name, "run": int(m.group(1)) if m else run,
               "task": m.group(2) if m else ""}
        for line in f.read_text(errors="replace").splitlines():
            hit = _METRIC.match(line)
            if hit:
                row[hit.group(1)] = float(hit.group(2))   # last value wins (final evaluation)
        rows.append(row)
    return pd.DataFrame(rows)


# -- driver ---------------------------------------------------------------------------


def _write(sections: Dict[str, pd.DataFrame], notes: List[str]) -> Path:
    out = report_dir("descriptive_stats")
    lines = [f"Descriptive statistics ({date_prefix()})", ""] + notes
    for name, df in sections.items():
        if df is None or df.empty:
            lines += ["", f"== {name}: no input found"]
            continue
        df.to_csv(out / f"{name}.csv", index=False)
        lines += ["", f"== {name}  ({display(out / (name + '.csv'))})", _table(df)]
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    path = STATS_DIR / f"{date_prefix()}_descriptive_stats.txt"
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {path}")
    return path


def datasets() -> Path:
    surveys = survey_table()
    summary = (surveys.groupby("source", sort=False)
               .agg(respondents=("respondents", "first"), mean_missing_share=("missing_share", "mean"))
               .reset_index() if not surveys.empty else surveys)
    return _write({
        "survey_respondents": summary,
        "survey_topics": surveys,
        "social_users_by_year": social_year_table(),
        "social_topic_coverage": social_coverage_table(),
        "pair_sample_sizes": pair_size_table(),
    }, [f"Main sources: {', '.join(MAIN_SOURCES)}."])


def labelling(merged_dir: str, platform: str = "weibo") -> Path:
    return _write({f"llm_labelling_{platform}": labelling_table(merged_dir, platform)}, [])


def bert(log_dir: Optional[str] = None) -> Path:
    return _write({"bert_evaluation": bert_table(log_dir or cfg.BERT_LOG_DIR)}, [])


def all(weibo_merged_dir: Optional[str] = None, twitter_merged_dir: Optional[str] = None,
        bert_log_dir: Optional[str] = None) -> Path:
    """Every section whose input exists (directories default to config)."""
    surveys = survey_table()
    sections = {
        "survey_respondents": (surveys.groupby("source", sort=False)
                               .agg(respondents=("respondents", "first"),
                                    mean_missing_share=("missing_share", "mean")).reset_index()
                               if not surveys.empty else surveys),
        "survey_topics": surveys,
        "social_users_by_year": social_year_table(),
        "social_topic_coverage": social_coverage_table(),
        "pair_sample_sizes": pair_size_table(),
    }
    for platform, d in (("weibo", weibo_merged_dir or cfg.BERT_DATASET_DIR),
                        ("twitter", twitter_merged_dir)):
        if d and Path(d).is_dir():
            sections[f"llm_labelling_{platform}"] = labelling_table(d, platform)
    log_dir = bert_log_dir or cfg.BERT_LOG_DIR
    if log_dir and Path(log_dir).is_dir():
        sections["bert_evaluation"] = bert_table(log_dir)
    return _write(sections, [f"Main sources: {', '.join(MAIN_SOURCES)}."])


if __name__ == "__main__":
    fire.Fire({"all": all, "datasets": datasets, "labelling": labelling, "bert": bert})
