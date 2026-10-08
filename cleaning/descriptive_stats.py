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
* labelling: LLM consistency per topic on the labelled sample (10,000 posts per
  topic; same columns as llm_consistency.csv written by the merge step):
  valid labels per model, share of posts with >= 2 and with all models
  agreeing, share relevant,
  label distribution of the BERT training set (agreement_count >= 2) and
  pairwise agreement / Cohen's kappa between models.
  Input: <topic>_merged.parquet files (columns <model>_opinion, agreement_count,
  agreement_value; -99 = irrelevant).
* bert: evaluation metrics parsed from the fine-tuning logs
  (<log_dir>/<topic>/run-<i><task>/log.txt, lines "metric: value"), per run
  and as mean / sd over the repeated runs.
* location (Twitter): rule-based classification of profile location strings,
  LLM verdicts on the strings the rules left undecided, size of each regional
  user id list, users per European country (LOCATION_DIR).
* twitter_users: users with an opinion and their relevant tweets per topic and
  region, overall and per year (TWITTER_OPINION_DIR/merged-<topic>.parquet
  filtered by the regional user id lists).

Run where the inputs are: Twitter sections on the Twitter cluster, Weibo
labelling where the Weibo merged files are; ``--tag`` keeps the reports apart.

Writes outputs/stats/<date>_descriptive_stats[_<tag>].txt and the tables to
outputs/reports/<date>_descriptive_stats[_<tag>]/.
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
            **{f"n_{m[: -len('_opinion')]}": int(df[m].notna().sum()) for m in models},
            "agree_2plus": float((agree >= 2).mean()),
            "agree_all": float((agree == len(models)).mean()),
            "training_rows": len(train),
            "relevant_share": float((train["agreement_value"] != -99).mean()) if len(train) else np.nan,
        }
        for v in (-2, -1, 0, 1, 2):
            row[f"label_{v}"] = int((train["agreement_value"] == v).sum())
        row["irrelevant"] = int((train["agreement_value"] == -99).sum())
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


def bert_summary_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and sd of each metric over the repeated runs of a topic / task."""
    if runs.empty:
        return runs
    metrics = [c for c in runs.columns if c not in ("topic", "run", "task")]
    g = runs.groupby(["topic", "task"])
    out = g.size().rename("runs").to_frame()
    for m in metrics:
        out[f"{m}_mean"] = g[m].mean()
        out[f"{m}_sd"] = g[m].std()
    return out.reset_index().dropna(axis=1, how="all")


# -- Twitter location and users -----------------------------------------------------------

REGION_ID_FILES = {"us": "us_user_ids.json", "eu": "eu_user_ids.json",
                   "en": "en_user_ids.json", "eu_nen": "eu_nen_user_ids.json"}
TWITTER_TOPICS = ["abo", "gun", "clc", "sxo", "vac", "soc", "dpp", "minwage", "ubi", "swe"]


def _load_ids(location_dir: Path, mode: str) -> set:
    with open(location_dir / REGION_ID_FILES[mode]) as f:
        return set(int(x) for x in json.load(f))


def location_tables(location_dir: Optional[str] = None,
                    llm_dirs: Optional[Dict[str, str]] = None) -> Dict[str, pd.DataFrame]:
    """Location-filtering counts; llm_dirs maps step (us / eu / eu_country) to its LLM batch folder."""
    d = Path(location_dir or cfg.LOCATION_DIR)
    llm_dirs = llm_dirs or {step: str(d / f"llm_location_{step}") for step in ("us", "eu", "eu_country")}
    rows = []
    for step, f in (("us", "non_us_user_analysis.json"), ("eu", "eu_location_classified.json")):
        if (d / f).exists():
            for label, strings in json.load(open(d / f)).items():
                rows.append({"step": step, "class": label, "location_strings": len(strings)})
    classes = pd.DataFrame(rows)

    rows = []
    for step, folder in llm_dirs.items():
        path = Path(folder) / "llm_result.parquet"
        if not path.exists():
            continue
        r = pd.read_parquet(path)["result"].astype(str).str.replace(r"\.0$", "", regex=True)
        meaning = {"0": "cannot tell", "1": "inside", "2": "outside"} if step != "eu_country" else {}
        for value, n in r.value_counts().items():
            rows.append({"step": step, "verdict": meaning.get(value, value), "location_strings": int(n),
                         "share": n / len(r)})
    llm = pd.DataFrame(rows)

    users = pd.DataFrame([
        {"region": mode, "file": f, "users": len(json.load(open(d / f)))}
        for mode, f in REGION_ID_FILES.items() if (d / f).exists()
    ])
    country_csv = d / "eu_country_user_count_in_dataset.csv"
    countries = pd.read_csv(country_csv) if country_csv.exists() else pd.DataFrame()
    return {"location_string_classes": classes, "location_llm_verdicts": llm,
            "location_user_lists": users, "location_users_by_country": countries}


def twitter_user_tables(opinion_dir: Optional[str] = None,
                        location_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Users with an opinion and relevant tweets per topic x region (x year)."""
    opinion_dir = Path(opinion_dir or cfg.TWITTER_OPINION_DIR)
    location_dir = Path(location_dir or cfg.LOCATION_DIR)
    ids = {m: _load_ids(location_dir, m) for m in REGION_ID_FILES if (location_dir / REGION_ID_FILES[m]).exists()}
    total, yearly = [], []
    for topic in TWITTER_TOPICS:
        path = opinion_dir / f"merged-{topic}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df.index = pd.to_numeric(df.index, errors="coerce")
        years = sorted(c for c in df.columns if c.isdigit())
        for mode, keep in ids.items():
            sub_df = df[df.index.isin(keep)]
            tweets = sub_df[[f"{y}_count" for y in years]].sum().sum() if years else np.nan
            total.append({"topic": topic, "region": mode,
                          "users": int(sub_df["average"].notna().sum()),
                          "relevant_tweets": int(tweets), "years": f"{years[0]}-{years[-1]}" if years else ""})
            for y in years:
                yearly.append({"topic": topic, "region": mode, "year": int(y),
                               "users": int(sub_df[y].notna().sum()),
                               "relevant_tweets": int(sub_df[f"{y}_count"].sum())})
    return {"twitter_users_by_topic": pd.DataFrame(total),
            "twitter_users_by_topic_year": pd.DataFrame(yearly)}


# -- driver ---------------------------------------------------------------------------


def _write(sections: Dict[str, pd.DataFrame], notes: List[str], tag: str = "") -> Path:
    name = "descriptive_stats" + (f"_{tag}" if tag else "")
    out = report_dir(name)
    lines = [f"Descriptive statistics ({date_prefix()}{', ' + tag if tag else ''})", ""] + notes
    for name, df in sections.items():
        if df is None or df.empty:
            lines += ["", f"== {name}: no input found"]
            continue
        df.to_csv(out / f"{name}.csv", index=False)
        lines += ["", f"== {name}  ({display(out / (name + '.csv'))})", _table(df)]
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    path = STATS_DIR / f"{date_prefix()}_{name}.txt"
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


def labelling(merged_dir: str, platform: str = "weibo", tag: str = "") -> Path:
    return _write({f"llm_labelling_{platform}": labelling_table(merged_dir, platform)}, [], tag or platform)


def bert(log_dir: Optional[str] = None, tag: str = "") -> Path:
    runs = bert_table(log_dir or cfg.BERT_LOG_DIR)
    return _write({"bert_evaluation": runs, "bert_evaluation_summary": bert_summary_table(runs)}, [], tag)


def twitter(location_dir: Optional[str] = None, opinion_dir: Optional[str] = None,
            merged_dir: Optional[str] = None, bert_log_dir: Optional[str] = None,
            llm_us: Optional[str] = None, llm_eu: Optional[str] = None,
            llm_eu_country: Optional[str] = None, tag: str = "twitter") -> Path:
    """Twitter cleaning statistics: location filtering, users, LLM labelling, BERT."""
    llm_dirs = None
    if llm_us or llm_eu or llm_eu_country:
        llm_dirs = {k: v for k, v in (("us", llm_us), ("eu", llm_eu), ("eu_country", llm_eu_country)) if v}
    sections = dict(location_tables(location_dir, llm_dirs))
    sections.update(twitter_user_tables(opinion_dir, location_dir))
    merged_dir = merged_dir or cfg.BERT_DATASET_DIR
    if merged_dir and Path(merged_dir).is_dir():
        sections["llm_labelling_twitter"] = labelling_table(merged_dir, "twitter")
    log_dir = bert_log_dir or cfg.BERT_LOG_DIR
    if log_dir and Path(log_dir).is_dir():
        runs = bert_table(log_dir)
        sections["bert_evaluation"] = runs
        sections["bert_evaluation_summary"] = bert_summary_table(runs)
    return _write(sections, [], tag)


def all(weibo_merged_dir: Optional[str] = None, twitter_merged_dir: Optional[str] = None,
        bert_log_dir: Optional[str] = None, tag: str = "") -> Path:
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
        runs = bert_table(log_dir)
        sections["bert_evaluation"] = runs
        sections["bert_evaluation_summary"] = bert_summary_table(runs)
    return _write(sections, [f"Main sources: {', '.join(MAIN_SOURCES)}."], tag)


if __name__ == "__main__":
    fire.Fire({"all": all, "datasets": datasets, "labelling": labelling, "bert": bert,
               "twitter": twitter})
