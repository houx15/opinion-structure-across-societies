"""Export user-level LGBT x Environment opinion scores for the Fig 2 case-study panel (plotting/figures.py).

This script is meant to run on the machine that has the upstream user-level
opinion data (e.g. lustre). It writes a small, focused parquet per source so
the plotting environment only needs the LGBT / Environment slice instead of
the full multi-topic dataset.

Output schema (one file per source):
    Path:    data/opinions/user_opinion_<src>_lgbt_env.parquet
    Columns: user_id (str|int), LGBT (float), Environment (float)
    Rows:    one row per user, restricted to users who have at least one
             non-null opinion on either topic. Values are the user's average
             opinion across years (range roughly [-2, 2]).

Sources (matching plotting.figures conventions):
    twitter        -> Twitter (US)
    weibo          -> Weibo  (CN)
    eutwitter      -> Twitter (EU, full EU)
    entwitter      -> Twitter (EN-only)
    eu_nentwitter  -> Twitter (EU non-English-native: EU minus EN)

Upstream layouts the script knows how to read:

    weibo (see analysis/correlation/weibo_correlation.py):
        <weibo_root>/<topic_id>/avg_opinion.parquet
        Each parquet has user_id as index, year columns ("2016".."2023"),
        and an "average" column. weibo topic 10 = LGBT, 13 = Environment.

    twitter (see analysis/correlation/twitter_correlation.py):
        <twitter_root>/merged-<topic>.parquet       (flat — one file per topic)
        Index is numeric user_id, columns are year strings. Region (US/EU/EN)
        is selected at compute time by filtering against a JSON allowlist:
            <LOCATION_DIR>/<mode>_user_ids.json
        Topic codes: sxo = LGBT, clc = Climate change. Mode in {us, eu, en}.

Usage:
    # weibo  (CN)
    python -m cleaning.twitter.opinion.export_user_opinion_data weibo --weibo_root <WEIBO_OPINION_DIR>

    # twitter (US)
    python -m cleaning.twitter.opinion.export_user_opinion_data twitter \
        --twitter_root <TWITTER_OPINION_DIR> \
        --user_ids_file <LOCATION_DIR>/us_user_ids.json

    # twitter (EU)
    python -m cleaning.twitter.opinion.export_user_opinion_data eutwitter \
        --twitter_root <TWITTER_OPINION_DIR> \
        --user_ids_file <LOCATION_DIR>/eu_user_ids.json

    # twitter (EN only)
    python -m cleaning.twitter.opinion.export_user_opinion_data entwitter \
        --twitter_root <TWITTER_OPINION_DIR> \
        --user_ids_file <LOCATION_DIR>/en_user_ids.json

    # twitter (EU non-English-native)
    python -m cleaning.twitter.opinion.export_user_opinion_data eu_nentwitter \
        --twitter_root <TWITTER_OPINION_DIR> \
        --user_ids_file <LOCATION_DIR>/eu_nen_user_ids.json

    # verify an export by comparing |r| with data/network_analysis_<src>.csv
    python -m cleaning.twitter.opinion.export_user_opinion_data verify --src weibo
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, Optional, Set

import fire
import numpy as np
import pandas as pd
from cleaning.settings import *


WEIBO_LGBT_TOPIC_ID = 10
WEIBO_ENV_TOPIC_ID = 13
TWITTER_LGBT_TOPIC = "sxo"
TWITTER_ENV_TOPIC = "clc"

from common.paths import CORRELATION_DIR, OPINION_DIR

DEFAULT_OUTPUT_DIR = OPINION_DIR
DEFAULT_AVERAGE_COL_CANDIDATES = ("average", "merged", "all")

# Network-analysis reference CSV columns for each source's LGBT-x-env pair.
_VERIFY_TARGET_TIDS: Dict[str, FrozenSet] = {
    "weibo": frozenset({10, 13}),
    "twitter": frozenset({"sxo", "clc"}),
    "eutwitter": frozenset({"sxo", "clc"}),
    "entwitter": frozenset({"sxo", "clc"}),
    "eu_nentwitter": frozenset({"sxo", "clc"}),
}


def _pick_average_column(df: pd.DataFrame) -> Optional[str]:
    """Pick the column that already aggregates across years (e.g. 'average')."""
    for candidate in DEFAULT_AVERAGE_COL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def _per_user_mean_opinion(df: pd.DataFrame) -> pd.Series:
    """Return a Series indexed by user_id holding the user's mean opinion.

    If the upstream parquet already has an "average"-style column, prefer it.
    Otherwise average across the numeric year columns, clipped to [-2, 2] to
    match the rest of the pipeline (see analysis/correlation/weibo_correlation.py).
    """
    avg_col = _pick_average_column(df)
    if avg_col is not None:
        series = df[avg_col]
    else:
        year_cols = [c for c in df.columns if str(c).isdigit()]
        if not year_cols:
            raise ValueError(
                f"No average column and no year columns found; got {list(df.columns)[:10]}..."
            )
        series = df[year_cols].mean(axis=1)
    series = series.astype(float).clip(lower=-2.0, upper=2.0)
    series.name = "opinion"
    return series


def _load_weibo_topic(weibo_root: Path, topic_id: int) -> pd.Series:
    """Load weibo per-user mean opinion for one topic_id."""
    path = weibo_root / str(topic_id) / "avg_opinion.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Expected weibo parquet at {path}. Adjust --weibo_root if this is wrong."
        )
    df = pd.read_parquet(path)
    series = _per_user_mean_opinion(df)
    series.index.name = "user_id"
    return series


def _load_user_id_allowlist(user_ids_file: Optional[str]) -> Optional[Set[int]]:
    """Load a JSON list of user ids and coerce to a set of int64 — mirrors
    analysis/correlation/twitter_correlation.py. Returns None when no allowlist is provided
    (i.e. the export is unfiltered)."""
    if not user_ids_file:
        return None
    with open(user_ids_file) as f:
        raw = json.load(f)
    return set(int(x) for x in raw)


def _load_twitter_topic(
    twitter_root: Path,
    topic: str,
    keep_ids: Optional[Set[int]] = None,
) -> pd.Series:
    """Load twitter per-user mean opinion for one topic code (e.g. 'sxo', 'clc').

    Matches analysis/correlation/twitter_correlation.py: each topic lives in a single flat
    parquet at <twitter_root>/merged-<topic>.parquet, indexed by numeric user
    id with year-string columns. Region (US/EU/EN) is selected at runtime via
    a JSON allowlist passed through ``keep_ids``.

    Falls back to the older mode-prefixed layout for environments that already
    split twitter data per region on disk.
    """
    candidates = [
        twitter_root / f"merged-{topic}.parquet",
        twitter_root / "us" / f"merged-{topic}.parquet",
        twitter_root / "eu" / f"merged-{topic}.parquet",
        twitter_root / "en" / f"merged-{topic}.parquet",
        twitter_root / topic / "avg_opinion.parquet",
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        # Coerce user-id index to int64 to match the allowlist; mirrors
        # analysis/correlation/twitter_correlation.py: pd.to_numeric(df.index, errors='raise').
        try:
            df.index = pd.to_numeric(df.index, errors="raise").astype("int64")
        except (ValueError, TypeError):
            pass  # string ids — keep as-is and skip filtering
        if keep_ids is not None and df.index.dtype.kind in {"i", "u"}:
            df = df.loc[df.index.intersection(list(keep_ids))]
        series = _per_user_mean_opinion(df)
        series.index.name = "user_id"
        return series
    raise FileNotFoundError(
        "No twitter parquet found. Tried:\n  - "
        + "\n  - ".join(str(p) for p in candidates)
        + "\nAdjust --twitter_root if the layout differs."
    )


def _assemble_pair(lgbt_series: pd.Series, env_series: pd.Series) -> pd.DataFrame:
    """Merge two per-user opinion series into a user x {LGBT, Environment} table."""
    df = pd.concat(
        [
            lgbt_series.rename("LGBT"),
            env_series.rename("Environment"),
        ],
        axis=1,
        join="outer",
    )
    # Drop users that have neither value.
    df = df.dropna(how="all")
    df.index.name = "user_id"
    df = df.reset_index()
    return df


def _write_output(df: pd.DataFrame, src: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"user_opinion_{src}_lgbt_env.parquet"
    df.to_parquet(output_path, index=False)
    return output_path


def _report(df: pd.DataFrame, src: str, path: Path) -> None:
    n_total = len(df)
    n_both = int(df[["LGBT", "Environment"]].notna().all(axis=1).sum())
    lgbt_n = int(df["LGBT"].notna().sum())
    env_n = int(df["Environment"].notna().sum())
    print(
        f"[{src}] wrote {path}  rows={n_total:,}  "
        f"both_non_null={n_both:,}  lgbt_only={lgbt_n - n_both:,}  env_only={env_n - n_both:,}"
    )


# -- public entrypoints --------------------------------------------------------


def weibo(weibo_root: str, output_dir: str = str(DEFAULT_OUTPUT_DIR)) -> Path:
    """Export user-level LGBT and Environment opinions for weibo."""
    root = Path(weibo_root)
    lgbt = _load_weibo_topic(root, WEIBO_LGBT_TOPIC_ID)
    env = _load_weibo_topic(root, WEIBO_ENV_TOPIC_ID)
    df = _assemble_pair(lgbt, env)
    path = _write_output(df, "weibo", Path(output_dir))
    _report(df, "weibo", path)
    return path


def _export_twitter_variant(
    src: str,
    twitter_root: str,
    user_ids_file: Optional[str],
    output_dir: str,
) -> Path:
    """Shared body for twitter / eutwitter / entwitter exports."""
    root = Path(twitter_root)
    keep_ids = _load_user_id_allowlist(user_ids_file)
    lgbt = _load_twitter_topic(root, TWITTER_LGBT_TOPIC, keep_ids=keep_ids)
    env = _load_twitter_topic(root, TWITTER_ENV_TOPIC, keep_ids=keep_ids)
    df = _assemble_pair(lgbt, env)
    path = _write_output(df, src, Path(output_dir))
    if keep_ids is not None:
        print(f"[{src}] user-id allowlist applied: {len(keep_ids):,} ids from {user_ids_file}")
    _report(df, src, path)
    return path


def twitter(
    twitter_root: str,
    user_ids_file: Optional[str] = None,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
) -> Path:
    """Export user-level LGBT and Climate opinions for Twitter (US).

    Pass --user_ids_file pointing at us_user_ids.json to restrict to US users
    (matches analysis/correlation/twitter_correlation.py with mode='us').
    """
    return _export_twitter_variant("twitter", twitter_root, user_ids_file, output_dir)


def eutwitter(
    twitter_root: str,
    user_ids_file: Optional[str] = None,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
) -> Path:
    """Export user-level LGBT and Climate opinions for Twitter (EU).

    Pass --user_ids_file pointing at eu_user_ids.json.
    """
    return _export_twitter_variant("eutwitter", twitter_root, user_ids_file, output_dir)


def entwitter(
    twitter_root: str,
    user_ids_file: Optional[str] = None,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
) -> Path:
    """Export user-level LGBT and Climate opinions for Twitter (EN-only).

    Pass --user_ids_file pointing at en_user_ids.json.
    """
    return _export_twitter_variant("entwitter", twitter_root, user_ids_file, output_dir)


def eu_nentwitter(
    twitter_root: str,
    user_ids_file: Optional[str] = None,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
) -> Path:
    """Export user-level LGBT and Climate opinions for Twitter (EU non-English-native).

    Pass --user_ids_file pointing at eu_nen_user_ids.json — this is the
    EU-minus-EN allowlist used by the 4-region eu_nen robustness check
    (plotting.figures task ``robust_6_eu_nen``).
    """
    return _export_twitter_variant("eu_nentwitter", twitter_root, user_ids_file, output_dir)


def all(
    weibo_root: Optional[str] = None,
    twitter_root: Optional[str] = None,
    us_user_ids_file: Optional[str] = None,
    eu_user_ids_file: Optional[str] = None,
    en_user_ids_file: Optional[str] = None,
    eu_nen_user_ids_file: Optional[str] = None,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
) -> Dict[str, str]:
    """Export every source whose root + allowlist are provided. Missing inputs are skipped.

    Pass each region's user-id JSON file separately because the twitter source
    folder is the same for all regions; only the allowlist changes.
    """
    written: Dict[str, str] = {}
    if weibo_root:
        written["weibo"] = str(weibo(weibo_root, output_dir=output_dir))
    if twitter_root:
        targets = [
            ("twitter", us_user_ids_file),
            ("eutwitter", eu_user_ids_file),
            ("entwitter", en_user_ids_file),
            ("eu_nentwitter", eu_nen_user_ids_file),
        ]
        for src, ids_file in targets:
            try:
                written[src] = str(
                    _export_twitter_variant(src, twitter_root, ids_file, output_dir)
                )
            except FileNotFoundError as exc:
                print(f"[{src}] skipped: {exc}")
    if not written:
        print(
            "[WARN] nothing exported. Provide --weibo_root and/or --twitter_root."
        )
    return written


# -- verification ------------------------------------------------------------


def _verify_target_tids(src: str) -> FrozenSet:
    if src in _VERIFY_TARGET_TIDS:
        return _VERIFY_TARGET_TIDS[src]
    raise ValueError(f"verify not supported for src={src!r}")


def _reference_correlation(src: str, network_csv: Path) -> Dict[str, float]:
    """Look up the LGBT x Environment correlation for ``src`` in the precomputed CSV."""
    target = _verify_target_tids(src)
    df = pd.read_csv(network_csv)
    if "tid1" not in df.columns:
        df["tid1"] = df["Topic1"]
        df["tid2"] = df["Topic2"]

    def _matches(row) -> bool:
        return frozenset({row["tid1"], row["tid2"]}) == target

    matched = df[(df["year"].astype(str) == "average") & df.apply(_matches, axis=1)]
    if matched.empty:
        raise ValueError(
            f"No 'average' row in {network_csv} for target topic pair {target}"
        )
    row = matched.iloc[0]
    return {
        "r_ref": float(row["Correlation (Pearson)"]),
        "n_ref": int(row["Intersection"]),
    }


def verify(
    src: str,
    data_dir: str = str(DEFAULT_OUTPUT_DIR),
    csv_dir: str = str(CORRELATION_DIR),
    tolerance: float = 0.01,
) -> Dict[str, float]:
    """Recompute |r| from the exported parquet and compare with network_analysis_<src>.csv.

    Pass / fail is reported on stdout and returned in the result dict. Use a
    looser ``tolerance`` if the export uses a different aggregation than the
    reference pipeline.
    """
    from scipy.stats import pearsonr

    parquet_path = Path(data_dir) / f"user_opinion_{src}_lgbt_env.parquet"
    csv_path = Path(csv_dir) / f"network_analysis_{src}.csv"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} not found. Run export first."
        )
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Cannot verify without reference."
        )

    df = pd.read_parquet(parquet_path).dropna(subset=["LGBT", "Environment"])
    if len(df) < 2:
        raise ValueError(
            f"Exported parquet has only {len(df)} rows with both topics non-null"
        )
    r_ours, _ = pearsonr(df["LGBT"].to_numpy(), df["Environment"].to_numpy())
    n_ours = int(len(df))

    ref = _reference_correlation(src, csv_path)
    delta_r = abs(abs(r_ours) - abs(ref["r_ref"]))
    delta_n = int(n_ours - ref["n_ref"])

    status = "OK" if delta_r <= tolerance else "DIFF"
    print(f"[{src}] |r|: ours={abs(r_ours):.4f}  ref={abs(ref['r_ref']):.4f}  Δ={delta_r:.4f}")
    print(f"[{src}] N:   ours={n_ours:,}  ref={ref['n_ref']:,}  Δ={delta_n:+,}")
    print(f"[{src}] status: {status}  (tolerance={tolerance})")

    return {
        "src": src,
        "r_ours": float(abs(r_ours)),
        "r_ref": float(abs(ref["r_ref"])),
        "n_ours": n_ours,
        "n_ref": ref["n_ref"],
        "delta_r": float(delta_r),
        "delta_n": delta_n,
        "status": status,
    }


def verify_all(
    data_dir: str = str(DEFAULT_OUTPUT_DIR),
    csv_dir: str = str(CORRELATION_DIR),
    tolerance: float = 0.01,
) -> Dict[str, Dict]:
    """Verify every exported source whose parquet exists locally."""
    results: Dict[str, Dict] = {}
    for src in ("weibo", "twitter", "eutwitter", "entwitter", "eu_nentwitter"):
        parquet = Path(data_dir) / f"user_opinion_{src}_lgbt_env.parquet"
        if not parquet.exists():
            print(f"[{src}] skipped: parquet not found")
            continue
        results[src] = verify(src, data_dir=data_dir, csv_dir=csv_dir, tolerance=tolerance)
    return results


if __name__ == "__main__":
    fire.Fire(
        {
            "weibo": weibo,
            "twitter": twitter,
            "eutwitter": eutwitter,
            "entwitter": entwitter,
            "eu_nentwitter": eu_nentwitter,
            "all": all,
            "verify": verify,
            "verify_all": verify_all,
        }
    )
