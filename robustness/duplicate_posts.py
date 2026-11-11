"""Duplicate posts across topics (two-country SI, smaller item of SPEC check 2).

ROBUSTNESS ONLY. Runs on the clusters that hold the post-level predictions
(Twitter/X: Adroit; Weibo: PKU). Keyword retrieval can match one post to two
topics, so that post enters both users' topic scores. For every pair of
analysed topics: the number of relevant posts shared by the two topics, and
the pooled correlation of users' scores recomputed with the shared posts
removed from both topics (expected candidates on Weibo: economy-employment,
marriage-fertility).
    -> outputs/reports/<date>_duplicate_posts/<platform>_duplicates.csv

    python -m robustness.duplicate_posts --platform twitter   # Adroit
    python -m robustness.duplicate_posts --platform weibo     # PKU
"""

from __future__ import annotations

import glob
import itertools
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

YEARS = range(2016, 2024)
TWITTER_TOPICS = ["abo", "gun", "clc", "sxo", "vac", "dpp", "soc", "minwage", "ubi"]


# -- reading posts (cluster) ---------------------------------------------------

def twitter_posts(topic: str, keep_ids: Optional[set]) -> Iterator[pd.DataFrame]:
    """Relevant posts of one Twitter/X topic: columns post_id, user_id, opinion."""
    from cleaning.settings import TWITTER_DIR, folder_map

    for f in sorted(glob.glob(os.path.join(TWITTER_DIR, f"{folder_map[topic]}-opinion", "*-text.pickle4"))):
        if not os.path.basename(f)[:4].isdigit() or int(os.path.basename(f)[:4]) not in YEARS:
            continue
        d = pd.read_pickle(f)
        d = d[d["relevance"] == 1]
        d = pd.DataFrame({"post_id": d.index.astype("int64"), "user_id": d["author_id"].astype("int64"),
                          "opinion": d["opinion"].astype(float)})
        if keep_ids is not None:
            d = d[d["user_id"].isin(keep_ids)]
        yield d


def weibo_posts(topic: int, keep_ids=None) -> Iterator[pd.DataFrame]:
    """Relevant posts of one Weibo topic: columns post_id, user_id, opinion."""
    import pyarrow.parquet as pq
    from cleaning.settings import WEIBO_TOPIC_KEYWORD_DIR

    for f in sorted(glob.glob(os.path.join(WEIBO_TOPIC_KEYWORD_DIR, str(topic), "*.parquet"))):
        if not os.path.basename(f)[:4].isdigit() or int(os.path.basename(f)[:4]) not in YEARS:
            continue
        cols = pq.ParquetFile(f).schema.names
        if "opinion" not in cols:
            continue
        need = ["weibo_id", "user_id", "opinion"] + (["relevance"] if "relevance" in cols else [])
        d = pd.read_parquet(f, engine="fastparquet", columns=need).dropna(subset=["opinion"])
        if "relevance" in d:
            d = d[d["relevance"] == 1]
        yield pd.DataFrame({"post_id": d["weibo_id"].astype(str), "user_id": d["user_id"].astype(str),
                            "opinion": d["opinion"].astype(float)})


def topics_and_reader(platform: str):
    from common.topics import WEIBO_TOPICS

    if platform == "twitter":
        from cleaning.settings import BASE_DIR
        with open(os.path.join(BASE_DIR, "us_user_ids.json")) as fh:
            keep = {int(x) for x in json.load(fh)}
        return TWITTER_TOPICS, lambda t: twitter_posts(t, keep)
    if platform == "weibo":
        return [int(t) for t in WEIBO_TOPICS], lambda t: weibo_posts(t)
    raise ValueError(platform)


def out_dir() -> Path:
    from common.paths import report_dir
    return report_dir("duplicate_posts")


# -- duplicate posts across topics (cluster) -----------------------------------

def pooled_corr(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[float, int]:
    """Pearson r of users' pooled scores on two topics (users observed on both)."""
    sa, sb = a.groupby("user_id")["opinion"].mean(), b.groupby("user_id")["opinion"].mean()
    j = pd.concat([sa, sb], axis=1, join="inner")
    return (float(np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1]) if len(j) > 2 else float("nan")), len(j)


def duplicates(platform: str) -> Path:
    topics, read = topics_and_reader(platform)
    posts = {t: pd.concat(list(read(t)), ignore_index=True) for t in topics}
    ids = {t: set(p["post_id"]) for t, p in posts.items()}
    rows = []
    for a, b in itertools.combinations(topics, 2):
        shared = ids[a] & ids[b]
        r_all, n_all = pooled_corr(posts[a], posts[b])
        if shared:
            r_nd, n_nd = pooled_corr(posts[a][~posts[a]["post_id"].isin(shared)],
                                     posts[b][~posts[b]["post_id"].isin(shared)])
        else:
            r_nd, n_nd = r_all, n_all
        rows.append({"topic1": a, "topic2": b, "shared_posts": len(shared),
                     "share_of_topic1": len(shared) / max(len(ids[a]), 1), "share_of_topic2": len(shared) / max(len(ids[b]), 1),
                     "r_all": r_all, "n_all": n_all, "r_no_shared": r_nd, "n_no_shared": n_nd,
                     "change_abs_r": abs(r_nd) - abs(r_all)})
        print(rows[-1], flush=True)
    path = out_dir() / f"{platform}_duplicates.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(path)
    return path


if __name__ == "__main__":
    import fire

    fire.Fire(duplicates)
