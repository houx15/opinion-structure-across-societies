"""Measurement error in platform stance (two-country SI check 2a) and duplicate posts.

ROBUSTNESS ONLY. Two steps on the clusters that hold the post-level
predictions (Twitter/X: Adroit; Weibo: PKU), one step here.

1. ``reliability`` (cluster). For every analysed topic, the relevant posts of
   each user pooled over 2016-2023 (Twitter/X: users in us_user_ids.json).
   Users with at least two posts: shuffle their posts (seed 0), split them into
   two halves, average each half, correlate the half-scores across users and
   apply Spearman-Brown, rel = 2 r / (1 + r). This is the reliability of a
   score built from all of a user's posts for users with two or more posts;
   users with one post (no split possible) are counted and reported, and
   their scores are noisier. Inconsistency across a user's posts is captured;
   a systematic bias of the model is not.
       -> outputs/reports/<date>_platform_reliability/<platform>_reliability.csv

2. ``duplicates`` (cluster). Posts relevant to two topics (same post id in two
   topic folders). Per topic pair: shared posts, and the pooled correlation
   recomputed with the shared posts removed from both topics.
       -> outputs/reports/<date>_platform_reliability/<platform>_duplicates.csv

3. ``correct`` (here). Corrects the 36 platform correlations of the main
   analysis for attenuation, r / sqrt(rel_i rel_j), capped at 1, and recomputes
   each platform's mean |r|, the gap and the U.S./China ratio with the topic-
   pair bootstrap; writes a stats file. Then rerun robustness/survey_noise.py
   with the measured reliabilities (median and lowest).

    python -m robustness.platform_reliability reliability --platform twitter   # Adroit
    python -m robustness.platform_reliability reliability --platform weibo     # PKU
    python -m robustness.platform_reliability duplicates --platform twitter
    python -m robustness.platform_reliability duplicates --platform weibo
    python -m robustness.platform_reliability correct --twitter <csv> --weibo <csv>
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
    return report_dir("platform_reliability")


# -- 1. split-half reliability (cluster) ------------------------------------------

def split_half(posts: pd.DataFrame, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    posts = posts.assign(u=rng.random(len(posts))).sort_values(["user_id", "u"])
    posts["k"] = posts.groupby("user_id").cumcount()
    n = posts.groupby("user_id")["opinion"].transform("size")
    multi = posts[n >= 2]
    halves = multi.assign(h=(multi["k"] % 2)).groupby(["user_id", "h"])["opinion"].mean().unstack()
    r = float(np.corrcoef(halves[0], halves[1])[0, 1])
    counts = posts.groupby("user_id").size()
    return {"users": int(counts.size), "users_2plus": int((counts >= 2).sum()),
            "share_single_post": float((counts == 1).mean()), "posts": int(len(posts)),
            "split_half_r": r, "reliability": 2 * r / (1 + r)}


def reliability(platform: str) -> Path:
    topics, read = topics_and_reader(platform)
    rows = []
    for t in topics:
        posts = pd.concat(list(read(t)), ignore_index=True)
        rows.append({"topic": t, **split_half(posts)})
        print(rows[-1], flush=True)
    path = out_dir() / f"{platform}_reliability.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(path)
    return path


# -- 2. duplicate posts across topics (cluster) -----------------------------------

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


# -- 3. attenuation-corrected platform correlations (here) -------------------------

def correct(twitter: str, weibo: str, n_boot: int = 2000) -> Path:
    from common.paths import STATS_DIR, date_prefix
    from plotting import figures as v4

    rel = {"US": pd.read_csv(twitter), "CN": pd.read_csv(weibo)}
    src = {"US": "twitter", "CN": "weibo"}
    rng = np.random.default_rng(0)
    res, L = {}, []
    for c in ("US", "CN"):
        r_t = dict(zip(rel[c]["topic"].astype(str), rel[c]["reliability"]))
        df = v4.load_pairwise_correlation(src[c])
        obs = df["correlation"].to_numpy(float)
        denom = np.sqrt([r_t[str(a)] * r_t[str(b)] for a, b in zip(df["tid1"], df["tid2"])])
        cor = np.minimum(obs / denom, 1.0)
        boot = cor[rng.integers(0, len(cor), (n_boot, len(cor)))].mean(axis=1)
        res[c] = {"obs": obs.mean(), "cor": cor.mean(), "boot": boot, "capped": int((obs / denom >= 1).sum()),
                  "rel": rel[c]}
    q = lambda a: f"[{np.percentile(a, 2.5):.3f}, {np.percentile(a, 97.5):.3f}]"
    gap, ratio = res["US"]["boot"] - res["CN"]["boot"], res["US"]["boot"] / res["CN"]["boot"]
    L += ["Paper statistics report  (task: platform_reliability_2c)", "Generated by robustness/platform_reliability.py.", "",
          f"SOURCE: {twitter}; {weibo} (split-half reliabilities, cluster step); data/correlation/network_analysis_<src>.csv.",
          "CALCULATION: module docstring; corrected r = |r| / sqrt(rel_i rel_j), capped at 1; mean over 36 pairs;",
          "  percentile bootstrap over topic pairs (2,000 draws, each platform independently, numpy seed 0).", ""]
    for c in ("US", "CN"):
        t = res[c]["rel"]
        L.append(f"{c} ({src[c]}): reliability per topic " + ", ".join(f"{a} {b:.3f}" for a, b in zip(t['topic'], t['reliability']))
                 + f"; median {t['reliability'].median():.3f}; single-post users {t['share_single_post'].mean():.1%} (mean over topics)")
    L += ["",
          f"Mean |r| observed: US {res['US']['obs']:.3f}, CN {res['CN']['obs']:.3f}; corrected: US {res['US']['cor']:.3f} "
          f"{q(res['US']['boot'])}, CN {res['CN']['cor']:.3f} {q(res['CN']['boot'])}; capped pairs US {res['US']['capped']}, CN {res['CN']['capped']}",
          f"Corrected gap (US - CN) = {res['US']['cor'] - res['CN']['cor']:.3f} {q(gap)}; ratio = {res['US']['cor'] / res['CN']['cor']:.3f} {q(ratio)}",
          "Compare: survey gap 0.270, ratio 4.713 (stats Part 4b / Part 3)."]
    path = STATS_DIR / f"{date_prefix()}_platform_reliability_2c_results_stats.txt"
    path.write_text("\n".join(L) + "\n")
    print(path)
    return path


if __name__ == "__main__":
    import fire

    fire.Fire({"reliability": reliability, "duplicates": duplicates, "correct": correct})
