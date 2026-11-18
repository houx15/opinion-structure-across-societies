"""Semantic dimensionality of each source's topic set (Fig 3B and the SI).

The counterpart of the opinion PR / eRank (Fig 3C), computed the same way:
the nine topic centroids (unit 1536-D OpenAI vectors, the ones behind the
Fig 2 similarities) are treated as nine variables observed over the
embedding coordinates, and PR and eRank are taken from the eigenvalues of
their 9 x 9 cosine matrix. Embedding coordinates average almost exactly zero
within a topic and the centroids have unit length, so this equals the 9 x 9
covariance and correlation matrices: nothing is removed across topics, as
in the opinion analysis.

A topic's centroid is the weighted mean of its keyword vectors (survey: the
``top5`` codebook keywords, equal weights; platforms: the ten highest
TF-IDF words pooled over years, weighted by frequency; see
``semantic_similarity``). The rebuilt main centroids are checked against
``data/embedding/topic_centroids``.

Uncertainty: sensitivity to the keyword choice, not a bootstrap.
Resampling keywords with replacement is biased here (repeated keywords make
centroids less alike and raise PR; the main estimate fell outside its own
interval) and many survey topics have one to five keywords. Variants:
survey keyword sets ``selected`` and ``curated``; platform top 5 and top 20
TF-IDF words.

Also reported, for the SI: the share of the first eigenvalue (the direction
all topics share) and PR / eRank after removing it (mean topic vector
subtracted; at most K - 1 dimensions). That variant describes the shared
component; it is not the measure, because it centres across topics, which
the opinion analysis never does.

Only cached keyword vectors are used (no API call). Words without a cached
vector are dropped and counted.

    python -m analysis.semantic.semantic_rank            # stats + JSON for the figures
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from common.paths import EMBEDDING_DIR, STATS_DIR, TF_IDF_DIR, date_prefix

SURVEY_REGION = {"anes": "us", "wvs": "china"}
SOURCES = ["anes", "wvs", "twitter", "weibo"]   # United States and China (two-country analysis)
SOURCE_LABEL = {"anes": "ANES (US survey)", "wvs": "WVS (CN survey)",
                "twitter": "Twitter/X (US)", "weibo": "Weibo (CN)"}
MAIN = {"survey": "top5", "platform": 10}
VARIANTS = {"survey": ["top5", "selected", "curated"], "platform": [10, 5, 20]}
CENTROID_DIR = EMBEDDING_DIR / "topic_centroids"
RESULT_JSON = EMBEDDING_DIR / "semantic_rank.json"


def participation_ratio(lam: np.ndarray) -> float:
    lam = lam[lam > 1e-12]
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def exponential_rank(lam: np.ndarray) -> float:
    lam = lam[lam > 1e-12]
    p = lam / lam.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def kind(src: str) -> str:
    return "survey" if src in SURVEY_REGION else "platform"


def topic_ids(src: str) -> List[str]:
    """The nine analysed topics, in the order of the stored centroid file."""
    return pd.read_parquet(CENTROID_DIR / f"{src}_topic_centroids.parquet")["topic_id"].astype(str).tolist()


def keyword_weights(src: str, variant) -> Dict[str, List[Tuple[str, float]]]:
    """{topic: [(keyword, weight)]} for one keyword variant."""
    if kind(src) == "survey":
        from analysis.semantic.semantic_similarity import survey_topic_keyword_counts

        table = survey_topic_keyword_counts(SURVEY_REGION[src], variant)
        return {t: list(zip(kw["word"].astype(str), kw["count"].astype(float))) for t, kw in table.items()}
    word, count = ("Noun", "Frequency") if src == "weibo" else ("word", "count")
    out = {}
    for t in topic_ids(src):
        name = f"{t}_merged_tf_idf.parquet" if src == "weibo" else f"tf-idf-{t}-merged.parquet"
        tf = pd.read_parquet(TF_IDF_DIR / f"tf_idf_{src}" / name).sort_values("TF-IDF", ascending=False).head(int(variant))
        out[t] = list(zip(tf[word].astype(str), tf[count].astype(float)))
    return out


def centroid_matrix(src: str, variant, cache: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, int], int]:
    """(unit centroid rows in topic_ids order, keywords used per topic, keywords without a vector)."""
    kw = keyword_weights(src, variant)
    rows, used, missing = [], {}, 0
    for t in topic_ids(src):
        pairs = [(cache[w], c) for w, c in kw[t] if w in cache]
        missing += len(kw[t]) - len(pairs)
        if not pairs:
            raise KeyError(f"{src}/{t} ({variant}): no keyword with a cached vector")
        E = np.vstack([np.asarray(v, dtype=float) for v, _ in pairs])
        w = np.array([c for _, c in pairs])
        c = (E * w[:, None]).sum(0) / w.sum()
        rows.append(c / np.linalg.norm(c))
        used[t] = len(pairs)
    return np.vstack(rows), used, missing


def metrics(C: np.ndarray) -> Dict[str, float]:
    G = C @ C.T
    lam = np.clip(np.linalg.eigvalsh(G)[::-1], 0, None)
    Cc = C - C.mean(0)
    lam_c = np.clip(np.linalg.eigvalsh(Cc @ Cc.T)[::-1], 0, None)
    return {"PR": participation_ratio(lam), "eRank": exponential_rank(lam),
            "first_share": float(lam[0] / lam.sum()), "mean_cos": float(G[np.triu_indices(len(C), 1)].mean()),
            "PR_centered": participation_ratio(lam_c), "eRank_centered": exponential_rank(lam_c)}


def analyse(src: str, cache) -> Dict[str, object]:
    out = {"source": src, "k": len(topic_ids(src)), "variants": {}}
    for variant in VARIANTS[kind(src)]:
        C, used, missing = centroid_matrix(src, variant, cache)
        out["variants"][str(variant)] = {**metrics(C), "kw_min": min(used.values()), "kw_max": max(used.values()),
                                         "kw_missing": missing}
        if variant == MAIN[kind(src)]:
            stored = pd.read_parquet(CENTROID_DIR / f"{src}_topic_centroids.parquet")
            S = stored[[c for c in stored.columns if c.startswith("dim_")]].to_numpy(float)
            S = S / np.linalg.norm(S, axis=1, keepdims=True)
            out["min_cos_rebuilt_vs_stored"] = float(np.min(np.sum(C * S, axis=1)))
            out["main"] = out["variants"][str(variant)]
    vals = {m: [v[m] for v in out["variants"].values()] for m in ("PR", "eRank")}
    out["range"] = {m: (float(min(v)), float(max(v))) for m, v in vals.items()}
    return out


def load_results(path: Path = RESULT_JSON) -> Dict[str, Dict]:
    """{src: result} written by ``run``; empty when it has not been run."""
    return json.loads(path.read_text()) if path.exists() else {}


def run() -> Path:
    with open(EMBEDDING_DIR / "cached_embedding.pkl", "rb") as fh:
        cache = pickle.load(fh)
    res = {src: analyse(src, cache) for src in SOURCES}
    RESULT_JSON.write_text(json.dumps(res, indent=1))
    L = ["Paper statistics report  (task: semantic_rank_2c)", "Generated by analysis/semantic/semantic_rank.py.", "",
         "SOURCE: data/embedding/cached_embedding.pkl (OpenAI text-embedding-3-small keyword vectors); survey keywords",
         "  data/tf_idf/survey_<region>_topic_keywords_top5.csv (main), _selected.csv, _tfidf_curated.csv; platform",
         "  keywords: top TF-IDF words pooled over years, data/tf_idf/tf_idf_<src>/; main centroids checked against",
         "  data/embedding/topic_centroids/<src>_topic_centroids.parquet.",
         "CALCULATION: module docstring. PR = (sum lam)^2 / sum lam^2, eRank = exp(entropy of lam / sum lam); lam the",
         "  eigenvalues of the 9 x 9 cosine matrix of unit topic centroids (= covariance and correlation over the",
         "  embedding coordinates, as the opinion PR uses the topic covariance). No interval: keyword-set sensitivity",
         "  below (a keyword bootstrap is biased upward here; see docstring).",
         "  'Shared removed': mean topic vector subtracted first (centres across topics; at most K - 1 = 8 dims);",
         "  describes the shared component only.", "",
         "MAIN (Fig 3B)"]
    L.append(f"  {'source':18s} {'rebuilt~stored':>14s} {'mean cos':>8s} {'PR':>6s} {'eRank':>6s} {'1st eig share':>13s}"
             f" {'PR shared removed':>18s} {'eRank shared removed':>21s}")
    for src, r in res.items():
        m = r["main"]
        L.append(f"  {SOURCE_LABEL[src]:18s} {r['min_cos_rebuilt_vs_stored']:14.4f} {m['mean_cos']:8.3f} {m['PR']:6.2f}"
                 f" {m['eRank']:6.2f} {m['first_share']:13.2f} {m['PR_centered']:18.2f} {m['eRank_centered']:21.2f}")
    L += ["", "KEYWORD-SET SENSITIVITY (survey: top5 = main, selected, curated; platform: top 10 = main, 5, 20 words)",
          f"  {'source':18s} {'variant':>9s} {'kw/topic':>9s} {'no vector':>9s} {'mean cos':>8s} {'PR':>6s} {'eRank':>6s}"]
    for src, r in res.items():
        for variant, v in r["variants"].items():
            L.append(f"  {SOURCE_LABEL[src]:18s} {variant:>9s} {v['kw_min']:>4d}-{v['kw_max']:<4d} {v['kw_missing']:9d}"
                     f" {v['mean_cos']:8.3f} {v['PR']:6.2f} {v['eRank']:6.2f}")
        L.append(f"  {'':18s} {'range':>9s} {'':9s} {'':9s} {'':8s} {r['range']['PR'][0]:.2f}-{r['range']['PR'][1]:.2f}"
                 f" {r['range']['eRank'][0]:.2f}-{r['range']['eRank'][1]:.2f}")
    L += ["", "Read within a medium only: platform topics are described by TF-IDF words that share platform vocabulary",
          "(higher mean cosine), so semantic PR is not comparable between surveys and platforms."]
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    out = STATS_DIR / f"{date_prefix()}_semantic_rank_2c_results_stats.txt"
    out.write_text("\n".join(L) + "\n")
    print(out.read_text())
    return out


if __name__ == "__main__":
    import fire

    fire.Fire(run)
