"""Ngram-level embedding subspace diagnostics for the 6 formal sources.

Rebuild note (2026-07-23): earlier revision loaded the pre-aggregated 9x9
pairwise cosine matrix from ``<src>_topic_distance.csv`` /
``dynamic_embedding_<src>_gpt.csv`` and did SVD on that. That is a
centroid-only diagnostic and discards the underlying ngram geometry. The
current version loads the actual 1536-D ngram embeddings from the OpenAI
text-embedding-3-small cache used by the rest of the pipeline
(``data/embedding/cached_embedding.pkl``) and works on the full ngram matrix.

Ngram provenance mirrors the reproducible pipeline:
* Social (twitter, weibo, eutwitter/entwitter/eu_nentwitter):
  ``data/tf_idf/tf_idf_<src>/tf-idf-<topic>-merged.parquet`` (weibo uses
  ``<topic>_merged_tf_idf.parquet`` with Noun/Frequency column names).
  Ngrams: top ``top_k`` rows sorted by TF-IDF (same as
  ``semantic_similarity.process_embedding``).
* Surveys (anes/anes_media, evs/en_evs/evs_media/..., wvs/wvs_media):
  ``data/tf_idf/survey_{us,europe,china}_topic_keywords_top5.csv``: codebook
  keywords selected against the topic names, top 5, equal weights (same as
  ``semantic_similarity.survey_topic_keyword_counts``).

Each ngram row is unit-normalized before stacking so the analysis reports
purely on semantic directions. Every diagnostic operates on the
per-source stacked matrix ``X ∈ R^{N × 1536}`` (N = sum of per-topic
ngram counts) and per-topic row-block ``X_t ⊂ X``:

1. ``svd`` — rank, effective rank (participation ratio), condition
   number, and top singular-value spectrum of ``X``. Answers "how many
   independent semantic directions does this source's ngram corpus
   actually span?"

2. ``expressibility`` — for every (source, target topic t, k=1..n-1),
   enumerate the C(n-1, k) subsets S of *other* topics and compute two
   residuals of X_t against the row-span of X_S:
     centroid_residual = ||c_t - c_t Q_S Q_S^T|| / ||c_t||
     cloud_residual    = ||X_t - X_t Q_S Q_S^T||_F / ||X_t||_F
   where Q_S is the orthonormal basis of row-span(X_S) obtained via SVD.
   Reports the (target, subset) minimizing each residual per (source, k).

Both write CSVs under ``outputs/reports/<date>_embedding_subspace/``. Missing ngrams (not in the cache) are
skipped and a per-source coverage summary is printed.

Run from the repo root:

    python -m robustness.embedding_subspace_analysis centroids
    python -m robustness.embedding_subspace_analysis svd
    python -m robustness.embedding_subspace_analysis expressibility
"""

from __future__ import annotations

import pickle
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from common.paths import EMBEDDING_DIR, TF_IDF_DIR, report_dir
from plotting.figures import RESTRICTED_TOPICS, SOCIAL_SOURCES, _pretty_topic


FORMAL_SOURCES: List[str] = ["anes", "evs", "wvs", "twitter", "eutwitter", "weibo"]

_CACHE_PATH = EMBEDDING_DIR / "cached_embedding.pkl"
_TFIDF_DIR = TF_IDF_DIR

# Survey source -> curated-keyword region (semantic_similarity.SURVEY_REGIONS).
_SURVEY_REGION: Dict[str, str] = {
    **{s: "us" for s in ("anes", "anes_media")},
    **{s: "europe" for s in ("evs", "evs_media", "evs_resample2", "evs_media_resample2",
                             "en_evs", "en_evs_media", "eu_nen_evs")},
    **{s: "china" for s in ("wvs", "wvs_media")},
}

_SOCIAL_WORD_COL = {"weibo": "Noun"}       # default: "word"
_SOCIAL_COUNT_COL = {"weibo": "Frequency"} # default: "count"

_CENTROIDS_DIR = EMBEDDING_DIR / "topic_centroids"


# -- Cache --------------------------------------------------------------------

_cache_singleton: Optional[Dict[str, Any]] = None


def _load_ngram_cache() -> Dict[str, Any]:
    """Load the OpenAI text-embedding-3-small ngram cache from disk (once)."""
    global _cache_singleton
    if _cache_singleton is None:
        if not _CACHE_PATH.exists():
            raise FileNotFoundError(
                f"{_CACHE_PATH} not found. Run the embedding pipeline (see "
                f"semantic_similarity.DynamicSemanticEmbedding) to build it."
            )
        with open(_CACHE_PATH, "rb") as f:
            _cache_singleton = pickle.load(f)
    return _cache_singleton


def _lookup_ngram(word: str, cache: Dict[str, Any]) -> Optional[np.ndarray]:
    """Return the 1536-D embedding for ``word`` or None if not cached."""
    if word not in cache:
        return None
    vec = np.asarray(cache[word], dtype=np.float64)
    if vec.ndim != 1 or vec.size == 0:
        return None
    return vec


def ensure_cache_coverage(
    words: Sequence[str],
    verbose: bool = True,
    persist: bool = True,
) -> Dict[str, int]:
    """Embed any ``words`` missing from ``cached_embedding.pkl`` via OpenAI.

    Mirrors ``DynamicSemanticEmbedding.get_embedding``: fills the same on-disk
    cache used by the rest of the pipeline. Returns
    {"already_cached", "newly_embedded", "still_missing"}.
    """
    cache = _load_ngram_cache()
    todo = [w for w in dict.fromkeys(words) if w and w not in cache]
    if not todo:
        return {"already_cached": len(words), "newly_embedded": 0, "still_missing": 0}
    if verbose:
        print(f"[cache] {len(todo)} ngrams missing; embedding via OpenAI...")

    from semantic_embedding import get_semantic_embedding  # lazy: only when needed

    embedded = 0
    for w in todo:
        try:
            vec = get_semantic_embedding(w)
        except Exception as exc:
            if verbose:
                print(f"[cache] failed to embed {w!r}: {exc}")
            continue
        cache[w] = vec
        embedded += 1
    if persist and embedded > 0:
        with open(_CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
        if verbose:
            print(f"[cache] wrote {embedded} new embeddings to {_CACHE_PATH}")
    return {
        "already_cached": len(words) - len(todo),
        "newly_embedded": embedded,
        "still_missing": len(todo) - embedded,
    }


def _collect_all_source_words(
    sources: Sequence[str],
    top_k_social: int = 50,
    top_k_survey: int = 100,
) -> List[str]:
    """Enumerate every ngram string that ``_load_topic_ngram_blocks`` will
    need across the given sources, without touching the cache.
    """
    words: List[str] = []
    for src in sources:
        is_social = src in SOCIAL_SOURCES
        for topic_id in RESTRICTED_TOPICS[src]:
            if is_social:
                df = _social_topic_ngrams(src, topic_id, top_k=top_k_social)
            else:
                df = _survey_topic_ngrams(src, str(topic_id), top_k=top_k_survey)
            words.extend(df["word"].astype(str).tolist())
    return words


# -- Per-source ngram loading -------------------------------------------------

def _social_topic_ngrams(
    src: str,
    topic_id: Union[str, int],
    year: str = "merged",
    top_k: int = 50,
) -> pd.DataFrame:
    """Top-``top_k`` TF-IDF ngrams for one (social src, topic).

    Returns DataFrame with columns ``word`` and ``count`` (both harmonized
    across weibo vs twitter/eutwitter schemas), already sorted by TF-IDF desc.
    """
    tf_idf_folder = _TFIDF_DIR / f"tf_idf_{src}"
    if src == "weibo":
        path = tf_idf_folder / f"{topic_id}_{year}_tf_idf.parquet"
    else:
        path = tf_idf_folder / f"tf-idf-{topic_id}-{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing for src={src} topic={topic_id}")
    df = pd.read_parquet(path)
    df = df.sort_values(by="TF-IDF", ascending=False).head(top_k)
    word_col = _SOCIAL_WORD_COL.get(src, "word")
    count_col = _SOCIAL_COUNT_COL.get(src, "count")
    out = pd.DataFrame({
        "word": df[word_col].astype(str).str.strip().tolist(),
        "count": df[count_col].astype(float).tolist(),
    })
    return out[out["word"].str.len() > 0].reset_index(drop=True)


def _survey_topic_ngrams(src: str, topic_id: str, top_k: int = 100) -> pd.DataFrame:
    """Main (top-5) keywords for one (survey src, topic), equal weights.

    Same input as ``semantic_similarity.survey_topic_keyword_counts``
    (``data/tf_idf/survey_<region>_topic_keywords_top5.csv``); ``top_k`` caps the list.
    """
    from analysis.semantic.semantic_similarity import survey_topic_keyword_counts

    region = _SURVEY_REGION.get(src)
    if region is None:
        raise KeyError(f"No survey region registered for src={src}")
    keywords = survey_topic_keyword_counts(region)
    if str(topic_id) not in keywords:
        raise KeyError(f"Topic '{topic_id}' not in curated {region} keywords: {list(keywords)}")
    return keywords[str(topic_id)].head(top_k).reset_index(drop=True)


def _load_topic_ngram_blocks(
    src: str,
    top_k_social: int = 50,
    top_k_survey: int = 100,
    unit_norm: bool = True,
) -> Tuple[List[Union[str, int]], List[np.ndarray], List[np.ndarray], Dict[str, int]]:
    """Load unit-normalized ngram matrices per topic for one source.

    Returns:
        topics: the topic ids in RESTRICTED_TOPICS[src] order
        X_blocks: list of (n_i, 1536) matrices, one per topic; rows are
                  unit-normalized when ``unit_norm=True``.
        w_blocks: list of (n_i,) weights per topic (freq for social,
                  rank for survey), for centroid computations.
        coverage: {"requested": total_requested, "found": total_found,
                   "missing": total_missing} across all topics.
    """
    cache = _load_ngram_cache()
    topics = list(RESTRICTED_TOPICS[src])
    X_blocks: List[np.ndarray] = []
    w_blocks: List[np.ndarray] = []
    total_req = total_found = 0
    missing_words: List[str] = []
    is_social = src in SOCIAL_SOURCES

    for topic_id in topics:
        if is_social:
            df = _social_topic_ngrams(src, topic_id, top_k=top_k_social)
        else:
            df = _survey_topic_ngrams(src, str(topic_id), top_k=top_k_survey)

        vecs: List[np.ndarray] = []
        weights: List[float] = []
        for word, weight in zip(df["word"].tolist(), df["count"].tolist()):
            total_req += 1
            vec = _lookup_ngram(word, cache)
            if vec is None:
                missing_words.append(f"{src}:{topic_id}:{word}")
                continue
            total_found += 1
            vecs.append(vec)
            weights.append(float(weight))

        if not vecs:
            raise RuntimeError(
                f"{src}/{topic_id}: no cached embeddings for any of the "
                f"{len(df)} ngrams — cache may be stale or wrong source."
            )
        X = np.asarray(vecs, dtype=np.float64)
        if unit_norm:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            X = X / norms
        X_blocks.append(X)
        w_blocks.append(np.asarray(weights, dtype=np.float64))

    coverage = {
        "requested": total_req,
        "found": total_found,
        "missing": total_req - total_found,
    }
    if coverage["missing"] > 0:
        preview = missing_words[:10]
        more = "..." if len(missing_words) > 10 else ""
        print(f"[{src}] cache miss: {coverage['missing']}/{coverage['requested']} "
              f"ngrams (preview: {preview}{more})")
    return topics, X_blocks, w_blocks, coverage


# -- 1. SVD / rank of the full ngram matrix ----------------------------------

def _effective_rank(sigmas: np.ndarray) -> float:
    """Participation ratio of sigma^2 — a soft rank."""
    s2 = sigmas.astype(np.float64) ** 2
    total = s2.sum()
    if total <= 0:
        return 0.0
    denom = (s2 ** 2).sum()
    return float(total * total / denom) if denom > 0 else 0.0


def embedding_svd_report(
    sources: Optional[Sequence[str]] = None,
    top_k_social: int = 50,
    top_k_survey: int = 100,
    out_csv: Union[str, Path, None] = "auto",
    tol_ratio: float = 1e-10,
    keep_spectrum: int = 30,
) -> pd.DataFrame:
    """Per-source rank / singular-value spectrum of the stacked ngram matrix X.

    X ∈ R^{N × 1536} with N = sum of per-topic ngram counts across the 9
    restricted topics. Rows are unit-normalized. Writes summary and full
    (or truncated at ``keep_spectrum``) singular-value spectrum.
    """
    sources = list(sources) if sources is not None else list(FORMAL_SOURCES)
    summary_rows: List[Dict] = []
    sval_rows: List[Dict] = []
    for src in sources:
        topics, X_blocks, _, coverage = _load_topic_ngram_blocks(
            src, top_k_social=top_k_social, top_k_survey=top_k_survey,
        )
        X = np.vstack(X_blocks)
        # SVD on X (N × D). singular values match sqrt(eigvals of X X^T = X^T X).
        # full_matrices=False keeps it thin.
        sigmas = np.linalg.svd(X, compute_uv=False)
        sigma_max = float(sigmas[0])
        rank_tol = tol_ratio * sigma_max
        rank = int((sigmas > rank_tol).sum())
        nonzero = sigmas[sigmas > rank_tol]
        sigma_min = float(nonzero.min()) if nonzero.size else 0.0
        cond = float(sigma_max / sigma_min) if sigma_min > 0 else float("inf")
        eff_rank = _effective_rank(sigmas)

        summary_rows.append({
            "source": src,
            "n_topics": len(topics),
            "n_ngrams": int(X.shape[0]),
            "embed_dim": int(X.shape[1]),
            "rank": rank,
            "effective_rank": eff_rank,
            "sigma_max": sigma_max,
            "sigma_min": sigma_min,
            "sigma_min_over_max": sigma_min / sigma_max if sigma_max > 0 else float("nan"),
            "cond_number": cond,
            "ngram_coverage": coverage["found"] / coverage["requested"]
                if coverage["requested"] else float("nan"),
        })
        cap = min(keep_spectrum, sigmas.size) if keep_spectrum else sigmas.size
        for i, sv in enumerate(sigmas[:cap]):
            sval_rows.append({"source": src, "index": i + 1, "sigma": float(sv)})

    summary = pd.DataFrame(summary_rows)
    pd.set_option("display.float_format", lambda x: f"{x:.6g}")
    print(summary.to_string(index=False))
    pd.reset_option("display.float_format")

    if out_csv is not None:
        out_path = report_dir("embedding_subspace") / "embedding_svd_report.csv" if out_csv == "auto" else Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_path, index=False)
        sval_path = out_path.with_name(out_path.stem + "_singular_values.csv")
        pd.DataFrame(sval_rows).to_csv(sval_path, index=False)
        print(f"wrote {out_path}")
        print(f"wrote {sval_path}")
    return summary


# -- 2. Ngram-level expressibility -------------------------------------------

def _basis_of_rowspan(X_S: np.ndarray, tol_ratio: float = 1e-10) -> np.ndarray:
    """Orthonormal basis (D × r) of row-space of X_S via SVD.

    Row-space of X_S is the subspace of R^D spanned by the ngram directions
    of subset S. Projection matrix is Q Q^T for the returned Q.
    """
    # SVD: X_S = U S V^T. V's columns (Vt.T) form an orthonormal basis of
    # the row-space; keep the ones with sigma above tolerance.
    try:
        U, s, Vt = np.linalg.svd(X_S, full_matrices=False)
    except np.linalg.LinAlgError:
        # LAPACK's divide-and-conquer driver (gesdd) occasionally fails to
        # converge on valid input; the QR-based driver is slower but robust.
        from scipy.linalg import svd as scipy_svd
        U, s, Vt = scipy_svd(X_S, full_matrices=False, lapack_driver="gesvd")
    if s.size == 0:
        return np.zeros((X_S.shape[1], 0))
    tol = tol_ratio * s.max()
    r = int((s > tol).sum())
    return Vt[:r].T  # (D, r)


def _residuals(
    Q: np.ndarray,
    X_t: np.ndarray,
    w_t: np.ndarray,
) -> Tuple[float, float]:
    """Cloud and centroid residuals of X_t against span(Q).

    - cloud_residual = ||X_t - X_t Q Q^T||_F / ||X_t||_F
    - centroid_residual = ||c_t - c_t Q Q^T|| / ||c_t||
      where c_t = weighted mean of X_t rows (weights=w_t).
    """
    if Q.size == 0:
        return 1.0, 1.0
    proj_rows = X_t @ Q          # (n_t, r)
    reconstructed = proj_rows @ Q.T
    residual_rows = X_t - reconstructed
    fro_X = float(np.linalg.norm(X_t, "fro"))
    fro_R = float(np.linalg.norm(residual_rows, "fro"))
    cloud = fro_R / fro_X if fro_X > 0 else 1.0

    if w_t.sum() > 0:
        c = (w_t @ X_t) / w_t.sum()
    else:
        c = X_t.mean(axis=0)
    c_norm = float(np.linalg.norm(c))
    if c_norm == 0:
        return cloud, 1.0
    c_proj = c @ Q
    c_hat = c_proj @ Q.T
    centroid = float(np.linalg.norm(c - c_hat) / c_norm)
    return cloud, centroid


def topic_subspace_residuals(
    src: str,
    top_k_social: int = 50,
    top_k_survey: int = 100,
) -> pd.DataFrame:
    """For every (target topic t, k=1..n-1), find the k-subset of *other*
    topics minimizing (a) cloud_residual and (b) centroid_residual of X_t
    against the row-span of the subset's stacked ngram matrix.
    """
    topics, X_blocks, w_blocks, _ = _load_topic_ngram_blocks(
        src, top_k_social=top_k_social, top_k_survey=top_k_survey,
    )
    n = len(topics)
    rows: List[Dict] = []
    for t_idx in range(n):
        others = [j for j in range(n) if j != t_idx]
        X_t = X_blocks[t_idx]
        w_t = w_blocks[t_idx]
        for k in range(1, n):
            best_cloud = float("inf")
            best_centroid = float("inf")
            best_cloud_subset: Tuple[int, ...] = ()
            best_centroid_subset: Tuple[int, ...] = ()
            for combo in combinations(others, k):
                X_S = np.vstack([X_blocks[j] for j in combo])
                Q = _basis_of_rowspan(X_S)
                cloud, centroid = _residuals(Q, X_t, w_t)
                if cloud < best_cloud:
                    best_cloud = cloud
                    best_cloud_subset = combo
                if centroid < best_centroid:
                    best_centroid = centroid
                    best_centroid_subset = combo

            # Report both. angles derived via arcsin(residual) — same
            # interpretation as principal-angle projection.
            rows.append({
                "source": src,
                "target_topic": str(topics[t_idx]),
                "target_label": _pretty_topic(topics[t_idx]),
                "k": k,
                "cloud_residual": best_cloud,
                "cloud_angle_deg": float(np.degrees(np.arcsin(
                    min(1.0, max(0.0, best_cloud))))),
                "cloud_cos_principal_angle": float(np.sqrt(
                    max(0.0, 1.0 - best_cloud ** 2))),
                "cloud_subset_topics": "+".join(
                    str(topics[j]) for j in best_cloud_subset),
                "cloud_subset_labels": " + ".join(
                    _pretty_topic(topics[j]) for j in best_cloud_subset),
                "centroid_residual": best_centroid,
                "centroid_angle_deg": float(np.degrees(np.arcsin(
                    min(1.0, max(0.0, best_centroid))))),
                "centroid_cos_principal_angle": float(np.sqrt(
                    max(0.0, 1.0 - best_centroid ** 2))),
                "centroid_subset_topics": "+".join(
                    str(topics[j]) for j in best_centroid_subset),
                "centroid_subset_labels": " + ".join(
                    _pretty_topic(topics[j]) for j in best_centroid_subset),
            })
    return pd.DataFrame(rows)


def topic_expressibility_report(
    sources: Optional[Sequence[str]] = None,
    top_k_social: int = 50,
    top_k_survey: int = 100,
    out_csv: Union[str, Path, None] = "auto",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep per-topic ngram-level residuals across (k=1..n-1) for all sources.

    Produces three CSVs:
    - `<stem>.csv` — full per-(src, target, k) table with both residuals.
    - `<stem>_per_source_min.csv` — the (src, target) with smallest
      cloud_residual across all k.
    - `<stem>_per_source_per_k.csv` — the (src, k, target) with smallest
      cloud_residual at each k (traces the k -> 0 decay curve).
    """
    sources = list(sources) if sources is not None else list(FORMAL_SOURCES)
    full = pd.concat(
        [topic_subspace_residuals(src,
                                  top_k_social=top_k_social,
                                  top_k_survey=top_k_survey) for src in sources],
        ignore_index=True,
    )

    best = (
        full.sort_values(["source", "cloud_residual"])
            .groupby("source", as_index=False)
            .first()
    )
    per_k = (
        full.sort_values(["source", "k", "cloud_residual"])
            .groupby(["source", "k"], as_index=False)
            .first()
    )

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    pd.set_option("display.max_colwidth", 100)
    print("=== Per-source most-expressible topic (min cloud_residual over all k) ===")
    print(best[[
        "source", "target_topic", "target_label", "k",
        "cloud_residual", "cloud_angle_deg", "cloud_subset_labels",
        "centroid_residual", "centroid_angle_deg",
    ]].to_string(index=False))
    print()
    print("=== Per-source x per-k minimum cloud_residual (k -> 0 decay) ===")
    print(per_k[[
        "source", "k", "target_topic", "target_label",
        "cloud_residual", "cloud_angle_deg",
        "centroid_residual", "centroid_angle_deg",
        "cloud_subset_labels",
    ]].to_string(index=False))
    pd.reset_option("display.float_format")
    pd.reset_option("display.max_colwidth")

    if out_csv is not None:
        out_path = report_dir("embedding_subspace") / "topic_expressibility_report.csv" if out_csv == "auto" else Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        full.to_csv(out_path, index=False)
        best_path = out_path.with_name(out_path.stem + "_per_source_min.csv")
        per_k_path = out_path.with_name(out_path.stem + "_per_source_per_k.csv")
        best.to_csv(best_path, index=False)
        per_k.to_csv(per_k_path, index=False)
        print(f"wrote {out_path}")
        print(f"wrote {best_path}")
        print(f"wrote {per_k_path}")
    return full, best


# -- 3. Per-topic centroid export --------------------------------------------

def _pipeline_topic_centroid(
    words: Sequence[str],
    weights: Sequence[float],
    cache: Dict[str, Any],
) -> Tuple[np.ndarray, int]:
    """Weighted mean of cached ngram embeddings for one topic (raw, not unit-normalized).

    Mirrors ``get_weighted_keywords_centroid`` in ``semantic_similarity``:
    ``sum(w * vec) / sum(w)`` with raw 1536-D ngram vectors (no per-row
    unit-normalization). Returns (centroid, n_ngrams_used).
    """
    vecs: List[np.ndarray] = []
    ws: List[float] = []
    for word, w in zip(words, weights):
        vec = _lookup_ngram(word, cache)
        if vec is None:
            continue
        vecs.append(vec)
        ws.append(float(w))
    if not vecs:
        return np.zeros(1536, dtype=np.float64), 0
    X = np.asarray(vecs, dtype=np.float64)
    w_arr = np.asarray(ws, dtype=np.float64)
    total = w_arr.sum()
    if total <= 0:
        centroid = X.mean(axis=0)
    else:
        centroid = (w_arr @ X) / total
    return centroid, len(vecs)


def export_topic_centroids(
    sources: Optional[Sequence[str]] = None,
    out_dir: Union[str, Path] = _CENTROIDS_DIR,
    top_k_social: int = 10,
    top_k_survey: int = 100,
    unit_norm: bool = False,
) -> Dict[str, Path]:
    """Export per-topic centroid vectors to ``<out_dir>/<src>_topic_centroids.parquet``.

    Weighting matches the pipeline that produces the similarity CSVs:
    * Social: top-10 TF-IDF ngrams per (topic, "merged" year), Frequency-weighted
      (``semantic_similarity.process_embedding``).
    * Survey: selected codebook keywords per topic, equal weights
      (``semantic_similarity.process_survey_embedding``).

    Parquet layout (one file per source, 9 rows):
        source, topic_id, topic_label, n_ngrams, weight_sum, centroid_norm,
        dim_0, dim_1, ..., dim_1535

    When ``unit_norm=True`` the ``dim_i`` columns are the unit-normalized
    centroid; otherwise the raw weighted mean is exported (matches the
    pipeline's ``keywords_centroid``). ``centroid_norm`` records the raw
    L2 norm regardless.
    """
    sources = list(sources) if sources is not None else list(FORMAL_SOURCES)
    cache = _load_ngram_cache()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {}
    for src in sources:
        is_social = src in SOCIAL_SOURCES
        rows: List[Dict[str, Any]] = []
        for topic_id in RESTRICTED_TOPICS[src]:
            if is_social:
                df = _social_topic_ngrams(src, topic_id, top_k=top_k_social)
            else:
                df = _survey_topic_ngrams(src, str(topic_id), top_k=top_k_survey)
            centroid, n = _pipeline_topic_centroid(
                df["word"].tolist(), df["count"].tolist(), cache,
            )
            norm = float(np.linalg.norm(centroid))
            weight_sum = float(sum(df["count"].tolist()))
            row_vec = centroid
            if unit_norm and norm > 0:
                row_vec = centroid / norm
            row: Dict[str, Any] = {
                "source": src,
                "topic_id": str(topic_id),
                "topic_label": _pretty_topic(topic_id),
                "n_ngrams": int(n),
                "weight_sum": weight_sum,
                "centroid_norm": norm,
            }
            row.update({f"dim_{i}": float(row_vec[i]) for i in range(row_vec.size)})
            rows.append(row)
        out_df = pd.DataFrame(rows)
        out_path = out_dir / f"{src}_topic_centroids.parquet"
        out_df.to_parquet(out_path, index=False)
        written[src] = out_path
        print(f"[{src}] wrote {out_path} — {len(out_df)} topics, "
              f"n_ngrams range {out_df['n_ngrams'].min()}–{out_df['n_ngrams'].max()}, "
              f"norm range {out_df['centroid_norm'].min():.3f}–{out_df['centroid_norm'].max():.3f}"
              + (" (unit-normalized)" if unit_norm else ""))
    return written


# -- CLI ---------------------------------------------------------------------

if __name__ == "__main__":
    import fire

    def _cli_svd(
        sources: Optional[Sequence[str]] = None,
        top_k_social: int = 50,
        top_k_survey: int = 100,
        out_csv: Union[str, Path, None] = "auto",
        keep_spectrum: int = 30,
    ) -> None:
        embedding_svd_report(
            sources=sources,
            top_k_social=top_k_social,
            top_k_survey=top_k_survey,
            out_csv=out_csv,
            keep_spectrum=keep_spectrum,
        )

    def _cli_expressibility(
        sources: Optional[Sequence[str]] = None,
        top_k_social: int = 50,
        top_k_survey: int = 100,
        out_csv: Union[str, Path, None] = "auto",
    ) -> None:
        topic_expressibility_report(
            sources=sources,
            top_k_social=top_k_social,
            top_k_survey=top_k_survey,
            out_csv=out_csv,
        )

    def _cli_centroids(
        sources: Optional[Sequence[str]] = None,
        out_dir: Union[str, Path] = _CENTROIDS_DIR,
        top_k_social: int = 10,
        top_k_survey: int = 100,
        unit_norm: bool = False,
    ) -> None:
        export_topic_centroids(
            sources=sources,
            out_dir=out_dir,
            top_k_social=top_k_social,
            top_k_survey=top_k_survey,
            unit_norm=unit_norm,
        )

    fire.Fire({
        "svd": _cli_svd,
        "expressibility": _cli_expressibility,
        "centroids": _cli_centroids,
    })
