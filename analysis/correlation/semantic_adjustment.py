"""Holding semantic similarity fixed when comparing pairwise |r| across societies.

Each society's 36 topic pairs have their own semantic-similarity distribution,
so a raw difference in mean |r| can partly come from one society asking about
more (or less) similar topics. Two tools (Fig 2G, H; paper_stats Parts 9, 10):

1. Reweighting to the U.S. similarity distribution (DiNardo-Fortin-Lemieux).
   Within one medium, the other society's pairs are reweighted so their
   similarities are distributed like the U.S. pairs'; the weighted mean |r| is
   the society's mean |r| "had it been asked about topics as similar as the
   U.S. topics". The raw gap to the U.S. then splits into a part due to the
   different similarity mix (raw - adjusted) and a part due to different |r|
   at equal similarity (U.S. - adjusted).

   Methods: "kde" (main): weight_j = f_US(x_j) / f_other(x_j), both densities
   Gaussian KDEs (Scott bandwidth); "nn": for each U.S. pair, the mean |r| of
   the k other-society pairs with the closest similarity; "strata": U.S.
   similarity quintiles, each quintile weighted by its U.S. share (other
   pairs outside the U.S. range dropped).

2. Shapley R^2 decomposition. Pool one medium's pairs across societies and
   regress Fisher z = arctanh(|r|) on society (dummies) and similarity. Each
   factor's share is its R^2 gain averaged over both entry orders, so the two
   shares add up to the full model's R^2 whichever is entered first.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.stats import gaussian_kde

ADJUSTMENT_METHODS = ("kde", "nn", "strata")


def density_ratio_weights(x_ref: np.ndarray, x_other: np.ndarray) -> np.ndarray:
    """Weights (summing to 1) that give ``x_other`` the distribution of ``x_ref``."""
    x_ref, x_other = np.asarray(x_ref, float), np.asarray(x_other, float)
    if np.array_equal(np.sort(x_ref), np.sort(x_other)):
        return np.full(len(x_other), 1 / len(x_other))
    ratio = gaussian_kde(x_ref)(x_other) / gaussian_kde(x_other)(x_other)
    return ratio / ratio.sum()


def effective_sample_size(w: np.ndarray) -> float:
    """Kish effective number of pairs behind a weighted mean."""
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum())


def _nn_mean(x_ref, x_other, y_other, k: int) -> float:
    k = min(k, len(x_other))
    dist = np.abs(x_ref[:, None] - x_other[None, :])
    nearest = np.argsort(dist, axis=1, kind="stable")[:, :k]
    return float(y_other[nearest].mean())


def _strata_mean(x_ref, x_other, y_other, n_strata: int) -> float:
    edges = np.quantile(x_ref, np.linspace(0, 1, n_strata + 1))
    inside = (x_other >= edges[0]) & (x_other <= edges[-1])
    ref_bin = np.clip(np.searchsorted(edges, x_ref, side="right") - 1, 0, n_strata - 1)
    other_bin = np.clip(np.searchsorted(edges, x_other, side="right") - 1, 0, n_strata - 1)
    share = np.bincount(ref_bin, minlength=n_strata) / len(x_ref)
    total, used = 0.0, 0.0
    for b in range(n_strata):
        members = inside & (other_bin == b)
        if members.any() and share[b] > 0:
            total += share[b] * y_other[members].mean()
            used += share[b]
    return float(total / used)


def adjusted_mean(
    x_ref: np.ndarray,
    x_other: np.ndarray,
    y_other: np.ndarray,
    method: str = "kde",
    *,
    k: int = 3,
    n_strata: int = 5,
) -> float:
    """Mean of ``y_other`` with its similarities redistributed like ``x_ref``."""
    x_ref, x_other, y_other = (np.asarray(a, float) for a in (x_ref, x_other, y_other))
    if method == "kde":
        return float(density_ratio_weights(x_ref, x_other) @ y_other)
    if method == "nn":
        return _nn_mean(x_ref, x_other, y_other, k)
    if method == "strata":
        return _strata_mean(x_ref, x_other, y_other, n_strata)
    raise ValueError(f"method must be one of {ADJUSTMENT_METHODS}, got {method!r}")


def gap_decomposition(x_ref, y_ref, x_other, y_other, method: str = "kde") -> Dict[str, float]:
    """Split mean|r|(ref) - mean|r|(other) into similarity mix and society parts."""
    adj = adjusted_mean(x_ref, x_other, y_other, method=method)
    raw_gap = float(np.mean(y_ref) - np.mean(y_other))
    return {
        "raw_gap": raw_gap,
        "adjusted_other": adj,
        "similarity_part": float(adj - np.mean(y_other)),
        "society_part": float(np.mean(y_ref) - adj),
    }


def _r2(columns: Sequence[np.ndarray], y: np.ndarray) -> float:
    X = np.column_stack([np.ones(len(y)), *columns])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    centred = y - y.mean()
    return float(1 - resid @ resid / (centred @ centred))


def shapley_r2(y: np.ndarray, groups: np.ndarray, x: np.ndarray) -> Dict[str, float]:
    """Shapley shares of R^2 for society (``groups``) and similarity (``x``).

    Returns society, similarity, total (= full-model R^2), unexplained, and
    the R^2 of each factor alone (``society_alone``, ``similarity_alone``).
    """
    y, x, groups = np.asarray(y, float), np.asarray(x, float), np.asarray(groups)
    levels = np.unique(groups)
    dummies = [(groups == g).astype(float) for g in levels[1:]]
    r_soc, r_sim = _r2(dummies, y), _r2([x], y)
    r_full = _r2(dummies + [x], y)
    society = 0.5 * (r_soc + (r_full - r_sim))
    similarity = 0.5 * (r_sim + (r_full - r_soc))
    return {
        "society": society,
        "similarity": similarity,
        "total": r_full,
        "unexplained": 1 - r_full,
        "society_alone": r_soc,
        "similarity_alone": r_sim,
    }
