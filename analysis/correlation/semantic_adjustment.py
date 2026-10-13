"""Holding semantic similarity fixed when comparing pairwise |r| across societies.

Each society's 36 topic pairs have their own semantic-similarity distribution,
so a raw difference in mean |r| can partly come from one society asking about
more (or less) similar topics. Tools (Fig 2G, supplementary Fig 2s;
paper_stats Parts 9, 10):

1. Reweighting to the U.S. similarity distribution (main, Fig 2G). Within one
   medium, the other society's observed pairs are reweighted so their
   similarities are distributed like the U.S. pairs' -- no fitted curve.
   "kde" (main): weight_j = f_US(x_j) / f_other(x_j), both densities Gaussian
   KDEs (Scott bandwidth; DiNardo-Fortin-Lemieux); the weighted mean |r| is the
   society's mean |r| "had its topics been as similar as the U.S. topics".
   Sensitivity: "nn": for each U.S. pair, the mean |r| of the k other-society
   pairs with the closest similarity; "strata": U.S. similarity quintiles,
   each weighted by its U.S. share (other pairs outside the U.S. range
   dropped).

   The raw gap to the U.S. then splits into a part due to the different
   similarity mix (adjusted - raw) and a part due to different |r| at equal
   similarity (U.S. - adjusted).

2. Shifting each society to the U.S. similarities along a fitted curve (OLS
   or LOWESS; supplementary Fig 2s and sensitivity). Within one medium, fit
   the other society's |r| on similarity (f). Its pair of similarity rank k is moved to the U.S. similarity of the
   same rank (quantile matching), and its |r| moves along the fitted curve:
       |r|*_j = |r|_j + f(x_US,(k)) - f(x_j).
   Each pair keeps its residual, so the result is the society's whole |r|
   distribution "had its topics been as similar as the U.S. topics"; its mean
   is the adjusted mean |r|. With OLS (and equal pair counts) that mean is
   mean|r| + b * (mean sim_US - mean sim_other). LOWESS (span ``frac``) is
   evaluated by linear interpolation of the fitted curve and held constant
   beyond the society's own similarity range.

3. Shapley R^2 decomposition. Pool one medium's pairs across societies and
   regress Fisher z = arctanh(|r|) on society (dummies) and similarity. Each
   factor's share is its R^2 gain averaged over both entry orders, so the two
   shares add up to the full model's R^2 whichever is entered first.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.smoothers_lowess import lowess

FIT_METHODS = ("ols", "lowess")
ADJUSTMENT_METHODS = ("kde", "nn", "strata") + FIT_METHODS


def matched_similarities(x_ref: np.ndarray, x_other: np.ndarray) -> np.ndarray:
    """The reference similarity of the same rank for every ``x_other`` value.

    Rank k of n maps to the reference quantile (k + 0.5) / n (Hazen), which is
    exactly the k-th smallest reference value when both have n pairs.
    """
    x_ref, x_other = np.asarray(x_ref, float), np.asarray(x_other, float)
    rank = np.empty(len(x_other))
    rank[np.argsort(x_other, kind="stable")] = np.arange(len(x_other))
    return np.quantile(x_ref, (rank + 0.5) / len(x_other), method="hazen")


def fitted_curve(x: np.ndarray, y: np.ndarray, fit: str = "ols", frac: float = 0.4):
    """Callable fitted |r| = f(similarity) for one society."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if fit == "ols":
        slope, intercept = np.polyfit(x, y, 1)
        return lambda v: intercept + slope * np.asarray(v, float)
    if fit == "lowess":
        curve = lowess(y, x, frac=frac)
        return lambda v: np.interp(np.asarray(v, float), curve[:, 0], curve[:, 1])
    raise ValueError(f"fit must be one of {FIT_METHODS}, got {fit!r}")


def shifted_values(
    x_ref: np.ndarray, x_other: np.ndarray, y_other: np.ndarray,
    fit: str = "ols", frac: float = 0.4,
) -> np.ndarray:
    """Each other-society pair's |r| moved to the U.S. similarity of its rank."""
    f = fitted_curve(x_other, y_other, fit, frac)
    return np.asarray(y_other, float) + f(matched_similarities(x_ref, x_other)) - f(x_other)


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
    frac: float = 0.4,
) -> float:
    """Mean of ``y_other`` with its similarities redistributed like ``x_ref``."""
    x_ref, x_other, y_other = (np.asarray(a, float) for a in (x_ref, x_other, y_other))
    if method in FIT_METHODS:
        return float(shifted_values(x_ref, x_other, y_other, method, frac).mean())
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
