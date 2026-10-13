"""Holding semantic similarity fixed when comparing pairwise |r| across societies.

Each society's 36 topic pairs have their own semantic-similarity distribution,
so a raw difference in mean |r| can partly come from one society asking about
more (or less) similar topics. Tools (Fig 2G, supplementary Figs 2s;
paper_stats Parts 9-11):

1. Density-ratio reweighting to the U.S. similarity distribution (Fig 2G).
   Within one medium, the other society's observed pairs are reweighted so
   their similarities are distributed like the U.S. pairs':
   weight_j = f_US(x_j) / f_other(x_j), both densities Gaussian KDEs (Scott
   bandwidth; DiNardo-Fortin-Lemieux). The weighted mean |r| is the society's
   mean |r| "had its topics been as similar as the U.S. topics". Weighting
   rather than matching: with 36 pairs per society, matching would let a few
   pairs stand in for many. Sensitivity: "strata" (U.S. similarity quintiles,
   each weighted by its U.S. share; other pairs outside the U.S. range
   dropped). U.S. pairs beyond the other society's similarity range cannot be
   reproduced by any weighting, so the reweighted distribution is close to,
   not identical with, the U.S. one.

   The raw gap to the U.S. then splits into a part due to the different
   similarity mix (adjusted - raw) and a part due to different |r| at equal
   similarity (U.S. - adjusted).

2. Weighted fits of |r| on similarity (supplementary Figs 2s): OLS, or a
   local-linear LOESS with tricube kernel over the nearest ``frac`` of the
   pairs, each optionally carrying the reweighting weights. They describe the
   |r|-similarity relation before and after reweighting; they do not adjust.

3. Shapley R^2 decomposition. Pool one medium's pairs across societies and
   regress Fisher z = arctanh(|r|) on society (dummies) and similarity. Each
   factor's share is its R^2 gain averaged over both entry orders, so the two
   shares add up to the full model's R^2 whichever is entered first.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
import statsmodels.api as sm
from scipy.stats import gaussian_kde

ADJUSTMENT_METHODS = ("kde", "strata")
FITS = ("ols", "loess")


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
    n_strata: int = 5,
) -> float:
    """Mean of ``y_other`` with its similarities redistributed like ``x_ref``."""
    x_ref, x_other, y_other = (np.asarray(a, float) for a in (x_ref, x_other, y_other))
    if method == "kde":
        return float(density_ratio_weights(x_ref, x_other) @ y_other)
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


def weighted_fit(
    x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None,
    fit: str = "ols", frac: float = 0.4,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fitted |r| = f(similarity) with pair weights ``w`` (None = equal).

    "ols": weighted least-squares line. "loess": at each evaluation point a
    weighted straight-line fit over the nearest ceil(frac * n) pairs, local
    weights tricube(distance / distance to the farthest of them) * w; no
    robustness iterations, so the same estimator serves raw and weighted
    pairs.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.ones(len(x)) if w is None else np.asarray(w, float)
    if fit == "ols":
        slope, intercept = np.polyfit(x, y, 1, w=np.sqrt(w))
        return lambda v: intercept + slope * np.asarray(v, float)
    if fit != "loess":
        raise ValueError(f"fit must be one of {FITS}, got {fit!r}")
    k = max(3, int(np.ceil(frac * len(x))))

    def f(v):
        v = np.atleast_1d(np.asarray(v, float))
        out = np.empty(len(v))
        for i, v0 in enumerate(v):
            d = np.abs(x - v0)
            h = np.sort(d)[k - 1] * 1.0001 or 1.0
            lw = np.clip(1 - (d / h) ** 3, 0, None) ** 3 * w
            X = np.column_stack([np.ones(len(x)), x - v0])
            beta = np.linalg.lstsq(X * np.sqrt(lw)[:, None], y * np.sqrt(lw), rcond=None)[0]
            out[i] = beta[0]
        return out
    return f


def weighted_ols(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> Dict[str, float]:
    """|r| = intercept + slope * similarity by weighted least squares
    (statsmodels WLS; equal weights = OLS): intercept, slope, slope SE,
    two-sided p, and the weighted R^2."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.ones(len(x)) if w is None else np.asarray(w, float)
    res = sm.WLS(y, sm.add_constant(x), weights=w).fit()
    return {
        "intercept": float(res.params[0]), "slope": float(res.params[1]),
        "se": float(res.bse[1]), "p": float(res.pvalues[1]), "r2": float(res.rsquared),
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
