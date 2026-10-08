#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10 Sep 2025
Compute intrinsic dimensions of data with missing entries, including
spectral metrics and Hausdorff dimension estimation.

Produced by ChatGPT 5 and Junming Huang

Estimate intrinsic (Hausdorff) dimension from high-dimensional point clouds with
missing entries, using:
  - Levina-Bickel kNN-MLE estimator (primary)
  - TwoNN estimator (parameter-free sanity check)
  - Spectral metrics via PCA/SVD: Participation Ratio (PR), eRank, stable-rank

Levina-Bickel 最大似然方法 (Levina & Bickel, 2005): 基于 k 最近邻距离分布的似然函数来估计局部维度。
TwoNN (Facco et al., 2017): 基于最近邻/次近邻距离比率的无参数估计。
PR / eRank / srank: 基于协方差谱的“有效秩”度量，刻画线性子空间维度。

Distance correction under MCAR missingness:
  对于两个样本 i, j, 若观测到的共同特征集合为 S_ij, 则校正平方距离为:
      r_ij^2 ≈ (n_features / |S_ij|) * Σ_{k∈S_ij} (x_{ik} - x_{jk})^2

NEW in v2:
  - Robust handling of zero-overlap and tiny negative round-off in sqrt.
  - Input auto-orientation: rows = samples (N, 大边), cols = features (L, 小边)。

Usage examples:
  # simulate with toy data
  python -m analysis.dimension.dimensions

  # load real matrix (sparse .npz or dense .npy)
  python -m analysis.dimension.dimensions --input data.npz

  from analysis.dimension import dimensions
  dimensions.estimate(data_path_or_simulated="data.npz", distance_space="spectral")
  dimensions.estimate(data_path_or_simulated="data.npy", distance_space="spectral")
  dimensions.estimate(data_path_or_simulated=np.load("data.npy"), distance_space="spectral")
  dimensions.estimate(data_path_or_simulated=sparse.load_npz("data.npz"), distance_space="spectral")
  dimensions.estimate(data_path_or_simulated=np.load("data.npy"), distance_space="raw")
  dimensions.estimate(data_path_or_simulated=np.load("data.npy"), distance_space="bagged")

"""

import argparse
import numpy as np
from numpy.random import default_rng
from scipy import sparse
from pathlib import Path

# from sklearn.decomposition import IncrementalPCA

# ---------------- Simulation ----------------


def simulate_linear_manifold(
    L: int = 1000, N: int = 1000, n: int = 10, m: float = 0.3, seed: int = 42
):
    """Simulate N samples in R^L lying on an n-dim linear subspace.
    Return (N, L) matrix: rows = samples, cols = features.
    """
    rng = default_rng(seed)
    A = rng.normal(size=(L, n))
    Q, _ = np.linalg.qr(A)  # L x n orthonormal basis
    Z = rng.normal(size=(N, n))  # N x n latent coords (samples)
    X_full = Z @ Q.T  # (N x L)

    mask = rng.random(size=X_full.shape) < m
    rows, cols = np.where(mask)
    data = X_full[rows, cols].astype(np.float64)
    X_sparse = sparse.csr_matrix((data, (rows, cols)), shape=(N, L))
    B = X_sparse.copy()
    B.data = np.ones_like(B.data, dtype=float)
    A2 = X_sparse.copy()
    A2.data = A2.data**2
    return X_sparse, B, A2, Q, Z


# ---------------- Input orientation ----------------


def orient_rows_as_samples(X_any):
    """Ensure rows = samples (N, larger dim), cols = features (L, smaller dim)."""
    if sparse.issparse(X_any):
        Xs = X_any.tocsr()
        r, c = Xs.shape
        flipped = r < c
        if flipped:
            Xs = Xs.T.tocsr()
        return Xs, flipped
    else:
        Xd = np.asarray(X_any)
        r, c = Xd.shape
        flipped = r < c
        if flipped:
            Xd = Xd.T
        rows, cols = np.where(np.isfinite(Xd))
        data = Xd[rows, cols].astype(float)
        Xs = sparse.csr_matrix((data, (rows, cols)), shape=Xd.shape)
        return Xs, flipped


def build_indicator_and_square(X_sparse):
    B = X_sparse.copy()
    B.data = np.ones_like(B.data, dtype=float)
    A2 = X_sparse.copy()
    A2.data = A2.data**2
    return B, A2


# ---------------- Distance with missing data ----------------


def corrected_distances_to_i(i, X, B, A2, min_overlap: int = 5):
    n_samples, n_features = X.shape
    xi = X.getrow(i).T
    bi = B.getrow(i).T
    vi2 = xi.copy()
    vi2.data = vi2.data**2
    s0 = np.asarray((B @ vi2).toarray()).ravel()
    s1 = np.asarray((A2 @ bi).toarray()).ravel()
    p = np.asarray((X @ xi).toarray()).ravel()
    c = np.asarray((B @ bi).toarray()).ravel()
    sq = s0 + s1 - 2.0 * p
    d = np.full_like(sq, np.nan, dtype=float)
    valid = c >= max(1, int(min_overlap))
    if not np.any(valid):
        return d, c
    sq_corr = np.empty_like(sq, dtype=float)
    sq_corr[~valid] = np.nan
    sq_corr[valid] = (n_features / c[valid]) * sq[valid]
    tiny_neg = np.isfinite(sq_corr) & (sq_corr < 0) & (sq_corr > -1e-12)
    sq_corr[tiny_neg] = 0.0
    valid2 = np.isfinite(sq_corr) & (sq_corr >= 0)
    d[valid2] = np.sqrt(sq_corr[valid2])
    d[i] = np.nan
    return d, c


# -------- Feature-bagging distances (to stabilize low-m regimes) --------


def _column_submatrix_triplet(X, cols):
    """Return (X_sub, B_sub, A2_sub) for selected column indices (features)."""
    X_sub = X[:, cols]
    B_sub = X_sub.copy()
    B_sub.data = np.ones_like(B_sub.data, dtype=float)
    A2_sub = X_sub.copy()
    A2_sub.data = A2_sub.data**2
    return X_sub.tocsr(), B_sub.tocsr(), A2_sub.tocsr()


def bagged_corrected_distances_to_i(
    i,
    X,
    B,
    A2,
    num_bags: int = 15,
    bag_size: int = 20,
    min_overlap: int = 3,
    seed: int = 42,
):
    """Median-aggregate corrected distances over random feature bags.
    Useful when m is small: reduces variance from tiny overlaps dominating kNN.
    """
    rng = default_rng(seed)
    _, n_features = X.shape
    num_bags = max(1, int(num_bags))
    bag_size = min(n_features, max(1, int(bag_size)))
    all_dists = []
    # Pre-sample all column bags (without replacement per bag)
    for _ in range(num_bags):
        cols = rng.choice(n_features, size=bag_size, replace=False)
        Xs, Bs, A2s = _column_submatrix_triplet(X, cols)
        di, _ = corrected_distances_to_i(i, Xs, Bs, A2s, min_overlap=min_overlap)
        all_dists.append(di)
    D = np.vstack(all_dists)  # shape: (num_bags, n_samples)
    # Avoid RuntimeWarning: All-NaN slice encountered in np.nanmedian
    valid_any = np.any(np.isfinite(D), axis=0)
    med = np.full(D.shape[1], np.nan, dtype=float)
    if np.any(valid_any):
        med[valid_any] = np.nanmedian(D[:, valid_any], axis=0)
    return med


# ---------------- Estimators ----------------


def mle_intrinsic_dim(
    X,
    B,
    A2,
    k1=10,
    k2=30,
    max_points=300,
    seed=42,
    min_overlap: int = 5,
    bagging: bool = False,
    num_bags: int = 15,
    bag_size: int = 20,
):
    rng = default_rng(seed)
    n_samples, _ = X.shape
    if max_points is None or max_points >= n_samples:
        idx_points = np.arange(n_samples)
    else:
        idx_points = rng.choice(n_samples, size=max_points, replace=False)
    dims_per_point = []
    for i in idx_points:
        if bagging:
            dists = bagged_corrected_distances_to_i(
                i,
                X,
                B,
                A2,
                num_bags=num_bags,
                bag_size=bag_size,
                min_overlap=max(1, min_overlap),
                seed=int(default_rng(seed).integers(1e9)),
            )
        else:
            dists, _ = corrected_distances_to_i(i, X, B, A2, min_overlap=min_overlap)
        valid = np.isfinite(dists) & (dists > 0)
        di = dists[valid]
        if di.size < k2:
            continue
        di.sort()
        logs = np.log(di)
        csum = np.cumsum(logs)
        local_vals = []
        for k in range(k1, k2 + 1):
            if k - 1 <= 0:
                continue
            s = (k - 1) * logs[k - 1] - csum[k - 2]
            denom = (k - 1) if k < 3 else (k - 2)
            val = denom / s
            if np.isfinite(val) and val > 0:
                local_vals.append(val)
        if local_vals:
            dims_per_point.append(np.mean(local_vals))
    if not dims_per_point:
        return np.nan, np.nan, np.nan
    dims_per_point = np.array(dims_per_point)
    mean_dim = float(np.nanmean(dims_per_point))
    std_dim = float(np.nanstd(dims_per_point, ddof=1))
    se = std_dim / np.sqrt(len(dims_per_point))
    return mean_dim, mean_dim - 1.96 * se, mean_dim + 1.96 * se


def twonn_intrinsic_dim(
    X,
    B,
    A2,
    max_points=500,
    drop_small=0.02,
    seed=42,
    min_overlap: int = 5,
    bagging: bool = False,
    num_bags: int = 15,
    bag_size: int = 20,
):
    rng = default_rng(seed)
    n_samples, _ = X.shape
    if max_points is None or max_points >= n_samples:
        idx_points = np.arange(n_samples)
    else:
        idx_points = rng.choice(n_samples, size=max_points, replace=False)
    r1_list, r2_list = [], []
    for i in idx_points:
        if bagging:
            dists = bagged_corrected_distances_to_i(
                i,
                X,
                B,
                A2,
                num_bags=num_bags,
                bag_size=bag_size,
                min_overlap=max(1, min_overlap),
                seed=int(default_rng(seed).integers(1e9)),
            )
        else:
            dists, _ = corrected_distances_to_i(i, X, B, A2, min_overlap=min_overlap)
        valid = np.isfinite(dists) & (dists > 0)
        di = dists[valid]
        if di.size < 2:
            continue
        di.sort()
        r1_list.append(di[0])
        r2_list.append(di[1])
    if len(r1_list) < 30:
        return np.nan, (np.nan, np.nan)
    r1 = np.array(r1_list)
    r2 = np.array(r2_list)
    if drop_small > 0:
        q = np.quantile(r1, drop_small)
        keep = r1 > q
        r1 = r1[keep]
        r2 = r2[keep]
    mu = r2 / r1
    mu_sorted = np.sort(mu)
    F = (np.arange(1, len(mu_sorted) + 1)) / len(mu_sorted)
    eps = 1e-9
    F = np.clip(F, eps, 1 - eps)
    x = np.log(mu_sorted)
    y = -np.log(1 - F)
    slope = float(np.dot(x, y) / np.dot(x, x))
    return slope, (np.nan, np.nan)


# ---------------- Spectral metrics (pairwise-available covariance) ----------------

# ===== New: pairwise covariance eigendecomposition & spectral-distance pipeline =====


def pairwise_cov_eigendecomp(X):
    """
    Return eigenvalues/eigenvectors of pairwise-available covariance.
    cov_ij = (X^T X)_ij - C_ij * mu_i * mu_j  all divided by (C_ij - 1) when C_ij > 1.
    """
    B = X.copy()
    B.data = np.ones_like(B.data, dtype=float)
    counts = (B.T @ B).toarray().astype(float)  # L x L
    cross = (X.T @ X).toarray().astype(float)  # L x L
    col_sum = np.asarray(X.sum(axis=0)).ravel().astype(float)
    col_cnt = np.asarray(B.sum(axis=0)).ravel().astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.divide(col_sum, col_cnt, out=np.zeros_like(col_sum), where=col_cnt > 0)
    outer_mu = np.outer(mu, mu)
    cov = np.zeros_like(cross)
    mask = counts > 1
    denom = np.maximum(counts - 1.0, 1.0)
    cov[mask] = (cross[mask] - counts[mask] * outer_mu[mask]) / denom[mask]
    cov = 0.5 * (cov + cov.T)

    # Add regularization to ensure numerical stability
    # ridge = 1e-6
    # cov += ridge * np.eye(cov.shape[0])

    # try:
    lam, V = np.linalg.eigh(cov)  # ascending
    # except np.linalg.LinAlgError:
    #     # If eigendecomposition fails, try with more regularization
    #     print(f"[WARN] Eigendecomposition failed, using more regularization")
    #     cov += 1e-3 * np.eye(cov.shape[0])
    #     lam, V = np.linalg.eigh(cov)

    return lam.astype(float), V.astype(float)


def spectral_effective_ranks_from_lam(lam):
    lam = np.asarray(lam, dtype=float)
    lam = lam[lam > 0]
    if lam.size == 0:
        return np.nan, np.nan, np.nan
    S1 = float(lam.sum())
    S2 = float((lam**2).sum())
    PR = (S1**2) / S2 if S2 > 0 else np.nan
    p = lam / S1
    H = -np.sum(p * np.log(p + 1e-12))
    eRank = float(np.exp(H))
    srank = float(S1 / lam.max())
    return PR, eRank, srank


def choose_spectral_rank(PR, eRank, L, r_min=2):
    r = int(round(eRank)) if np.isfinite(eRank) else int(round(PR))
    r = min(max(r, r_min), L)
    return r


def spectral_coordinates_from_missing(X, V, r, ridge=1e-6):
    """
    Compute per-row spectral coordinates Z (n_samples x r) with missing data:
    For row x over observed cols J, solve (V_J^T V_J + ridge I) z = V_J^T x_J.
    Use the top-r eigenvectors (columns) from V (i.e., last r columns when V is ascending).
    """
    n_samples, n_features = X.shape
    Vr = V[:, -r:]  # largest r eigenvectors
    Z = np.full((n_samples, r), np.nan, dtype=float)
    for i in range(n_samples):
        row = X.getrow(i).tocoo()
        if row.nnz == 0:
            continue
        J = row.col
        xJ = row.data
        VJ = Vr[J, :]
        AtA = VJ.T @ VJ
        Atx = VJ.T @ xJ
        A = AtA + ridge * np.eye(r)
        try:
            zi = np.linalg.solve(A, Atx)
        except np.linalg.LinAlgError:
            zi = np.linalg.lstsq(A, Atx, rcond=None)[0]
        Z[i, :] = zi
    return Z


def point_distances_in_Z(i, Z):
    zi = Z[i, :]
    d = np.full(Z.shape[0], np.nan, dtype=float)
    if not np.all(np.isfinite(zi)):
        return d
    valid = np.all(np.isfinite(Z), axis=1)
    diff = Z[valid, :] - zi
    dd = np.sqrt(np.sum(diff * diff, axis=1))
    d[np.where(valid)[0]] = dd
    d[i] = np.nan
    return d


def mle_intrinsic_dim_from_Z(Z, k1=10, k2=30, max_points=300, seed=42):
    rng = default_rng(seed)
    n_samples = Z.shape[0]
    if max_points is None or max_points >= n_samples:
        idx_points = np.arange(n_samples)
    else:
        idx_points = rng.choice(n_samples, size=max_points, replace=False)
    dims_per_point = []
    for i in idx_points:
        dists = point_distances_in_Z(i, Z)
        valid = np.isfinite(dists) & (dists > 0)
        di = dists[valid]
        if di.size < k2:
            continue
        di.sort()
        logs = np.log(di)
        csum = np.cumsum(logs)
        local_vals = []
        for k in range(k1, k2 + 1):
            if k - 1 <= 0:
                continue
            s = (k - 1) * logs[k - 1] - csum[k - 2]
            denom = (k - 1) if k < 3 else (k - 2)
            val = denom / s
            if np.isfinite(val) and val > 0:
                local_vals.append(val)
        if local_vals:
            dims_per_point.append(np.mean(local_vals))
    if not dims_per_point:
        return np.nan, np.nan, np.nan
    dims_per_point = np.array(dims_per_point)
    mean_dim = float(np.nanmean(dims_per_point))
    std_dim = float(np.nanstd(dims_per_point, ddof=1))
    se = std_dim / np.sqrt(len(dims_per_point))
    return mean_dim, mean_dim - 1.96 * se, mean_dim + 1.96 * se


def twonn_intrinsic_dim_from_Z(Z, max_points=500, drop_small=0.02, seed=42):
    rng = default_rng(seed)
    n_samples = Z.shape[0]
    if max_points is None or max_points >= n_samples:
        idx_points = np.arange(n_samples)
    else:
        idx_points = rng.choice(n_samples, size=max_points, replace=False)
    r1_list, r2_list = [], []
    for i in idx_points:
        dists = point_distances_in_Z(i, Z)
        valid = np.isfinite(dists) & (dists > 0)
        di = dists[valid]
        if di.size < 2:
            continue
        di.sort()
        r1_list.append(di[0])
        r2_list.append(di[1])
    if len(r1_list) < 30:
        return np.nan, (np.nan, np.nan)
    r1 = np.array(r1_list)
    r2 = np.array(r2_list)
    if drop_small > 0:
        q = np.quantile(r1, drop_small)
        keep = r1 > q
        r1 = r1[keep]
        r2 = r2[keep]
    mu = r2 / r1
    mu_sorted = np.sort(mu)
    F = (np.arange(1, len(mu_sorted) + 1)) / len(mu_sorted)
    eps = 1e-9
    F = np.clip(F, eps, 1 - eps)
    x = np.log(mu_sorted)
    y = -np.log(1 - F)
    slope = float(np.dot(x, y) / np.dot(x, x))
    return slope, (np.nan, np.nan)


# ---------------- Run spectral metrics and Hausdorff dimension estimation ----------------
def estimate(
    data_path_or_simulated,  #: str|Path|np.ndarray|sparse.spmatrix,
    min_overlap=None,
    bag_size=None,
    num_bags=None,
    distance_space: str = "spectral",
    max_points_mle: int = 300,
    max_points_twonn: int = 500,
    random_seed: int = 42,
):
    """
    Estimate intrinsic dimension of data with missing entries.

    Parameters
    ----------
    data_path_or_simulated : str | Path | np.ndarray | scipy.sparse.spmatrix
        Input data. Can be a filepath to `.npz` (sparse) or `.npy` (dense) file,
        or an in-memory NumPy array / SciPy sparse matrix. Rows = samples, cols = features.
        NaN entries are treated as missing.

    Parameters only used for raw/bagged distance when estimating Hausdorff dimension
    ----------
    min_overlap : int or None, optional
        Minimum number of overlapping observed features required to compute raw/bagged distances.
        If None, an auto-suggested value is chosen based on observation rate.
    bag_size : int or None, optional
        Number of features per random bag when using bagged distances. If None, auto-suggested.
    num_bags : int or None, optional
        Number of random feature bags for bagged distances. If None, auto-suggested.
    distance_space : {"spectral", "raw", "bagged", None}, default="spectral"
        Which space to compute distances for MLE/TwoNN estimators.
        - "spectral": embed samples into spectral coordinates (from covariance eigendecomp).
        - "raw": use raw corrected distances directly in observed feature space.
        - "bagged": use median-aggregated corrected distances from random feature bags.
        - None: do not compute MLE/TwoNN estimators.
    max_points_mle : int, default=300
        Max number of samples to subsample for the MLE estimator.
    max_points_twonn : int, default=500
        Max number of samples to subsample for the TwoNN estimator.
    random_seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    estimated : dict
        Dictionary with the following keys:
            - "PR": participation ratio (spectral effective rank)
            - "eRank": exponential rank
            - "srank": stable rank
            - "mle_mean", "mle_lo", "mle_hi": MLE estimate and CI
            - "twonn_est": TwoNN estimate
            - "spectral_rank": chosen spectral rank (if spectral mode)
            - "distance_space": which distance space was used
    """
    estimated = {
        "PR": None,
        "eRank": None,
        "srank": None,
        "mle_mean": None,
        "mle_lo": None,
        "mle_hi": None,
        "twonn_est": None,
        "distance_space": distance_space,
    }

    # load data from disk
    if isinstance(data_path_or_simulated, str) or isinstance(
        data_path_or_simulated, Path
    ):
        data_path_or_simulated = Path(data_path_or_simulated)
        suffix = data_path_or_simulated.suffix.lower()
        if suffix == ".npz":
            try:
                # Try to load as scipy sparse matrix first
                X_loaded = sparse.load_npz(data_path_or_simulated)
            except (KeyError, ValueError):
                # If that fails, try to load as numpy array from .npz
                print(
                    f"[INFO] Loading {data_path_or_simulated} as numpy array from .npz"
                )
                npz_data = np.load(data_path_or_simulated)
                # Get the first (and likely only) array from the .npz file
                array_name = list(npz_data.keys())[0]
                X_loaded = npz_data[array_name]
                print(f"[INFO] Loaded array '{array_name}' with shape {X_loaded.shape}")
                npz_data.close()
        elif suffix == ".npy":
            X_loaded = np.load(data_path_or_simulated)
        else:
            raise ValueError("--input must be .npz or .npy")
        X_sparse, flipped = orient_rows_as_samples(X_loaded)
    # load data from memory (dense numpy.array or any scipy.sparse)
    elif isinstance(data_path_or_simulated, np.ndarray) or sparse.issparse(
        data_path_or_simulated
    ):
        X_sparse, flipped = orient_rows_as_samples(data_path_or_simulated)
    else:
        raise ValueError(
            "data_path_or_simulated must be a string/Path, numpy.ndarray, or scipy.sparse matrix"
        )
    N_eff, L_eff = X_sparse.shape
    print(
        f"Working with matrix (rows=samples, cols=features) = {X_sparse.shape}. distance_space={distance_space}"
    )

    # Data validation and preprocessing
    if N_eff < 2:
        print(f"[ERROR] Need at least 2 samples, got {N_eff}")
        return estimated
    if L_eff < 2:
        print(f"[ERROR] Need at least 2 features, got {L_eff}")
        return estimated

    # Check for NaN or infinite values
    if sparse.issparse(X_sparse):
        if np.any(np.isnan(X_sparse.data)) or np.any(np.isinf(X_sparse.data)):
            print(f"[WARN] Found NaN or infinite values, removing them")
            valid_mask = np.isfinite(X_sparse.data)
            X_sparse.data = X_sparse.data[valid_mask]
            X_sparse.indices = X_sparse.indices[valid_mask]
            X_sparse.indptr = np.concatenate(
                [
                    [0],
                    np.cumsum(
                        np.bincount(X_sparse.indices, minlength=X_sparse.shape[0])
                    ),
                ]
            )
    else:
        if np.any(np.isnan(X_sparse)) or np.any(np.isinf(X_sparse)):
            print(f"[WARN] Found NaN or infinite values, setting them to 0")
            X_sparse = np.nan_to_num(X_sparse, nan=0.0, posinf=0.0, neginf=0.0)
            X_sparse = sparse.csr_matrix(X_sparse)

    # build indicator and square
    B, A2 = build_indicator_and_square(X_sparse)

    # ---- Spectral metrics ----
    # Pairwise covariance eigendecomp + metrics (always compute for reporting)
    print(f"Computing spectral metrics...")
    lam, V = pairwise_cov_eigendecomp(X_sparse)
    estimated["PR"], estimated["eRank"], estimated["srank"] = (
        spectral_effective_ranks_from_lam(lam)
    )
    print(
        f"PR (pairwise): {estimated['PR']:.2f}, eRank: {estimated['eRank']:.2f}, stable-rank: {estimated['srank']:.2f}, "
    )

    # ---- Estimate Hausdorff dimension in spectral space (after spectral metrics) ----
    if distance_space == "spectral":
        print(f"Computing spectral coordinates...")
        # Choose spectral rank and compute spectral coordinates Z
        estimated["spectral_rank"] = choose_spectral_rank(
            estimated["PR"], estimated["eRank"], L_eff, r_min=2
        )
        Z = spectral_coordinates_from_missing(
            X_sparse, V, estimated["spectral_rank"], ridge=1e-6
        )
        # Run MLE / TwoNN in Z-space
        print(f"Computing MLE / TwoNN in Z-space...")
        estimated["mle_mean"], estimated["mle_lo"], estimated["mle_hi"] = (
            mle_intrinsic_dim_from_Z(
                Z, k1=10, k2=30, max_points=max_points_mle, seed=random_seed
            )
        )
        estimated["twonn_est"], _ = twonn_intrinsic_dim_from_Z(
            Z, max_points=max_points_twonn, seed=random_seed
        )

        print(f"Spectral rank r (chosen): {estimated['spectral_rank']}")
        print(
            f"MLE [{distance_space}]: {estimated['mle_mean']:.2f} 95% CI [{estimated['mle_lo']:.2f},{estimated['mle_hi']:.2f}]"
        )
        print(f"TwoNN [{distance_space}]: {estimated['twonn_est']:.2f}")

    # ---- Estimate Hausdorff dimension in raw feature space (after spectral metrics) ----
    elif distance_space in ["raw", "bagged"]:
        print(f"Computing MLE / TwoNN in {distance_space} feature space...")
        # ---- Auto-suggestion for overlap/bagging params (only used by raw/bagged distance) ----
        m_eff = X_sparse.nnz / float(max(1, N_eff * L_eff))

        def suggest_params(m, L):
            # 目标：每袋期望重叠 >= target_overlap
            # 建议表（和我们之前讨论一致）
            if m <= 0.05:
                return 1, min(L, 50), 60
            if m <= 0.10:
                return 2, min(L, 40), 40
            if m <= 0.20:
                return 3, min(L, 25), 30
            if m <= 0.50:
                return 5, min(L, 20), 20
            return 5, min(L, 20), 15

        s_min_overlap, s_bag_size, s_num_bags = suggest_params(m_eff, L_eff)
        if min_overlap is None:
            min_overlap = s_min_overlap
        if bag_size is None:
            bag_size = s_bag_size
        if num_bags is None:
            num_bags = s_num_bags

        # Raw / Bagged distance directly in original feature space
        use_bag = distance_space == "bagged"
        print(f"Computing MLE in {distance_space} feature space...")
        estimated["mle_mean"], estimated["mle_lo"], estimated["mle_hi"] = (
            mle_intrinsic_dim(
                X_sparse,
                B,
                A2,
                max_points=max_points_mle,
                seed=random_seed,
                min_overlap=min_overlap,
                bagging=use_bag,
                num_bags=num_bags,
                bag_size=bag_size,
            )
        )
        print(f"Computing TwoNN in {distance_space} feature space...")
        estimated["twonn_est"], _ = twonn_intrinsic_dim(
            X_sparse,
            B,
            A2,
            max_points=max_points_twonn,
            seed=random_seed,
            min_overlap=min_overlap,
            bagging=use_bag,
            num_bags=num_bags,
            bag_size=bag_size,
        )

        mode = "bagged" if (distance_space == "bagged") else "raw"
        print(
            f"MLE [{distance_space}]: {estimated['mle_mean']:.2f} 95% CI [{estimated['mle_lo']:.2f},{estimated['mle_hi']:.2f}]"
        )
        print(f"TwoNN [{distance_space}]: {estimated['twonn_est']:.2f}")

    elif distance_space is None:
        print(f"No MLE/TwoNN estimators computed.")
        estimated["mle_mean"] = estimated["mle_lo"] = estimated["mle_hi"] = estimated[
            "twonn_est"
        ] = np.nan

    # print out
    print(
        f"PR (pairwise): {estimated['PR']:.2f}, eRank: {estimated['eRank']:.2f}, stable-rank: {estimated['srank']:.2f}, "
        f"MLE [{distance_space}]: {estimated['mle_mean']:.2f} 95% CI [{estimated['mle_lo']:.2f},{estimated['mle_hi']:.2f}], "
        f"TwoNN [{distance_space}]: {estimated['twonn_est']:.2f}, "
        + f"Spectral rank r (chosen): {estimated['spectral_rank']}"
        if distance_space == "spectral"
        else ""
    )

    return estimated


# estimate()


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--L", type=int, default=50)  # number of FEATURES (cols)
    ap.add_argument("--N", type=int, default=100000)  # number of SAMPLES (rows)
    ap.add_argument("--n", type=int, default=15)  # manifold dimension (sim)
    ap.add_argument("--m", type=float, default=0.05)  # fraction observed
    ap.add_argument("--max-points-mle", type=int, default=300)
    ap.add_argument("--max-points-twonn", type=int, default=500)
    ap.add_argument(
        "--min-overlap",
        type=lambda v: None if v.lower() == "none" else int(v),
        default=None,
    )
    # ap.add_argument("--bagging", action="store_true", default=False)
    ap.add_argument(
        "--num-bags",
        type=lambda v: None if v.lower() == "none" else int(v),
        default=None,
    )
    ap.add_argument(
        "--bag-size",
        type=lambda v: None if v.lower() == "none" else int(v),
        default=None,
    )
    ap.add_argument("--input", type=str, default=None)
    ap.add_argument(
        "--distance-space", choices=["spectral", "raw", "bagged"], default="spectral"
    )  # when estimating Hausdorff dimension, compute distance between two points with raw Euclidean distance, bagged distance, or spectral space distance
    args = ap.parse_args()

    for args.m in [0.05, 0.1, 0.2]:

        if args.input is None:
            X_sparse, B, A2, Q, Z = simulate_linear_manifold(
                L=args.L, N=args.N, n=args.n, m=args.m, seed=args.seed
            )
            print(
                f"ground truth (simulation): samples N={args.N:,}, features L={args.L}, true dim={args.n}, observe rate={args.m:.0%}, nnz={X_sparse.nnz:,}"
            )

        for distance_space in ["spectral", "raw", "bagged"]:
            estimated = estimate(
                data_path_or_simulated=X_sparse,
                min_overlap=args.min_overlap,
                bag_size=args.bag_size,
                num_bags=args.num_bags,
                distance_space=distance_space,
                max_points_mle=args.max_points_mle,
                max_points_twonn=args.max_points_twonn,
                random_seed=args.seed,
            )
            print("\n\n")


if __name__ == "__main__":
    main()
