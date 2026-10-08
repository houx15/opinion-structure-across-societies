"""Tests for spectral_bootstrap: weighted resampling == pipeline covariance."""
import json

import numpy as np
from scipy import sparse

from analysis.dimension import spectral_bootstrap as sb
from analysis.dimension.dimensions import pairwise_cov_eigendecomp


def _sparse_opinions(n=400, L=5, missing=0.3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, L)) @ rng.normal(size=(L, L))
    X[rng.random((n, L)) < missing] = np.nan
    r, c = np.where(np.isfinite(X))
    return sparse.csr_matrix((X[r, c], (r, c)), shape=X.shape)


def _indicator(X):
    B = X.copy()
    B.data = np.ones_like(B.data)
    return B


def test_unit_weights_reproduce_pipeline_eigenvalues():
    X = _sparse_opinions()
    lam_pipeline, _ = pairwise_cov_eigendecomp(X)
    lam_ours = np.linalg.eigvalsh(sb.weighted_pairwise_cov(X, _indicator(X), np.ones(X.shape[0])))
    np.testing.assert_allclose(lam_ours, lam_pipeline, rtol=1e-10, atol=1e-12)


def test_integer_weights_equal_duplicated_rows():
    X = _sparse_opinions(seed=1)
    rng = np.random.default_rng(2)
    idx = rng.integers(0, X.shape[0], size=X.shape[0])
    w = np.bincount(idx, minlength=X.shape[0]).astype(float)
    lam_dup, _ = pairwise_cov_eigendecomp(X[idx])
    lam_w = np.linalg.eigvalsh(sb.weighted_pairwise_cov(X, _indicator(X), w))
    np.testing.assert_allclose(lam_w, lam_dup, rtol=1e-10, atol=1e-12)


def test_run_writes_interval_containing_point(tmp_path):
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    for year, seed in ((2019, 3), (2020, 4)):
        sparse.save_npz(npz_dir / f"{year}.csr.npz", _sparse_opinions(seed=seed))
    path = sb.run("toy", npz_dir=str(npz_dir), out_dir=str(tmp_path), n_boot=200, seed=0)
    out = json.loads(path.read_text())
    assert out["files"] == ["2019.csr.npz", "2020.csr.npz"]
    for metric in ("PR", "eRank"):
        assert out[f"{metric}_lo"] < out[metric] < out[f"{metric}_hi"]
        assert len(out[f"{metric}_reps"]) == 200
    assert out["PR"] <= out["eRank"]
