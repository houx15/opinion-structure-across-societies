"""spectral_bootstrap — bootstrap intervals for PR / eRank (Fig 3 b, c).

Resamples rows (respondents / users) of each source's ``<year>.csr.npz``
with replacement and recomputes the pairwise-available covariance spectrum
exactly as ``dimensions.pairwise_cov_eigendecomp`` does (same ridge). A resample is
expressed as integer row weights (multinomial counts), which is identical
to duplicating rows but avoids copying the matrix every replicate.

Sources with several yearly matrices (e.g. weibo) are resampled within each
year independently and averaged across years per replicate, matching the
point estimate ``mean(summary[metric])`` used by plotting.figures.

Reads data/dimension/csr/<stem>/*.csr.npz and writes
data/dimension/results/<stem>-none-bootstrap.json:

    python -m analysis.dimension.spectral_bootstrap all            # every Fig 3 source found
    python -m analysis.dimension.spectral_bootstrap run anes       # one source
    python -m analysis.dimension.spectral_bootstrap run twitter-us --npz_dir PATH

``--matrix corr`` uses the pairwise-available correlation matrix instead
(robustness check) and writes <stem>-none-corr-bootstrap.json.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from analysis.dimension.dimensions import cov_to_corr, regularize, spectral_effective_ranks_from_lam
from common.paths import CSR_DIR, DIMENSION_RESULTS_DIR

NPZ_ROOT = CSR_DIR
OUT_DIR = DIMENSION_RESULTS_DIR
# Same resample count as paper_stats.DEFAULT_N_BOOT (kept local so this
# script runs on the clusters without matplotlib).
N_BOOT = 2000

# Summary stems behind every Fig 3 panel (main + robustness), see
# plotting.figures._SPECTRAL_STEM_OVERRIDE for the social-media names.
FIG3_STEMS: Tuple[str, ...] = (
    "anes", "anes_media", "wvs", "wvs_media",
    "evs", "evs_media", "evs_resample2", "evs_media_resample2", "en_evs", "en_evs_media", "eu_nen_evs",
    "twitter-us", "twitter-eu", "twitter-en", "twitter-eu_nen", "weibo",
)


def npz_folder_for(stem: str) -> Path:
    """Map a summary stem to the folder dimension_pipeline wrote its npz to."""
    return NPZ_ROOT / stem


def weighted_pairwise_cov(X: sparse.csr_matrix, B: sparse.csr_matrix, w: np.ndarray) -> np.ndarray:
    """Pairwise-available covariance with integer row weights ``w``.

    With ``w = 1`` this is exactly ``dimensions.pairwise_cov_eigendecomp``'s
    matrix; with multinomial ``w`` it equals that matrix on the resampled rows.
    """
    Xw = sparse.csr_matrix(X.multiply(w[:, None]))
    Bw = sparse.csr_matrix(B.multiply(w[:, None]))
    counts = (B.T @ Bw).toarray().astype(float)
    cross = (X.T @ Xw).toarray().astype(float)
    col_sum = np.asarray(Xw.sum(axis=0)).ravel().astype(float)
    col_cnt = np.asarray(Bw.sum(axis=0)).ravel().astype(float)
    mu = np.divide(col_sum, col_cnt, out=np.zeros_like(col_sum), where=col_cnt > 0)
    cov = np.zeros_like(cross)
    mask = counts > 1
    denom = np.maximum(counts - 1.0, 1.0)
    cov[mask] = (cross[mask] - counts[mask] * np.outer(mu, mu)[mask]) / denom[mask]
    return 0.5 * (cov + cov.T)


def spectral_metrics(X, B, w, standardize: bool = False) -> Tuple[float, float]:
    cov = weighted_pairwise_cov(X, B, w)
    lam = np.linalg.eigvalsh(regularize(cov_to_corr(cov) if standardize else cov))
    pr, erank, _ = spectral_effective_ranks_from_lam(lam)
    return pr, erank


def bootstrap_matrices(
    matrices: Sequence[sparse.csr_matrix], n_boot: int, seed: int, standardize: bool = False,
) -> Dict[str, np.ndarray]:
    """Point estimates and replicates, averaged across the yearly matrices."""
    rng = np.random.default_rng(seed)
    point = np.zeros(2)
    reps = np.zeros((n_boot, 2))
    for X in matrices:
        X = sparse.csr_matrix(X, dtype=float)
        B = X.copy()
        B.data = np.ones_like(B.data)
        n = X.shape[0]
        point += spectral_metrics(X, B, np.ones(n), standardize)
        for b in range(n_boot):
            w = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(float)
            reps[b] += spectral_metrics(X, B, w, standardize)
    k = len(matrices)
    return {"point": point / k, "reps": reps / k}


def run(
    stem: str,
    npz_dir: Optional[str] = None,
    out_dir: str = str(OUT_DIR),
    n_boot: int = N_BOOT,
    seed: int = 20261008,
    alpha: float = 0.05,
    matrix: str = "cov",
) -> Path:
    """Bootstrap one source and write ``<out_dir>/<stem>-none[-corr]-bootstrap.json``."""
    if matrix not in ("cov", "corr"):
        raise ValueError(f"matrix must be 'cov' or 'corr', got {matrix!r}")
    tag = "none-corr" if matrix == "corr" else "none"
    folder = Path(npz_dir) if npz_dir else npz_folder_for(stem)
    files = sorted(folder.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz files in {folder}")
    print(f"[{stem}] {len(files)} matrix file(s) from {folder}, n_boot={n_boot}, matrix={matrix}")
    res = bootstrap_matrices([sparse.load_npz(f) for f in files], n_boot, seed, standardize=matrix == "corr")

    out: Dict[str, object] = {
        "stem": stem, "matrix": matrix, "files": [f.name for f in files], "n_boot": n_boot,
        "seed": seed, "alpha": alpha,
        "unit": "rows (respondents / users) resampled with replacement within each file",
    }
    for j, metric in enumerate(("PR", "eRank")):
        col = res["reps"][:, j]
        out[metric] = float(res["point"][j])
        out[f"{metric}_lo"] = float(np.quantile(col, alpha / 2))
        out[f"{metric}_hi"] = float(np.quantile(col, 1 - alpha / 2))
        out[f"{metric}_se"] = float(col.std(ddof=1))
        out[f"{metric}_reps"] = col.tolist()
        print(f"  {metric} = {out[metric]:.3f}  95% CI [{out[f'{metric}_lo']:.3f}, {out[f'{metric}_hi']:.3f}]")

    # The point estimate must reproduce the pipeline's summary when present.
    summary = Path(out_dir) / f"{stem}-{tag}-summary.json"
    if summary.exists():
        with open(summary) as f:
            ref = json.load(f)
        for metric in ("PR", "eRank"):
            expected = float(np.mean(ref[metric]))
            if not np.isclose(out[metric], expected, rtol=1e-6):
                print(f"  [WARN] {metric} {out[metric]:.6f} != summary {expected:.6f}")

    path = Path(out_dir) / f"{stem}-{tag}-bootstrap.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  saved {path}")
    return path


def run_all(stems: Sequence[str] = FIG3_STEMS, **kwargs) -> List[Path]:
    """Bootstrap every stem whose npz folder exists here (skip the rest)."""
    paths = []
    for stem in stems:
        if not npz_folder_for(stem).exists() and "npz_dir" not in kwargs:
            print(f"[{stem}] skipped: {npz_folder_for(stem)} not found on this machine")
            continue
        paths.append(run(stem, **kwargs))
    return paths


if __name__ == "__main__":
    import fire

    fire.Fire({"run": run, "all": run_all})
