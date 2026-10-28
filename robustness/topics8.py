"""PR / eRank of the eight-topic robustness sources (task robust_12_wvs_us).

ROBUSTNESS ONLY. Every source of the check uses eight topics
(common.topics.TOPICS_8); its pairwise correlations and similarities are read
from the parent source's files (plotting.figures.ROBUSTNESS_PARENT). Here the
spectra are recomputed: each parent respondent / user x topic matrix loses
the dropped topic's column, the eight-column matrices are saved to
data/dimension/csr/<stem>/, and the pairwise-available covariance, PR, eRank
and the respondent / user bootstrap are computed with the estimator of the
main pipeline (analysis.dimension.spectral_bootstrap), written as
data/dimension/results/cov/<stem>-none-summary.json and -bootstrap.json.
Dropping a column leaves the covariances among the other topics unchanged.

Sources whose parent matrices are not on this machine are skipped, so the
same command runs locally (surveys) and on the clusters (social media):

    python -m robustness.topics8 spectral
    python -m robustness.topics8 spectral --stems twitter-us_8,twitter-en_8,weibo_8
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from analysis.dimension import dimension_pipeline as dp
from analysis.dimension import spectral_bootstrap as sb
from analysis.dimension.dimensions import regularize, spectral_effective_ranks_from_lam
from common.paths import CSR_DIR, dimension_results_dir
from common.topics import TOPICS_8

# spectral stem -> (robustness source, parent dataset_type, parent mode)
STEMS: Dict[str, Tuple[str, str, str]] = {
    "wvs_us8": ("wvs_us8", "wvs_us", "us"),
    "wvs_8": ("wvs_8", "wvs", "us"),
    "en_evs_8": ("en_evs_8", "en_evs", "us"),
    "twitter-us_8": ("twitter_8", "twitter", "us"),
    "weibo_8": ("weibo_8", "weibo", "us"),
    "twitter-en_8": ("entwitter_8", "twitter", "en"),
}


def keep_columns(stem: str) -> List[int]:
    """Column indices of the parent matrix that the eight-topic source keeps."""
    src, dataset, mode = STEMS[stem]
    parent_topics = dp._topics_for(dataset, mode)
    keep = [str(t) for t in TOPICS_8[src]]
    missing = set(keep) - set(parent_topics)
    if missing:
        raise KeyError(f"{stem}: topics {missing} not in parent columns {parent_topics}")
    return [parent_topics.index(t) for t in keep]


def subset_matrices(stem: str) -> Optional[Path]:
    """Write the eight-column matrices of ``stem``; None if the parent is absent."""
    _, dataset, mode = STEMS[stem]
    parent = dp.csr_folder(dataset, mode)
    files = sorted(parent.glob("*.csr.npz"))
    if not files:
        print(f"[{stem}] skipped: no parent matrices in {parent}")
        return None
    cols = keep_columns(stem)
    out = CSR_DIR / stem
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        X = sparse.load_npz(f).tocsc()[:, cols].tocsr()
        sparse.save_npz(out / f.name, X)
    print(f"[{stem}] {len(files)} matrix file(s): {parent} -> {out} (columns {cols})")
    return out


def summary(stem: str, folder: Path, matrix: str = "cov") -> Path:
    """<stem>-none-summary.json: PR / eRank / srank of each yearly matrix."""
    res = {"PR": [], "eRank": [], "srank": []}
    for f in sorted(folder.glob("*.csr.npz")):
        X = sparse.csr_matrix(sparse.load_npz(f), dtype=float)
        B = X.copy()
        B.data = np.ones_like(B.data)
        cov = sb.weighted_pairwise_cov(X, B, np.ones(X.shape[0]))
        pr, erank, srank = spectral_effective_ranks_from_lam(np.linalg.eigvalsh(regularize(cov)))
        for key, value in zip(("PR", "eRank", "srank"), (pr, erank, srank)):
            res[key].append(float(value))
    path = dimension_results_dir(matrix) / f"{stem}-none-summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[{stem}] PR {np.mean(res['PR']):.3f}, eRank {np.mean(res['eRank']):.3f} -> {path}")
    return path


def spectral(stems: Sequence[str] = tuple(STEMS), n_boot: int = sb.N_BOOT) -> List[Path]:
    """Matrices, summary and bootstrap for every stem whose parent is here."""
    if isinstance(stems, str):
        stems = [s.strip() for s in stems.split(",") if s.strip()]
    written = []
    for stem in stems:
        folder = subset_matrices(stem)
        if folder is None:
            continue
        written.append(summary(stem, folder))
        written.append(sb.run(stem, npz_dir=str(folder), n_boot=n_boot))
    return written


if __name__ == "__main__":
    import fire

    fire.Fire({"spectral": spectral, "keep_columns": keep_columns})
