"""Robustness of PR / eRank to sample size, missingness and true dimensionality.

Simulation study of SI "Robustness of estimated effective dimensionality
under variation in sample size, missingness, and true effective
dimensionality". For every cell (missingness pattern, N, r, n) and replicate:

* complete data on an n-dimensional linear subspace of R^L:
  A ~ N(0, 1)^{L x n}, Q = orthonormal basis of A (QR), Z ~ N(0, 1)^{N x n},
  X_full = Z Q^T (population covariance Q Q^T: PR = eRank = n);
* entrywise missingness at level r (observation probability m = 1 - r):
    uniform     every entry observed with probability m;
    rows_cols   heterogeneous: P_ij = c * a_i * b_j clipped to [0, 1], with
                user propensities a_i ~ Exponential(1) (many users observe
                little, as on social media), item propensities b_j evenly
                spread over [0.5, 1.5] in random order, and c chosen so that
                mean(P) = m;
* PR and eRank from the pairwise-available covariance exactly as for the
  real data (analysis/dimension/dimensions.py: pairwise_cov_eigendecomp,
  spectral_effective_ranks_from_lam).

`report` averages the replicates of each cell (M_{N,r,n}) and computes, per
missingness pattern, the statistics of the SI text:
* sample size: CV of M across N at fixed (r, n); mean and max over (r, n);
* missingness: CV of M across r at fixed (N, n); mean and max over (N, n);
  plus the direction of change from the lowest to the highest missingness;
* true dimensionality, within (N, r): Spearman rho(n, M) and
  MAE = mean |M - n|; pooled over all cells: Pearson gamma and Spearman rho.
SD in the CV is the sample SD (ddof = 1).

    python -m robustness.missingness_sensitivity simulate --workers 8      # -> outputs/reports/<date>_missingness_sensitivity_L20/cells.csv
    python -m robustness.missingness_sensitivity simulate --L 9 --n_values 1,2,3,4,5,6,7,8
    python -m robustness.missingness_sensitivity report outputs/reports/<date>_missingness_sensitivity_L20/cells.csv

Each matrix takes < 1 s (N = 10^6: ~0.5 GB memory); the default grid
(2 patterns x 4 N x 6 r x 10 n x 5 replicates) runs in a few minutes on a
laptop. `report` writes the stats file, the tables and the SI figures.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import fire
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

from analysis.dimension.dimensions import (
    pairwise_cov_eigendecomp,
    spectral_effective_ranks_from_lam,
)
from common.paths import FIGURE_DIR, STATS_DIR, date_prefix, display, report_dir

SAMPLE_SIZES = (1_000, 10_000, 100_000, 1_000_000)
MISSINGNESS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
PATTERNS = ("uniform", "rows_cols")
PATTERN_ID = {"uniform": 0, "rows_cols": 1}
METRICS = ("PR", "eRank")
BASE_SEED = 20261008


def _ints(values) -> List[int]:
    if isinstance(values, str):
        return [int(v) for v in values.split(",") if v.strip()]
    if isinstance(values, (int, np.integer)):
        return [int(values)]
    return [int(v) for v in values]


def _floats(values) -> List[float]:
    if isinstance(values, str):
        return [float(v) for v in values.split(",") if v.strip()]
    if isinstance(values, (float, int)):
        return [float(values)]
    return [float(v) for v in values]


def observation_probabilities(pattern: str, N: int, L: int, m: float, rng):
    """Observation probability of every entry (scalar for the uniform pattern)."""
    if pattern == "uniform":
        return m
    if pattern != "rows_cols":
        raise ValueError(f"unknown missingness pattern {pattern!r}")
    a = rng.exponential(1.0, size=N)
    b = rng.permutation(np.linspace(0.5, 1.5, L))
    base = np.outer(a, b)
    scale = m
    for _ in range(50):                       # rescale so mean(P) = m after clipping
        P = np.minimum(scale * base, 1.0)
        mean = P.mean()
        if abs(mean - m) < 1e-5:
            break
        scale *= m / mean
    return P


def simulate_cell(pattern: str, N: int, r: float, n: int, L: int, rep: int,
                  base_seed: int = BASE_SEED) -> Dict:
    """One synthetic matrix and its PR / eRank (pairwise-available covariance)."""
    seq = np.random.SeedSequence([base_seed, PATTERN_ID[pattern], N, int(round(r * 1000)), n, L, rep])
    rng = np.random.default_rng(seq)
    Q, _ = np.linalg.qr(rng.normal(size=(L, n)))
    Z = rng.normal(size=(N, n))
    X_full = Z @ Q.T
    P = observation_probabilities(pattern, N, L, 1.0 - r, rng)
    mask = rng.random(size=(N, L)) < P
    rows, cols = np.nonzero(mask)
    X = sparse.csr_matrix((X_full[rows, cols], (rows, cols)), shape=(N, L))
    del X_full, mask, Z

    B = X.copy()
    B.data = np.ones_like(B.data)
    pair_counts = (B.T @ B).toarray()
    lam, _ = pairwise_cov_eigendecomp(X)
    PR, eRank, srank = spectral_effective_ranks_from_lam(lam)
    return {
        "pattern": pattern, "L": L, "N": N, "missingness": r, "n": n, "rep": rep,
        "observed_share": X.nnz / (N * L),
        "min_pair_count": int(pair_counts[np.triu_indices(L, 1)].min()),
        "negative_eigenvalues": int((lam < 0).sum()),
        "PR": PR, "eRank": eRank, "srank": srank,
    }


def _run(args):
    return simulate_cell(*args)


def simulate(L: int = 20, n_values="1,2,3,4,5,6,7,8,9,10", sample_sizes=SAMPLE_SIZES,
             missingness=MISSINGNESS, patterns=PATTERNS, reps: int = 5,
             workers: Optional[int] = None, out: Optional[str] = None,
             base_seed: int = BASE_SEED) -> Path:
    """Simulate every (pattern, N, r, n, rep) cell; writes cells.csv (one row per replicate)."""
    n_values, sample_sizes, missingness = _ints(n_values), _ints(sample_sizes), _floats(missingness)
    patterns = [patterns] if isinstance(patterns, str) else list(patterns)
    if max(n_values) > L:
        raise ValueError(f"true dimensionality n must be <= L = {L}")
    jobs = [(p, N, r, n, L, rep, base_seed)
            for p, N, r, n, rep in product(patterns, sample_sizes, missingness, n_values, range(reps))]
    # largest matrices first so the pool stays busy at the end
    jobs.sort(key=lambda j: -j[1])
    workers = workers or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    out_path = Path(out) if out else report_dir(f"missingness_sensitivity_L{L}") / "cells.csv"
    print(f"[missingness] {len(jobs)} simulations, L={L}, n={n_values}, N={sample_sizes}, "
          f"r={missingness}, patterns={patterns}, reps={reps}, workers={workers}", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(_run, jobs, chunksize=1), 1):
            rows.append(row)
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done", flush=True)
                pd.DataFrame(rows).to_csv(out_path, index=False)
    df = pd.DataFrame(rows).sort_values(["pattern", "N", "missingness", "n", "rep"])
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    return out_path


# -- statistics --------------------------------------------------------------------


def cell_means(cells: pd.DataFrame) -> pd.DataFrame:
    keys = ["pattern", "L", "N", "missingness", "n"]
    return (cells.groupby(keys)[["PR", "eRank", "srank", "observed_share", "min_pair_count"]]
            .mean().reset_index())


def _cv(x: pd.Series) -> float:
    return float(x.std(ddof=1) / x.mean())


def sensitivity_tables(means: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """CV across N at fixed (r, n) and across r at fixed (N, n), per pattern and metric."""
    out = {}
    for name, across, fixed in (("cv_sample_size", "N", ["missingness", "n"]),
                                ("cv_missingness", "missingness", ["N", "n"])):
        rows = []
        for (pattern, *key), g in means.groupby(["pattern"] + fixed):
            row = {"pattern": pattern, **dict(zip(fixed, key))}
            for m in METRICS:
                row[f"CV_{m}"] = _cv(g.sort_values(across)[m])
            rows.append(row)
        out[name] = pd.DataFrame(rows)
    return out


def missingness_direction(means: pd.DataFrame) -> pd.DataFrame:
    """Change of each metric from the lowest to the highest missingness at fixed (N, n)."""
    lo, hi = means["missingness"].min(), means["missingness"].max()
    rows = []
    for (pattern, N, n), g in means.groupby(["pattern", "N", "n"]):
        g = g.set_index("missingness")
        row = {"pattern": pattern, "N": N, "n": n}
        for m in METRICS:
            row[f"{m}_at_{lo:.2f}"] = g.loc[lo, m]
            row[f"{m}_at_{hi:.2f}"] = g.loc[hi, m]
            row[f"{m}_rel_change"] = (g.loc[hi, m] - g.loc[lo, m]) / g.loc[lo, m]
        rows.append(row)
    return pd.DataFrame(rows)


def true_dimension_tables(means: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    within = []
    for (pattern, N, r), g in means.groupby(["pattern", "N", "missingness"]):
        row = {"pattern": pattern, "N": N, "missingness": r}
        for m in METRICS:
            row[f"spearman_{m}"] = float(spearmanr(g["n"], g[m]).statistic)
            row[f"MAE_{m}"] = float((g[m] - g["n"]).abs().mean())
        within.append(row)
    pooled = []
    for pattern, g in means.groupby("pattern"):
        row = {"pattern": pattern, "cells": len(g)}
        for m in METRICS:
            row[f"pearson_{m}"] = float(pearsonr(g["n"], g[m]).statistic)
            row[f"spearman_{m}"] = float(spearmanr(g["n"], g[m]).statistic)
            row[f"MAE_{m}"] = float((g[m] - g["n"]).abs().mean())
        pooled.append(row)
    return {"true_dim_within": pd.DataFrame(within), "true_dim_pooled": pd.DataFrame(pooled)}


def summary_text(means: pd.DataFrame, tables: Dict[str, pd.DataFrame], source: str) -> str:
    L = int(means["L"].iloc[0])
    lines = [
        "Robustness of PR / eRank to sample size, missingness and true dimensionality",
        f"Input: {source}",
        f"L = {L}; N = {[int(x) for x in sorted(means['N'].unique())]}; missingness = "
        f"{[round(float(x), 2) for x in sorted(means['missingness'].unique())]}; "
        f"n = {[int(x) for x in sorted(means['n'].unique())]}; "
        f"metric values = mean over replicates of each cell.",
        "CV = sample SD / mean. Patterns: uniform = entrywise missingness; rows_cols = heterogeneous "
        "user (Exponential) x item (0.5-1.5) observation propensities, same overall level.",
    ]
    for pattern in sorted(means["pattern"].unique()):
        lines += ["", f"=== pattern: {pattern}"]
        cvn = tables["cv_sample_size"].query("pattern == @pattern")
        cvr = tables["cv_missingness"].query("pattern == @pattern")
        dirn = tables["missingness_direction"].query("pattern == @pattern")
        within = tables["true_dim_within"].query("pattern == @pattern")
        pooled = tables["true_dim_pooled"].query("pattern == @pattern").iloc[0]
        lo, hi = means["missingness"].min(), means["missingness"].max()
        lines.append("Sample size (CV across N at fixed r, n):")
        for m in METRICS:
            lines.append(f"  {m:5}  mean CV = {cvn[f'CV_{m}'].mean():.3f}  max CV = {cvn[f'CV_{m}'].max():.3f}"
                         f"  ({len(cvn)} (r, n) cells)")
        lines.append("Missingness (CV across r at fixed N, n):")
        for m in METRICS:
            lines.append(f"  {m:5}  mean CV = {cvr[f'CV_{m}'].mean():.3f}  max CV = {cvr[f'CV_{m}'].max():.3f}"
                         f"  ({len(cvr)} (N, n) cells)")
        for m in METRICS:
            ch = dirn[f"{m}_rel_change"]
            lines.append(f"  {m:5}  from r = {lo:.0%} to {hi:.0%}: lower in {int((ch < 0).sum())} of {len(ch)} "
                         f"(N, n) cells; relative change median {ch.median():+.1%} "
                         f"[min {ch.min():+.1%}, max {ch.max():+.1%}]")
        lines.append("True dimensionality, within each (N, r):")
        for m in METRICS:
            s, a = within[f"spearman_{m}"], within[f"MAE_{m}"]
            lines.append(f"  {m:5}  Spearman rho {s.min():.3f} to {s.max():.3f}; "
                         f"MAE {a.min():.2f} to {a.max():.2f} (mean {a.mean():.2f})  ({len(within)} cells)")
        lines.append("True dimensionality, pooled over all cells:")
        for m in METRICS:
            lines.append(f"  {m:5}  Pearson gamma = {pooled[f'pearson_{m}']:.3f}  "
                         f"Spearman rho = {pooled[f'spearman_{m}']:.3f}  MAE = {pooled[f'MAE_{m}']:.2f}")
        lines.append("By sample size N (missingness CV across r; relative change from lowest to highest r;"
                     " within-(N, r) Spearman rho and MAE):")
        for N, g in cvr.groupby("N"):
            dN = dirn[dirn["N"] == N]
            wN = within[within["N"] == N]
            parts = []
            for m in METRICS:
                parts.append(f"{m}: CV mean {g[f'CV_{m}'].mean():.3f} max {g[f'CV_{m}'].max():.3f}, "
                             f"change median {dN[f'{m}_rel_change'].median():+.1%}, "
                             f"rho min {wN[f'spearman_{m}'].min():.3f}, MAE {wN[f'MAE_{m}'].min():.2f}-"
                             f"{wN[f'MAE_{m}'].max():.2f}")
            lines.append(f"  N = {int(N):>9,}  " + " | ".join(parts))
        pairs = means.query("pattern == @pattern").pivot_table(
            index="N", columns="missingness", values="min_pair_count")
        lines.append("Min respondents per item pair (rows N, columns missingness):")
        lines.append("  " + pairs.round(0).astype(int).to_string().replace("\n", "\n  "))
        worst = means.query("pattern == @pattern").nsmallest(1, "min_pair_count").iloc[0]
        lines.append(f"Sparsest cell: N = {int(worst['N']):,}, r = {worst['missingness']:.0%}: "
                     f"min respondents per item pair = {worst['min_pair_count']:.1f}")
    return "\n".join(lines) + "\n"


def plot(means: pd.DataFrame, path: Path, pattern: str = "uniform") -> None:
    import matplotlib.pyplot as plt

    d = means.query("pattern == @pattern")
    Ns = sorted(d["N"].unique())
    ns = sorted(d["n"].unique())
    cmap = plt.get_cmap("viridis")
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 9})
    fig, axes = plt.subplots(len(METRICS), len(Ns), figsize=(2.1 * len(Ns), 2.2 * len(METRICS)),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for i, m in enumerate(METRICS):
        for j, N in enumerate(Ns):
            ax = axes[i, j]
            for k, n in enumerate(ns):
                g = d[(d["N"] == N) & (d["n"] == n)].sort_values("missingness")
                color = cmap(k / max(len(ns) - 1, 1))
                ax.plot(100 * g["missingness"], g[m], marker="o", ms=2.5, lw=1.2, color=color,
                        label=f"n = {n}" if (i == 0 and j == len(Ns) - 1) else None)
                ax.axhline(n, color=color, lw=0.6, ls=":", alpha=0.7)
            if i == 0:
                ax.set_title(f"N = $10^{{{int(np.log10(N))}}}$")
            if j == 0:
                ax.set_ylabel(f"estimated {m}")
            if i == len(METRICS) - 1:
                ax.set_xlabel("missingness (%)")
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, -1].legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def report(cells_csv: str, plot_figure: bool = True) -> Path:
    """Statistics of the SI text from cells.csv; writes tables, stats file and figure."""
    cells_csv = Path(cells_csv)
    cells = pd.read_csv(cells_csv)
    means = cell_means(cells)
    L = int(means["L"].iloc[0])
    tables = sensitivity_tables(means)
    tables["missingness_direction"] = missingness_direction(means)
    tables.update(true_dimension_tables(means))
    out = cells_csv.parent
    means.to_csv(out / "cell_means.csv", index=False)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    text = summary_text(means, tables, display(cells_csv))
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    stats = STATS_DIR / f"{date_prefix()}_supp_missingness_sensitivity_L{L}_stats.txt"
    stats.write_text(text)
    print(text)
    print(f"wrote {stats} and tables in {out}")
    if plot_figure:
        for pattern in sorted(means["pattern"].unique()):
            fig = FIGURE_DIR / f"{date_prefix()}_supp_missingness_sensitivity_L{L}_{pattern}_fig.pdf"
            plot(means, fig, pattern)
            print(f"wrote {fig}")
    return stats


if __name__ == "__main__":
    fire.Fire({"simulate": simulate, "report": report})
