"""Sensitivity of effective dimensionality to the selection of topics.

The pairwise-available covariance of a topic subset is the submatrix of the
full one (each covariance uses the people observed on that pair only), so the
spectrum of any subset follows from the full covariance, rebuilt exactly from
the saved eigendecomposition (``<stem>-<year>.csr-loadings.json``: all K
eigenvalues, negative ones included, and eigenvectors). Rebuilt eight-topic
spectra reproduce robustness/topics8.py to four decimals.

For every society drop ``n_drop`` of its nine topics - the same concepts in the
survey and the platform source (figures.OFFLINE_ONLINE_TOPIC_MAP) - in every
possible way, and combine the societies' subsets independently (9^3 = 729
combinations for one topic, 36^3 = 46,656 for two). For each combination:
PR/K and eRank/K (K = 9 - n_drop) of every source (Weibo: mean over its
yearly matrices, as in the main analysis), the cross-societal range in each
medium, their ratio, and whether the survey-platform contrasts are
bidirectional (U.S. platform above its survey, Chinese and British below).
Point estimates; no bootstrap. Stats Part 15 and the SI figure
(``outputs/figures/<date>_supp_dimension_subsets_fig.pdf``).

    python -m robustness.dimension_subsets
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

METRICS = ("PR", "eRank")


def source_covariances(src: str, matrix: str = "cov") -> Tuple[List[str], List[np.ndarray]]:
    """Topic order and the full covariance of each yearly matrix of ``src``."""
    from common.paths import dimension_results_dir
    from plotting import figures as v4

    stem = v4._SPECTRAL_STEM_OVERRIDE.get(src, src)
    files = sorted(f for f in dimension_results_dir(matrix).glob(f"{stem}-*.csr-loadings.json")
                   if re.fullmatch(re.escape(stem) + r"-\d{4}\.csr-loadings\.json", f.name))
    if src != "weibo":   # Twitter/X spectra are 2021 only; surveys have one file
        files = files[-1:] if src not in v4.SOCIAL_SOURCES else [f for f in files if "-2021." in f.name]
    if not files:
        raise FileNotFoundError(f"no loadings for {src} ({stem})")
    topics, covs = None, []
    for f in files:
        d = json.load(open(f))
        lam = np.array([c["eigenvalue"] for c in d["components"]])
        V = np.array([[c["loadings"][t] for t in d["topics"]] for c in d["components"]]).T
        topics = [str(t) for t in d["topics"]]
        covs.append(V @ np.diag(lam) @ V.T)
    return topics, covs


def spectrum(covs: List[np.ndarray], keep: List[int]) -> Dict[str, float]:
    """PR and eRank of the kept topics, averaged over the yearly matrices."""
    from analysis.dimension.dimensions import regularize, spectral_effective_ranks_from_lam

    vals = [spectral_effective_ranks_from_lam(np.linalg.eigvalsh(regularize(c[np.ix_(keep, keep)])))[:2]
            for c in covs]
    return {"PR": float(np.mean([v[0] for v in vals])), "eRank": float(np.mean([v[1] for v in vals]))}


def subset_values(plotter, n_drop: int) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """{region: {"Survey"|"Social media": {metric: value / K of every subset}}}."""
    from plotting import figures as v4

    out = {}
    k = 9 - n_drop
    for r in plotter.regions:
        mapping = v4.OFFLINE_ONLINE_TOPIC_MAP[r.code]
        concepts = list(mapping)
        sources = {"Survey": (r.survey, lambda c: str(c)), "Social media": (r.social, lambda c: str(mapping[c]))}
        res = {m: {met: [] for met in METRICS} for m in sources}
        cov = {m: source_covariances(src) for m, (src, _) in sources.items()}
        for drop in itertools.combinations(concepts, n_drop):
            keep_c = [c for c in concepts if c not in drop]
            for medium, (src, key) in sources.items():
                topics, covs = cov[medium]
                vals = spectrum(covs, [topics.index(key(c)) for c in keep_c])
                for met in METRICS:
                    res[medium][met].append(vals[met] / k)
        out[r.code] = {m: {met: np.asarray(v) for met, v in d.items()} for m, d in res.items()}
    return out


def combinations(values, codes: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """Per metric: range across societies in each medium, the ratio, and the
    bidirectional indicator, over every combination of the societies' subsets."""
    out = {}
    for met in METRICS:
        grids = {}
        for medium in ("Survey", "Social media"):
            g = np.meshgrid(*[values[c][medium][met] for c in codes], indexing="ij")
            grids[medium] = np.stack([x.ravel() for x in g])     # societies x combinations
        rng = {m: grids[m].max(axis=0) - grids[m].min(axis=0) for m in grids}
        up = grids["Social media"] - grids["Survey"]              # >0: more dimensions online
        others = [i for i, c in enumerate(codes) if c != "US"]
        bidir = (up[codes.index("US")] > 0) & np.all(up[others] < 0, axis=0)
        out[met] = {"survey": rng["Survey"], "social": rng["Social media"],
                    "ratio": rng["Survey"] / rng["Social media"], "bidirectional": bidir}
    return out


def summary(c: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {
        "n": int(c["ratio"].size),
        "survey_median": float(np.median(c["survey"])), "survey_min": float(c["survey"].min()),
        "social_median": float(np.median(c["social"])), "social_max": float(c["social"].max()),
        "ratio_median": float(np.median(c["ratio"])), "ratio_p5": float(np.percentile(c["ratio"], 5)),
        "ratio_min": float(c["ratio"].min()), "share_ratio_gt1": float(np.mean(c["ratio"] > 1)),
        "share_bidirectional": float(np.mean(c["bidirectional"])),
    }


def figure(results: Dict[int, Dict[str, Dict[str, np.ndarray]]], path: Path,
           full: Optional[Dict[str, Dict[str, float]]] = None) -> Path:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Times New Roman"
    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 3.4 * n_rows), squeeze=False)
    titles = {"PR": "Participation ratio", "eRank": "Exponential rank"}
    for row, (n_drop, res) in enumerate(sorted(results.items())):
        for col, met in enumerate(METRICS):
            ax = axes[row, col]
            c = res[met]
            hi = max(c["survey"].max(), c["social"].max()) * 1.05
            bins = np.linspace(0, hi, 60)
            for key, color, label in (("survey", "0.25", "Surveys"), ("social", "#4a90d9", "Social media")):
                ax.hist(c[key], bins=bins, color=color, alpha=0.65, label=label)
                if full:
                    ax.axvline(full[met][key], color=color, linewidth=1.6, linestyle=(0, (4, 2)))
            k = 9 - n_drop
            ax.set_xlabel(f"Cross-societal range of {met}/K (K = {k})", fontsize=11)
            ax.set_ylabel("Topic combinations", fontsize=11)
            ax.set_title(f"{titles[met]}, {n_drop} topic{'s' if n_drop > 1 else ''} dropped per society",
                         fontsize=11, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=10)
    for ax, letter in zip(axes.ravel(), "ABCDEFGH"):
        ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", fontstyle="italic")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def full_ranges(plotter) -> Dict[str, Dict[str, float]]:
    """Range across societies of the nine-topic estimates (rebuilt covariances)."""
    vals = subset_values(plotter, 0)
    codes = [r.code for r in plotter.regions]
    return {met: {key: float(np.ptp([vals[c][medium][met][0] for c in codes]))
                  for key, medium in (("survey", "Survey"), ("social", "Social media"))} for met in METRICS}


def run(task: str = "main", drops: Tuple[int, ...] = (1, 2)) -> Path:
    from common.paths import FIGURE_DIR, date_prefix
    from plotting import figures as v4

    plotter = v4.Plotter(**v4.task_plotter_kwargs(task))
    codes = [r.code for r in plotter.regions]
    full = full_ranges(plotter)
    print("nine topics (rebuilt):", full)
    results = {}
    for n in drops:
        values = subset_values(plotter, n)
        results[n] = combinations(values, codes)
        for met in METRICS:
            print(n, met, {k: round(v, 3) for k, v in summary(results[n][met]).items()})
    return figure(results, FIGURE_DIR / f"{date_prefix()}_supp_dimension_subsets_fig.pdf", full)


if __name__ == "__main__":
    import fire

    fire.Fire(run)
