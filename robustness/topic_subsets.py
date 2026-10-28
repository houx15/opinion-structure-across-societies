"""Sensitivity of the dispersion contrast to the selection of topics.

For every society, drop two of its nine topics - the same two concepts in the
survey and the platform source (figures.OFFLINE_ONLINE_TOPIC_MAP) - giving 36
seven-topic subsets per society; combine the subsets of the three societies
independently (36^3 = 46,656 combinations). For each combination compute the
cross-societal dispersion of mean Fisher z = arctanh|r| (population SD over
the societies, as in the main analysis) in each medium, and their ratio.
Stats Part 13 and the SI figure (``outputs/figures/<date>_supp_topic_subsets_fig.pdf``).

    python -m robustness.topic_subsets
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

N_DROP = 2


def subset_means(plotter, n_drop: int = N_DROP) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """{region: {"Survey"|"Social media": mean z of every topic subset}}, or None
    when a region's survey topics are not in OFFLINE_ONLINE_TOPIC_MAP."""
    from plotting import figures as v4

    out = {}
    for r in plotter.regions:
        mapping = v4.OFFLINE_ONLINE_TOPIC_MAP.get(r.code)
        if mapping is None or set(map(str, mapping)) != set(map(str, v4.RESTRICTED_TOPICS[r.survey])):
            return None
        concepts = list(mapping)
        tables = {}
        for medium, src, key in (("Survey", r.survey, lambda t: t), ("Social media", r.social, lambda t: mapping[t])):
            df = v4.load_pairwise_correlation(src, year=plotter.correlation_year(src))
            z = {frozenset((str(a), str(b))): float(np.arctanh(min(abs(v), 1 - 1e-12)))
                 for a, b, v in zip(df["tid1"], df["tid2"], df["pearson"])}
            tables[medium] = (z, key)
        means: Dict[str, list] = {"Survey": [], "Social media": []}
        for drop in itertools.combinations(concepts, n_drop):
            keep = [c for c in concepts if c not in drop]
            pairs = list(itertools.combinations(keep, 2))
            for medium, (z, key) in tables.items():
                means[medium].append(np.mean([z[frozenset((str(key(a)), str(key(b))))] for a, b in pairs]))
        out[r.code] = {m: np.asarray(v) for m, v in means.items()}
    return out


def dispersions(means: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Dispersion per medium for every combination of the societies' subsets."""
    res = {}
    for medium in ("Survey", "Social media"):
        grid = np.meshgrid(*[means[c][medium] for c in means], indexing="ij")
        res[medium] = np.stack([g.ravel() for g in grid]).std(axis=0)
    res["ratio"] = res["Survey"] / res["Social media"]
    return res


def summary(d: Dict[str, np.ndarray]) -> Dict[str, float]:
    off, on, ratio = d["Survey"], d["Social media"], d["ratio"]
    return {
        "n": int(ratio.size),
        "off_median": float(np.median(off)), "off_min": float(off.min()),
        "on_median": float(np.median(on)), "on_max": float(on.max()),
        "ratio_median": float(np.median(ratio)), "ratio_p5": float(np.percentile(ratio, 5)),
        "ratio_min": float(ratio.min()), "share_ratio_gt1": float(np.mean(ratio > 1)),
        "share_overlap": float(np.mean(on.max() >= off)),
    }


def figure(d: Dict[str, np.ndarray], path: Path, full: Optional[Dict[str, float]] = None) -> Path:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bins = np.linspace(0, max(d["Survey"].max(), d["Social media"].max()) * 1.05, 70)
    for medium, color, label in (("Survey", "0.25", "Surveys (offline)"), ("Social media", "#4a90d9", "Social media (online)")):
        ax.hist(d[medium], bins=bins, color=color, alpha=0.65, label=label)
        if full:
            ax.axvline(full[medium], color=color, linewidth=1.6, linestyle=(0, (4, 2)))
    ax.set_xlabel("Cross-societal dispersion of mean Fisher z (|r|)", fontsize=11)
    ax.set_ylabel("Topic combinations", fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def run(task: str = "main") -> Optional[Path]:
    from common.paths import FIGURE_DIR, date_prefix
    from plotting import figures as v4

    plotter = v4.Plotter(**v4.task_plotter_kwargs(task))
    means = subset_means(plotter)
    if means is None:
        print(f"[topic_subsets] {task}: topics not mapped; skipped")
        return None
    d = dispersions(means)
    full = {}
    for medium, attr in (("Survey", "survey"), ("Social media", "social")):
        full[medium] = float(np.std([np.mean(np.arctanh(np.clip(
            v4.load_pairwise_correlation(getattr(r, attr))["correlation"], 0, 1 - 1e-12))) for r in plotter.regions]))
    print(summary(d))
    return figure(d, FIGURE_DIR / f"{date_prefix()}_supp_topic_subsets_fig.pdf", full)


if __name__ == "__main__":
    import fire

    fire.Fire(run)
