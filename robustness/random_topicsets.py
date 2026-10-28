"""Does the labelling pipeline make platform sources alike by itself?

The same LLM / BERT pipeline scored every labelled topic of a platform, not
only the nine matched to survey items. If the pipeline pulled associations
toward a common level, any set of topics would look alike across platforms.
For every platform source whose correlation file holds more than its nine
analysed topics, take every set of nine of its labelled topics (matched to a
survey item or not), compute the mean |r| over the 36 pairs, and compare the
platforms: the gap in mean |r| between two platforms over all pairs of
random sets, against the gap for the matched topic sets and the gap between
the same societies' surveys. Stats Part 14 and the SI figure
(``outputs/figures/<date>_supp_random_topicsets_fig.pdf``).

    python -m robustness.random_topicsets
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

K = 9


def all_topic_values(src: str) -> Dict[frozenset, float]:
    """|r| of every labelled topic pair in the source's file (pooled years)."""
    from common.paths import CORRELATION_DIR

    df = pd.read_csv(CORRELATION_DIR / f"network_analysis_{src}.csv")
    df = df[df["year"].astype(str) == "average"].dropna(subset=["Correlation (Pearson)"])
    return {frozenset((str(a), str(b))): abs(float(r))
            for a, b, r in zip(df["Topic1"], df["Topic2"], df["Correlation (Pearson)"])}


def set_means(values: Dict[frozenset, float], k: int = K) -> np.ndarray:
    """Mean |r| of every k-topic set whose pairs are all observed."""
    topics = sorted({t for pair in values for t in pair})
    out = []
    for sub in itertools.combinations(topics, k):
        pairs = [frozenset(p) for p in itertools.combinations(sub, 2)]
        if all(p in values for p in pairs):
            out.append(np.mean([values[p] for p in pairs]))
    return np.asarray(out)


def analysis(plotter) -> Optional[Dict[str, object]]:
    """Per platform: random-set means and the matched mean; per platform pair:
    gaps. None when fewer than two platforms hold extra topics."""
    from plotting import figures as v4

    platforms = {}
    for r in plotter.regions:
        values = all_topic_values(v4.parent_source(r.social))
        n_topics = len({t for pair in values for t in pair})
        if n_topics <= K:
            continue
        matched = float(v4.load_pairwise_correlation(r.social)["correlation"].mean())
        survey = float(v4.load_pairwise_correlation(r.survey)["correlation"].mean())
        platforms[r.code] = {"src": r.social, "n_topics": n_topics, "means": set_means(values),
                             "matched": matched, "survey": survey}
    if len(platforms) < 2:
        return None
    gaps = {}
    for a, b in itertools.combinations(platforms, 2):
        g = np.abs(platforms[a]["means"][:, None] - platforms[b]["means"][None, :]).ravel()
        gaps[(a, b)] = {
            "gaps": g,
            "matched": abs(platforms[a]["matched"] - platforms[b]["matched"]),
            "survey": abs(platforms[a]["survey"] - platforms[b]["survey"]),
        }
    return {"platforms": platforms, "gaps": gaps}


def figure(res: Dict[str, object], path: Path, colors: Dict[str, str]) -> Path:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Times New Roman"
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    for code, p in res["platforms"].items():
        ax.hist(p["means"], bins=40, color=colors.get(code, "0.5"), alpha=0.55,
                label=f"{code} platform ({p['n_topics']} topics)")
        ax.axvline(p["matched"], color=colors.get(code, "0.5"), linewidth=1.8, linestyle=(0, (4, 2)))
    ax.set_xlabel("Mean |r| of a nine-topic set", fontsize=11)
    ax.set_ylabel("Topic sets", fontsize=11)
    ax.set_title("Platform level depends on the topics", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    for (a, b), g in res["gaps"].items():
        ax.hist(g["gaps"], bins=60, color="0.55", alpha=0.7, label=f"{a}-{b}, random sets")
        ax.axvline(g["matched"], color="#4a90d9", linewidth=1.8, linestyle=(0, (4, 2)), label="matched topics (online)")
        ax.axvline(g["survey"], color="0.15", linewidth=1.8, label="surveys")
    ax.set_xlabel("Gap in mean |r| between the two platforms", fontsize=11)
    ax.set_ylabel("Pairs of topic sets", fontsize=11)
    ax.set_title("Unmatched topic sets do not converge", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    for ax, letter in zip(axes, "AB"):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", fontstyle="italic")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def run(task: str = "main") -> Optional[Path]:
    from common.paths import FIGURE_DIR, date_prefix
    from plotting import figures as v4

    plotter = v4.Plotter(**v4.task_plotter_kwargs(task))
    res = analysis(plotter)
    if res is None:
        print("[random_topicsets] fewer than two platforms hold extra topics; skipped")
        return None
    for code, p in res["platforms"].items():
        m = p["means"]
        print(code, p["src"], p["n_topics"], len(m), round(m.min(), 3), round(np.median(m), 3), round(m.max(), 3),
              "matched", round(p["matched"], 3))
    for pair, g in res["gaps"].items():
        print(pair, "median gap", round(float(np.median(g["gaps"])), 3), "matched", round(g["matched"], 3),
              "survey", round(g["survey"], 3), "share >= survey gap", round(float(np.mean(g["gaps"] >= g["survey"])), 3))
    return figure(res, FIGURE_DIR / f"{date_prefix()}_supp_random_topicsets_fig.pdf", plotter.country_color)


if __name__ == "__main__":
    import fire

    fire.Fire(run)
