"""SI rank / conditioning / projection-residual check on the nine topic centroids.

Works directly on the original 1536-D topic centroid vectors in
``data/embedding/topic_centroids/<src>_topic_centroids.parquet`` (exported by
``embedding_subspace_analysis centroids``: weighted mean of the cached
OpenAI text-embedding-3-small ngram vectors; top-10 TF-IDF ngrams weighted by
frequency for social sources, selected codebook keywords with equal weights
for surveys). No pairwise
similarity CSV is read. Each centroid is unit-normalized; C is the 9 x 1536
matrix of those rows.

1. Rank / conditioning: thin SVD of C.
2. Projection residual: for every target topic t and every subset S of the
   other topics (|S| = 1..8), least-squares projection of c_t onto the span
   of {c_j : j in S}; residual r = ||c_t - proj||, angle = arcsin(r).

Run from the repository root:

    python -m robustness.centroid_rank_check            # tables -> outputs/reports/<date>_centroid_rank_check/
    python -m robustness.centroid_rank_check --plot     # + SI figure (outputs/figures) and stats (outputs/stats)
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from common.paths import EMBEDDING_DIR, FIGURE_DIR, STATS_DIR, date_prefix, display, report_dir


SOURCES: List[str] = ["anes", "evs", "wvs", "twitter", "eutwitter", "weibo"]
SOURCE_LABEL: Dict[str, str] = {
    "anes": "ANES", "evs": "EVS", "wvs": "WVS",
    "twitter": "Twitter", "eutwitter": "EU Twitter", "weibo": "Weibo",
}
CENTROID_DIR = EMBEDDING_DIR / "topic_centroids"
RANK_TOL = 1e-10


def load_centroids(src: str) -> Tuple[List[str], List[str], np.ndarray]:
    """Return (topic_ids, topic_labels, C) with C's rows unit-normalized."""
    df = pd.read_parquet(CENTROID_DIR / f"{src}_topic_centroids.parquet")
    dims = [c for c in df.columns if c.startswith("dim_")]
    C = df[dims].to_numpy(dtype=np.float64).copy()
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    return df["topic_id"].astype(str).tolist(), df["topic_label"].tolist(), C


def residual(C: np.ndarray, target: int, subset: Tuple[int, ...]) -> float:
    """||c_t - P_S c_t|| for unit c_t, P_S = projection onto span of C[subset]."""
    A = C[list(subset)].T                       # (D, k)
    coef, *_ = np.linalg.lstsq(A, C[target], rcond=None)
    return float(np.linalg.norm(C[target] - A @ coef))


def analyse(src: str) -> Tuple[Dict, List[Dict], List[Dict], Dict]:
    topics, labels, C = load_centroids(src)
    n = len(topics)

    sig = np.linalg.svd(C, compute_uv=False)    # descending
    rank = int((sig > RANK_TOL * sig[0]).sum())
    summary = {
        "source": src, "n_topics": n, "dim": C.shape[1], "rank": rank,
        "sigma_max": sig[0], "sigma_min": sig[-1],
        "sigma_min_over_max": sig[-1] / sig[0], "kappa": sig[0] / sig[-1],
    }
    spectrum = [{"source": src, "i": i + 1, "sigma": s} for i, s in enumerate(sig)]

    sweep: List[Dict] = []
    for t in range(n):
        others = [j for j in range(n) if j != t]
        for k in range(1, n):
            r, best = min((residual(C, t, S), S) for S in combinations(others, k))
            sweep.append({
                "source": src, "target_topic": topics[t], "target_label": labels[t],
                "k": k, "residual": r, "angle_deg": np.degrees(np.arcsin(min(1.0, r))),
                "cos_theta": np.sqrt(max(0.0, 1.0 - r * r)),
                "subset_labels": " + ".join(labels[j] for j in best),
            })

    G = C @ C.T                                 # pairwise cosines, for reporting only
    off = np.triu(np.ones((n, n), dtype=bool), 1)
    i, j = np.unravel_index(np.where(off, G, -np.inf).argmax(), G.shape)
    top_pair = {"source": src, "pair": f"{labels[i]} - {labels[j]}", "cosine": G[i, j],
                "min_offdiag_cosine": G[off].min(), "mean_offdiag_cosine": G[off].mean()}
    return summary, spectrum, sweep, top_pair


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out_dir", default=None,
                    help="table folder (default outputs/reports/<date>_centroid_rank_check)")
    ap.add_argument("--plot", action="store_true",
                    help="also write the SI figure (outputs/figures) and stats (outputs/stats)")
    args = ap.parse_args()

    summ, spec, sweep, pairs = [], [], [], []
    for src in SOURCES:
        s, sp, sw, tp = analyse(src)
        summ.append(s); spec.extend(sp); sweep.extend(sw); pairs.append(tp)
    summ_df, sweep_df, pairs_df = pd.DataFrame(summ), pd.DataFrame(sweep), pd.DataFrame(pairs)
    spec_df = pd.DataFrame(spec).pivot(index="i", columns="source", values="sigma")[SOURCES]

    best = (sweep_df.sort_values("residual").groupby("source", as_index=False).first()
            .set_index("source").loc[SOURCES].reset_index())
    full_span = sweep_df[sweep_df["k"] == 8]          # best residual per target = k = 8

    out = Path(args.out_dir) if args.out_dir else report_dir("centroid_rank_check")
    out.mkdir(parents=True, exist_ok=True)
    summ_df.to_csv(out / "rank_conditioning.csv", index=False)
    spec_df.to_csv(out / "singular_values.csv")
    sweep_df.to_csv(out / "projection_residual_sweep.csv", index=False)
    best.to_csv(out / "most_expressible_topic.csv", index=False)
    pairs_df.to_csv(out / "strongest_pair.csv", index=False)

    lines = [f"Input: {display(CENTROID_DIR)}/<src>_topic_centroids.parquet (unit-normalized)",
             "", "Rank and conditioning"]
    lines += [f"  {SOURCE_LABEL[r.source]:<11} rank={r.rank}/{r.n_topics}  smax={r.sigma_max:.3f}  "
              f"smin={r.sigma_min:.3f}  smin/smax={r.sigma_min_over_max:.3f}  kappa={r.kappa:.2f}"
              for r in summ_df.itertuples()]
    lines += ["", "Singular values", spec_df.rename(columns=SOURCE_LABEL).to_string(
        float_format=lambda x: f"{x:.4f}"), "", "Most-expressible topic"]
    lines += [f"  {SOURCE_LABEL[r.source]:<11} {r.target_label:<12} k={r.k}  residual={r.residual:.3f}  "
              f"angle={r.angle_deg:.1f} deg  cos={r.cos_theta:.3f}" for r in best.itertuples()]
    lines += ["", "Strongest topic pair (single cosine)"]
    lines += [f"  {SOURCE_LABEL[r.source]:<11} {r.pair:<30} cos={r.cosine:.3f}  "
              f"(off-diag min={r.min_offdiag_cosine:.3f}, mean={r.mean_offdiag_cosine:.3f})"
              for r in pairs_df.itertuples()]
    med = full_span["residual"].median()
    lines += ["", f"Across {len(full_span)} source x topic cases (k = 8):",
              f"  min residual = {full_span['residual'].min():.3f}",
              f"  median residual = {med:.3f} ({np.degrees(np.arcsin(med)):.1f} deg)",
              f"  max kappa = {summ_df['kappa'].max():.2f}",
              "  sources whose most-expressible topic is > 36 deg from the span: "
              + str(int((best['angle_deg'] > 36).sum())) + " of 6"]
    report = "\n".join(lines)
    (out / "report.txt").write_text(report + "\n")
    print(report)

    if args.plot:
        name = f"{date_prefix()}_supp_centroid_rank_results"
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        STATS_DIR.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURE_DIR / f"{name}_fig.pdf"
        plot(spec_df, sweep_df, fig_path)
        (STATS_DIR / f"{name}_stats.txt").write_text(report + "\n")
        print(f"wrote {fig_path} and {STATS_DIR / (name + '_stats.txt')}")


SOURCE_COLOR: Dict[str, str] = {                 # region colors from plotting.figures
    "anes": "#207de6", "twitter": "#207de6",
    "evs": "#54457F", "eutwitter": "#54457F",
    "wvs": "#ff7333", "weibo": "#ff7333",
}
SURVEYS = {"anes", "evs", "wvs"}
MEDIA_DASH = (0, (4, 2))                          # social media = dashed, as in plotting.figures


def _style(src: str) -> Dict:
    return {"color": SOURCE_COLOR[src],
            "linestyle": "-" if src in SURVEYS else MEDIA_DASH, "linewidth": 2}


def plot(spec: pd.DataFrame, sweep: pd.DataFrame, pdf: Path) -> None:
    """Three-panel SI figure: spectra, best angle vs k, per-topic angle at k = 8."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams["font.family"] = "Times New Roman"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.35]})

    ax = axes[0]
    for src in SOURCES:
        ax.plot(spec.index, spec[src], marker="o", markersize=5, **_style(src))
    ax.set_xlabel("Index $i$")
    ax.set_ylabel(r"Singular value $\sigma_i$")
    ax.set_xticks(spec.index)
    ax.set_ylim(0, None)
    ax.set_title("Singular-value spectra", fontsize=12)

    ax = axes[1]
    curve = sweep.groupby(["source", "k"])["angle_deg"].min().unstack("source")
    for src in SOURCES:
        ax.plot(curve.index, curve[src], marker="o", markersize=5, **_style(src))
    ax.set_xlabel("Number of other topics in the span, $k$")
    ax.set_ylabel(r"Smallest principal angle $\theta$ (deg)")
    ax.set_xticks(curve.index)
    ax.set_ylim(0, 90)
    ax.set_title("Best reconstruction from $k$ topics", fontsize=12)

    ax = axes[2]
    full = sweep[sweep["k"] == 8]
    rng = np.random.default_rng(0)
    for x, src in enumerate(SOURCES):
        sub = full[full["source"] == src]
        jitter = rng.uniform(-0.15, 0.15, len(sub))
        filled = SOURCE_COLOR[src] if src in SURVEYS else "white"
        ax.scatter(x + jitter, sub["angle_deg"], s=40, facecolor=filled,
                   edgecolor=SOURCE_COLOR[src], linewidth=1.5, zorder=3)
        i_lo = int(np.argmin(sub["angle_deg"].to_numpy()))
        lo = sub.iloc[i_lo]
        ax.annotate(lo["target_label"], (x + jitter[i_lo], lo["angle_deg"]), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=9, color="#333333")
    ax.set_xticks(range(len(SOURCES)))
    ax.set_xticklabels([SOURCE_LABEL[s] for s in SOURCES])
    ax.set_ylabel(r"Principal angle $\theta$ to other eight (deg)")
    ax.set_ylim(0, 90)
    ax.set_title("Per-topic gap from the other eight", fontsize=12)

    for ax, letter in zip(axes, "abc"):
        ax.text(-0.14, 1.06, letter.upper(), transform=ax.transAxes,
                fontsize=14, fontweight="bold", fontstyle="italic", va="bottom", ha="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
        ax.set_axisbelow(True)

    handles = [Line2D([], [], marker="o", markersize=5, label=SOURCE_LABEL[s], **_style(s))
               for s in SOURCES]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.04), fontsize=10, handlelength=3)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(pdf.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
