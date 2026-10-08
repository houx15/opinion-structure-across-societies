"""make_graphical_abstract — the 9 data-driven panels for the paper's graphical abstract.

Each panel is exported as its own tight, transparent-background PDF so the author
can assemble them (with titles, country maps, logos, arrows, labels) in PowerPoint.

Data loading and the country palette are reused from ``plotting.figures`` so these
panels stay numerically consistent with the paper's Fig 1 / Fig 2 / Fig 3.

Conventions that differ from plotting.figures (per the GA spec):
    - Sans-serif typography (Arial/Helvetica), not Times New Roman.
    - No in-figure titles, legends, captions, or panel labels.
    - Transparent background, tight margins.
    - US blue / CN orange / EU purple, solid (offline) / dashed (online).

Row 2 (2A/2B/2C) is a *schematic* methodology illustration built on a small
synthetic user x topic matrix — it is pedagogical, not a finding, so it carries
no real-data dependency.

Usage:
    python -m plotting.graphical_abstract            # all panels -> outputs/figures/<date>_graphical_abstract_panels/
    python -m plotting.graphical_abstract --only 1A,3C
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from common.paths import FIGURE_DIR, ROOT as _REPO_ROOT, date_prefix

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from plotting import figures as P


# -- Style -------------------------------------------------------------------

OUTPUT_DIR = FIGURE_DIR / f"{date_prefix()}_graphical_abstract_panels"

# Country palette — reused verbatim from plotting.figures so the GA matches the
# paper's figures (US blue, CN orange, EU purple).
COLOR = {
    "US": P._TWITTER_COLOR,    # blue   #207de6
    "CN": P._WEIBO_COLOR,      # orange #ff7333
    "EU": P._EUTWITTER_COLOR,  # purple #54457F
}

# Offline (survey) / online (social) source per country for the main analysis.
REGION_SOURCES: Dict[str, Tuple[str, str]] = {
    "US": ("anes", "twitter"),
    "CN": ("wvs", "weibo"),
    "EU": ("evs_resample2", "eutwitter"),
}

_DASH = P.Plotter._MEDIA_DASH_PATTERN  # (0, (4, 2)) — the online dash pattern

# Typography. Panels are small in the final layout, so axis labels ~9pt / ticks ~8pt.
AXIS_LABEL_SIZE = 9
TICK_LABEL_SIZE = 8
ANNOT_SIZE = 8

# Vector export format for every panel (e.g. "svg" or "pdf").
EXPORT_FORMAT = "svg"


def _apply_rc() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.linewidth": 0.8,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "pdf.fonttype": 42,    # embed TrueType so text stays editable in PDF
        # Render SVG text as vector paths so glyph geometry is fixed: this
        # matches the tight bbox crop exactly in any viewer, avoiding the text
        # overflow/clipping that "none" causes when a viewer substitutes fonts.
        "svg.fonttype": "path",
    })


def _arrow_spines(ax, *, x_arrow: bool = False, y_arrow: bool = True) -> None:
    """L-shaped axes (top/right hidden) with arrow tips at the requested ends."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_AXIS_GRAY)
        ax.spines[s].set_linewidth(1.0)
    if x_arrow:
        ax.plot(1, 0, marker=">", ms=6, color=_AXIS_GRAY,
                transform=ax.transAxes, clip_on=False)
    if y_arrow:
        ax.plot(0, 1, marker="^", ms=6, color=_AXIS_GRAY,
                transform=ax.transAxes, clip_on=False)


def _save(fig, filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Honor EXPORT_FORMAT regardless of the extension the caller passed.
    path = OUTPUT_DIR / f"{Path(filename).stem}.{EXPORT_FORMAT}"
    fig.savefig(
        path, format=EXPORT_FORMAT, transparent=True,
        bbox_inches="tight", pad_inches=0.02,
    )
    plt.close(fig)
    print(f"  wrote {path.relative_to(_REPO_ROOT)}")
    return path


# -- Row 1: concept scatters -------------------------------------------------


_AXIS_GRAY = "#333333"


def _scatter_cloud(ax, x, y, color, *, x_label, y_label, alpha,
                   jitter: bool = True, point_size: float = 6,
                   fit_line: bool = False, annotate_r: bool = False) -> None:
    """Panel-A-style scatter cloud on directional opinion axes.

    With ``jitter`` (real Likert data) integers 1..5 get a small Gaussian
    jitter (sigma 0.08, the plotting.figures _scatter_with_fit treatment) at very
    low alpha, so points don't stack on the integer grid. With ``jitter=False``
    (continuous toy data) points are drawn as-is. Optional ``fit_line`` adds an
    OLS line and ``annotate_r`` an |r| / N label (both computed on the raw,
    un-jittered values). The left/bottom axes carry arrow tips and -/+ markers.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x_raw, y_raw = x[keep], y[keep]

    if jitter:
        rng = np.random.default_rng(0)
        x = x_raw + rng.normal(scale=0.08, size=x_raw.size)
        y = y_raw + rng.normal(scale=0.08, size=y_raw.size)
    else:
        x, y = x_raw, y_raw
    ax.scatter(x, y, s=point_size, alpha=alpha, color=color, edgecolors="none")

    if fit_line:
        slope, intercept = np.polyfit(x_raw, y_raw, 1)
        xs = np.linspace(x_raw.min(), x_raw.max(), 100)
        ax.plot(xs, slope * xs + intercept, color=color, linewidth=2.4, zorder=3)
    if annotate_r:
        r = abs(float(np.corrcoef(x_raw, y_raw)[0, 1]))
        ax.text(0.05, 0.95, f"|r| = {r:.3f}\nN = {x_raw.size:,}",
                transform=ax.transAxes, ha="left", va="top", fontsize=ANNOT_SIZE,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.85),
                zorder=5)

    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_xticks([]); ax.set_yticks([])  # no tick marks / numeric labels
    ax.set_aspect("equal")

    # L-shaped axes with arrow tips and -/+ markers at the ends.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_AXIS_GRAY)
        ax.spines[s].set_linewidth(1.0)
    ax.plot(1, 0, marker=">", ms=6, color=_AXIS_GRAY,
            transform=ax.transAxes, clip_on=False)
    ax.plot(0, 1, marker="^", ms=6, color=_AXIS_GRAY,
            transform=ax.transAxes, clip_on=False)
    end_kw = dict(transform=ax.transAxes, color=_AXIS_GRAY,
                  fontsize=11, clip_on=False)
    ax.text(0.02, -0.04, "−", ha="left", va="top", **end_kw)
    ax.text(0.95, -0.04, "+", ha="left", va="top", **end_kw)
    ax.text(-0.05, 0.02, "−", ha="right", va="bottom", **end_kw)
    ax.text(-0.05, 0.96, "+", ha="right", va="top", **end_kw)

    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE)


def _toy_cloud(corr_strength: float, seed: int, n: int = 400):
    """A 2D toy opinion cloud centered in the panel, with the given correlation."""
    rng = np.random.default_rng(seed)
    var = 0.62
    cov = [[var, corr_strength * var], [corr_strength * var, var]]
    pts = rng.multivariate_normal([3.0, 3.0], cov, size=n)
    return pts[:, 0], pts[:, 1]


def panel_1A() -> Path:
    """Concept scatter — strongly aligned toy cloud (brown, matches 2C low-dim)."""
    x, y = _toy_cloud(corr_strength=0.9, seed=1)
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    _scatter_cloud(
        ax, x, y, _C2_LOWDIM,
        x_label="Opinion 1", y_label="Opinion 2",
        alpha=0.5, jitter=False, point_size=9,
    )
    return _save(fig, "ga_1A_aligned_scatter.svg")


def panel_1C() -> Path:
    """Concept scatter — weakly aligned toy cloud (green, matches 2C high-dim)."""
    x, y = _toy_cloud(corr_strength=0.0, seed=2)
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    _scatter_cloud(
        ax, x, y, _C2_HIGHDIM,
        x_label="Opinion 1", y_label="Opinion 2",
        alpha=0.5, jitter=False, point_size=9,
    )
    return _save(fig, "ga_1C_unaligned_scatter.svg")


def panel_F1_real() -> Path:
    """Real US example scatter — LGBT x Climate (ANES), the former 1A (US blue)."""
    df = P.get_case_study_pair("anes")
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    _scatter_cloud(
        ax, df["x"], df["y"], COLOR["US"],
        x_label="LGBT", y_label="Climate", alpha=0.03,
        fit_line=True, annotate_r=True,
    )
    return _save(fig, "ga_F1_us_real_scatter.svg")


# -- Row 2: schematic measurement pipeline (mirrors fake-overview panel B) ----

_N_TOPICS = 6
_TOPIC_LETTERS = list("ABCDEF")

# Shared Row-2 regime colors: brown = highly aligned / highly correlated /
# collapses to low dimension; green = low-aligned / weakly correlated / stays
# high-dimensional. Used consistently across 1A/1C, 2B (split), and 2C.
_C2_LOWDIM = "#a23b1e"   # brown
_C2_HIGHDIM = "#4f7a3f"  # green
_CMAP_BROWN = LinearSegmentedColormap.from_list("ga_brown", ["white", _C2_LOWDIM])
_CMAP_GREEN = LinearSegmentedColormap.from_list("ga_green", ["white", _C2_HIGHDIM])


def _factor_matrix(n_users: int, factor_strength: float, seed: int,
                   noise: float = 0.6) -> np.ndarray:
    """Standardized user x topic matrix with a tunable dominant latent factor.

    A large ``factor_strength`` concentrates variance on one axis (the cloud
    collapses to low dimension); a small one leaves topics near-independent
    (high dimensional). Purely illustrative — Row 2 is schematic.
    """
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(_N_TOPICS, 3))
    loadings[:, 0] *= factor_strength
    scores = rng.normal(size=(n_users, 3))
    matrix = scores @ loadings.T + rng.normal(scale=noise, size=(n_users, _N_TOPICS))
    return (matrix - matrix.mean(0)) / matrix.std(0)


def _synthetic_user_topic(seed: int = 42) -> np.ndarray:
    """Large user x topic sample for 2A/2B (2A shows only its first few rows)."""
    return _factor_matrix(200, factor_strength=1.6, seed=seed)


def _bottom_colorbar(fig, ax, mesh, *, label: str, ticks) -> None:
    """Horizontal colorbar under a heatmap, styled like fake-overview panel B."""
    cbar = fig.colorbar(
        mesh, ax=ax, orientation="horizontal",
        fraction=0.055, pad=0.06, aspect=26,
    )
    cbar.set_ticks(ticks)
    cbar.set_label(label, fontsize=AXIS_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE, length=2)
    cbar.outline.set_linewidth(0.5)
    cbar.solids.set_rasterized(False)  # keep the gradient vector, not a raster strip


def _hide_spines(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_2A() -> Path:
    """User x topic opinion matrix — Topic A..F headers, User 1..N rows, colorbar."""
    matrix = _synthetic_user_topic()
    # Show User 1/2/3, a blank '...' row, then 'User N' — the panel-B layout.
    disp = np.vstack([
        matrix[0], matrix[1], matrix[2],
        np.full(_N_TOPICS, np.nan),   # gap rendered transparent -> the ellipsis
        matrix[3],
    ])
    fig, ax = plt.subplots(figsize=(3.8, 3.5))
    mesh = ax.pcolormesh(
        np.ma.masked_invalid(disp), cmap="RdBu", vmin=-2, vmax=2,
        edgecolors="white", linewidth=1.2,
    )
    ax.invert_yaxis()
    ax.set_aspect("auto")
    # Topic headers across the top.
    ax.xaxis.set_ticks_position("top")
    ax.set_xticks(np.arange(_N_TOPICS) + 0.5)
    ax.set_xticklabels([f"Topic {c}" for c in _TOPIC_LETTERS], fontsize=TICK_LABEL_SIZE)
    # User labels down the left, with an ellipsis for the omitted middle.
    ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])
    ax.set_yticklabels(["User 1", "User 2", "User 3", "⋮", "User N"],
                       fontsize=TICK_LABEL_SIZE)
    ax.tick_params(length=0)
    _hide_spines(ax)
    _bottom_colorbar(fig, ax, mesh, label="Opinion score", ticks=[-2, 0, 2])
    return _save(fig, "ga_2A_user_topic_matrix.pdf")


_C2_N_COMPONENTS = 9

# The two shared Row-2 regimes (same data drives 2B-split and 2C).
_REGIME_LOWDIM = dict(strengths=[2.2, 1.0, 0.45], noise=0.35, seed=11)
_REGIME_HIGHDIM = dict(
    strengths=[0.45, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30, 0.28], noise=1.2, seed=22,
)


def _regime_matrix(strengths: Sequence[float], *, noise: float, seed: int,
                   n_samp: int = 600) -> np.ndarray:
    """Synthetic (n_samp x 9) data matrix for one Row-2 regime.

    A few strong factors concentrate variance (highly correlated topics, low
    dimensional); many comparable weak ones leave topics near-independent
    (weakly correlated, high dimensional).
    """
    rng = np.random.default_rng(seed)
    k = len(strengths)
    loadings = rng.normal(size=(_C2_N_COMPONENTS, k)) * np.asarray(strengths)
    scores = rng.normal(size=(n_samp, k))
    return scores @ loadings.T + rng.normal(scale=noise, size=(n_samp, _C2_N_COMPONENTS))


def _regime_spectrum(**regime) -> np.ndarray:
    """Normalized (lambda_1 = 1) eigenvalue spectrum of a regime's correlation matrix."""
    ev = np.linalg.eigvalsh(np.corrcoef(_regime_matrix(**regime), rowvar=False))[::-1]
    return ev / ev[0]


def _correlation_matrix_panel(regime: Dict, cmap, filename: str) -> Path:
    """Heatmap of |correlation| for one regime, in its theme color (no labels)."""
    corr = np.abs(np.corrcoef(_regime_matrix(**regime), rowvar=False))
    fig, ax = plt.subplots(figsize=(3.2, 3.4))
    mesh = ax.pcolormesh(corr, cmap=cmap, vmin=0, vmax=1,
                         edgecolors="white", linewidth=1.2)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    _hide_spines(ax)
    _bottom_colorbar(fig, ax, mesh, label="Correlation strength", ticks=[0, 0.5, 1])
    return _save(fig, filename)


def panel_2B_high() -> Path:
    """Highly correlated topic matrix (brown) — the low-dimensional regime."""
    return _correlation_matrix_panel(_REGIME_LOWDIM, _CMAP_BROWN,
                                     "ga_2B_high_correlation.svg")


def panel_2B_low() -> Path:
    """Weakly correlated topic matrix (green) — the high-dimensional regime."""
    return _correlation_matrix_panel(_REGIME_HIGHDIM, _CMAP_GREEN,
                                     "ga_2B_low_correlation.svg")


def panel_2C() -> Path:
    """Two eigenvalue spectra over 9 components (panel B's two regimes).

    The low-dimensional regime collapses toward ~0 by the third component; the
    high-dimensional one stays elevated across all nine. Both normalized so
    lambda_1 = 1 on a linear axis.
    """
    low_dim = _regime_spectrum(**_REGIME_LOWDIM)
    high_dim = _regime_spectrum(**_REGIME_HIGHDIM)
    comp = np.arange(1, _C2_N_COMPONENTS + 1)

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    for data, color in [(low_dim, _C2_LOWDIM), (high_dim, _C2_HIGHDIM)]:
        ax.plot(
            comp, data, color=color, marker="o", markersize=5,
            linewidth=1.8, markeredgecolor="white", markeredgewidth=0.6,
        )
    ax.set_xticks(comp)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("Component", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Normalized eigenvalue", fontsize=AXIS_LABEL_SIZE)
    # Both labels sit in the empty right half, clear of either curve.
    ax.text(0.52, 0.97, "Low-aligned =\nhigh dimensional", transform=ax.transAxes,
            color=_C2_HIGHDIM, fontsize=8, style="italic", va="top", ha="left")
    ax.text(0.52, 0.24, "Highly aligned =\nlow dimensional", transform=ax.transAxes,
            color=_C2_LOWDIM, fontsize=8, style="italic", va="top", ha="left")
    _arrow_spines(ax, x_arrow=True, y_arrow=True)
    return _save(fig, "ga_2C_eigenvalue_spectrum.pdf")


# -- Row 3: results ----------------------------------------------------------


def _compression_chart(
    ax,
    values: Dict[str, Tuple[float, float]],
    *,
    ylim: Optional[Tuple[float, float]] = None,
    invert_y: bool = False,
) -> None:
    """Offline -> online compression as two columns of enlarged dots.

    ``values`` maps country code -> (offline, online). Each country contributes
    one large dot at the offline tick and one at the online tick, in its color.
    No connecting line, no numeric labels, and a bare y-axis (arrow only) — the
    convergence of the online dots is the visible 'compression'.

    ``invert_y`` flips the axis for dimensionality metrics (PR / eRank), where
    higher = higher dimension = *lower* alignment; inverting keeps "up = more
    aligned" consistent with the |r| panel.
    """
    tick_off, tick_on = 0.0, 1.0

    all_vals = [v for pair in values.values() for v in pair]
    v_min, v_max = min(all_vals), max(all_vals)
    span = max(v_max - v_min, 1e-6)
    lo, hi = (ylim if ylim is not None
              else (v_min - 0.14 * span, v_max + 0.14 * span))

    for code, (off, on) in values.items():
        # Thin connector between a country's offline and online dot.
        ax.plot([tick_off, tick_on], [off, on],
                color=COLOR[code], linewidth=1.0, zorder=2)
        ax.plot(
            [tick_off, tick_on], [off, on],
            color=COLOR[code], linestyle="None", marker="o", markersize=13,
            markeredgecolor="white", markeredgewidth=1.0, zorder=3,
        )

    ax.set_xticks([tick_off, tick_on])
    ax.set_xticklabels(["Offline", "Online"], fontsize=AXIS_LABEL_SIZE)
    ax.set_yticks([])  # no y ticks / numeric labels / label
    ax.set_xlim(-0.55, 1.55)
    # For PR/eRank, put low dimension (high alignment) at the top.
    ax.set_ylim((hi, lo) if invert_y else (lo, hi))
    _arrow_spines(ax, y_arrow=True)


def panel_3A() -> Path:
    """Raw correlation compression — mean pairwise |r| by country x setting."""
    values = {
        code: (
            float(P.load_pairwise_correlation(off)["correlation"].mean()),
            float(P.load_pairwise_correlation(on)["correlation"].mean()),
        )
        for code, (off, on) in REGION_SOURCES.items()
    }
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    _compression_chart(ax, values)  # auto-fit y
    return _save(fig, "ga_3A_raw_correlation.pdf")


def _semantic_curves(ax, sources: Sequence[Tuple[str, str]], *, line_style,
                     x0: float, x1: float) -> None:
    """Plot per-country OLS |r|-vs-similarity lines filling the half [x0, x1].

    Linear fits match the paper's Fig 2 (plotting.figures ``trend="linear"``).

    Each group (offline / online) is rescaled to its *own* similarity range so
    it spans the full sub-range — online no longer crowds into a narrow slice
    of its half, which also flattens the apparent fluctuation.
    """
    curves = []
    for code, src in sources:
        corr = P.load_pairwise_correlation(src)
        sem = P.load_semantic_similarity(src, embedding_type="gpt")
        merged = (
            corr.merge(sem[["topic_combination", "similarity"]],
                       on="topic_combination", how="left")
            .dropna(subset=["similarity"])
            .drop_duplicates(subset=["topic_combination"])
        )
        x = merged["similarity"].to_numpy()
        y = merged["correlation"].to_numpy()
        if len(merged) < 4 or np.ptp(x) == 0:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        x_grid = np.linspace(x.min(), x.max(), 100)
        curves.append((code, np.column_stack([x_grid, intercept + slope * x_grid])))
    if not curves:
        return

    lo = min(c[:, 0].min() for _, c in curves)
    hi = max(c[:, 0].max() for _, c in curves)
    span = hi - lo if hi > lo else 1.0
    for code, curve in curves:
        cx = x0 + (x1 - x0) * (curve[:, 0] - lo) / span
        ax.plot(cx, curve[:, 1], color=COLOR[code], linewidth=2.6,
                linestyle=line_style)


def panel_3B() -> Path:
    """|r| vs semantic similarity in one frame — offline (left) / online (right).

    A single x-axis (Offline | Online, like 3A/3C) and a single bare y-axis
    (no ticks/labels, arrow only). Offline lines are solid in the left half,
    online lines dashed in the right half; each group fills its own half.
    """
    offline = [(code, off) for code, (off, _) in REGION_SOURCES.items()]
    online = [(code, on) for code, (_, on) in REGION_SOURCES.items()]

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    _semantic_curves(ax, offline, line_style="solid", x0=0.05, x1=0.45)
    _semantic_curves(ax, online, line_style=_DASH, x0=0.55, x1=0.95)
    ax.axvline(0.5, color="0.85", linewidth=0.8, zorder=0)  # offline | online divider

    ax.set_xlim(0, 1)
    # A linear fit can dip below 0 at the low-similarity end; keep it in frame.
    ax.set_ylim(min(0.0, ax.dataLim.y0 - 0.02), 0.6)
    ax.set_xticks([0.25, 0.75])
    ax.set_xticklabels(["Offline", "Online"], fontsize=AXIS_LABEL_SIZE)
    ax.set_yticks([])  # no y ticks / numeric labels / label
    _arrow_spines(ax, y_arrow=True)
    return _save(fig, "ga_3B_semantic.pdf")


def panel_3C() -> Path:
    """Effective dimensionality (PR) compression — mean PR by country x setting."""
    values = {
        code: (
            float(np.mean(P.load_spectral_summary(off)["PR"])),
            float(np.mean(P.load_spectral_summary(on)["PR"])),
        )
        for code, (off, on) in REGION_SOURCES.items()
    }
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    # Invert: lower PR = lower dimension = higher alignment, drawn at the top.
    _compression_chart(ax, values, invert_y=True)  # auto-fit y
    return _save(fig, "ga_3C_pr.pdf")


# -- Extra explanatory panels ------------------------------------------------


def panel_fig2a_us() -> Path:
    """US offline pairwise-|r| distribution as a KDE density curve (Fig 2a, US only)."""
    from scipy.stats import gaussian_kde

    vals = P.load_pairwise_correlation("anes")["correlation"].dropna().to_numpy()
    kde = gaussian_kde(vals)
    xs = np.linspace(0, vals.max() * 1.05, 240)
    dens = kde(xs)
    mean = float(vals.mean())
    dens_mean = float(kde(mean)[0])

    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    ax.fill_between(xs, 0, dens, color=COLOR["US"], alpha=0.15, linewidth=0)
    ax.plot(xs, dens, color=COLOR["US"], linewidth=2.0)
    # Mean marker.
    ax.plot([mean, mean], [0, dens_mean], color=COLOR["US"], linewidth=1.4,
            linestyle=(0, (3, 2)))
    ax.text(mean, dens_mean * 1.03, f"mean = {mean:.2f}", ha="center", va="bottom",
            fontsize=ANNOT_SIZE, color=COLOR["US"])

    ax.set_yticks([])
    ax.set_ylim(0, dens.max() * 1.18)
    ax.set_xlim(0, vals.max() * 1.05)
    ax.set_xlabel("|r|", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    return _save(fig, "ga_F2a_us_r_distribution.svg")


def panel_venn() -> Path:
    """Word-overlap Venns (low- vs high-similarity US-Twitter topic pairs)."""
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    P.draw_word_venns(ax)
    return _save(fig, "ga_venn_word_overlap.svg")


def _classical_mds(dist: np.ndarray, dim: int = 2) -> np.ndarray:
    """Classical (Torgerson) MDS — embed a distance matrix into ``dim`` axes."""
    n = dist.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (dist ** 2) @ j
    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(vals)[::-1][:dim]
    return vecs[:, order] * np.sqrt(np.clip(vals[order], 0, None))


def panel_coordinator() -> Path:
    """Semantic similarity as vectors — 4 US-Twitter topics drawn from the origin.

    Similarity = cosine = the angle between vectors. UBI and Min Wage (high
    similarity) point in nearly the same direction; Vaccine and Social Media
    (low similarity) point far apart. Directions come from MDS on the chordal
    distance sqrt(2 - 2·cos) of the GPT topic similarities.
    """
    sem = P.load_semantic_similarity("twitter")
    topics = ["ubi", "minwage", "vac", "soc"]
    labels = {"ubi": "UBI", "minwage": "Min Wage", "vac": "Vaccine", "soc": "Social Media"}
    color = {"ubi": _C2_HIGHDIM, "minwage": _C2_HIGHDIM, "vac": _C2_LOWDIM, "soc": _C2_LOWDIM}
    idx = {t: i for i, t in enumerate(topics)}
    n = len(topics)

    sim = np.eye(n)
    for _, row in sem.iterrows():
        a, b = row["tid1"], row["tid2"]
        if a in idx and b in idx:
            sim[idx[a], idx[b]] = sim[idx[b], idx[a]] = float(row["similarity"])
    coords = _classical_mds(np.sqrt(np.clip(2.0 - 2.0 * sim, 0, None)), dim=2)

    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    lim = float(np.abs(coords).max()) * 1.45
    ax.axhline(0, color="#dddddd", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#dddddd", linewidth=0.8, zorder=0)
    for t in topics:
        x, y = coords[idx[t]]
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color[t], linewidth=2.2))
        ax.annotate(labels[t], (x, y), textcoords="offset points",
                    xytext=(7 * np.sign(x or 1), 7 * np.sign(y or 1)),
                    fontsize=9, fontweight="bold", color=color[t],
                    ha="center", va="center")

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return _save(fig, "ga_coordinator_semantic_map.svg")


# -- Fig 3 toy-dimensionality panels (reused from plotting.figures) ---------------

_TOY_PLOTTER = None


def _toy_plotter():
    """A bare Plotter just for its toy-cloud / scree drawing methods (no data I/O)."""
    global _TOY_PLOTTER
    if _TOY_PLOTTER is None:
        _TOY_PLOTTER = P.Plotter(figure_folder=OUTPUT_DIR)
    return _TOY_PLOTTER


def panel_fig3b_cloud() -> Path:
    """Fig 3 (b): the '2D (plate)' toy opinion cloud — no title."""
    pl = _toy_plotter()
    _, eigs, rendering = pl._TOY_CLOUDS[1]   # index 1 = "2D (plate)"
    fig = plt.figure(figsize=(3.0, 3.0))
    ax = fig.add_subplot(111, projection="3d")
    pl._toy_cloud_panel(ax, eigs, title="", seed=102, rendering=rendering)
    return _save(fig, "ga_fig3b_cloud.svg")


def panel_fig3e_scree() -> Path:
    """Fig 3 (e): the scree / variance spectrum of the '2D (plate)' cloud."""
    pl = _toy_plotter()
    _, eigs, _ = pl._TOY_CLOUDS[1]
    fig = plt.figure(figsize=(3.2, 3.0))
    ax = fig.add_subplot(111)
    pl._toy_scree_panel(ax, eigs, seed=102)
    return _save(fig, "ga_fig3e_scree.svg")


# -- Orchestration -----------------------------------------------------------

PANELS = {
    "1A": panel_1A,
    "1C": panel_1C,
    "F1_real": panel_F1_real,
    "2A": panel_2A,
    "2B_high": panel_2B_high,
    "2B_low": panel_2B_low,
    "2C": panel_2C,
    "venn": panel_venn,
    "coordinator": panel_coordinator,
    "fig3b_cloud": panel_fig3b_cloud,
    "fig3e_scree": panel_fig3e_scree,
    "3A": panel_3A,
    "3B": panel_3B,
    "3C": panel_3C,
    "F2a_us": panel_fig2a_us,
}


def main(only: Optional[str] = None) -> List[Path]:
    """Build every panel (or a comma-separated subset via ``--only 1A,3C``)."""
    _apply_rc()
    selected = list(PANELS) if only is None else [s.strip() for s in str(only).split(",")]
    print(f"[graphical_abstract] building {len(selected)} panel(s) -> {OUTPUT_DIR}")
    paths = []
    for key in selected:
        if key not in PANELS:
            raise KeyError(f"Unknown panel {key!r}. Known: {list(PANELS)}")
        paths.append(PANELS[key]())
    return paths


if __name__ == "__main__":
    import fire

    fire.Fire(main)
