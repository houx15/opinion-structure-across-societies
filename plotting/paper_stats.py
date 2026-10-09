"""Plain-text statistics report for the paper, regenerated with every figure run.

Every number comes from the same loaders as ``plotting/figures.py`` (pooled
"average" year, nine topics per source -> 36 topic pairs), so the text and the
figures cannot drift apart. ``figures.run_task`` calls ``write_report``
after drawing, producing ``<prefix>_<task>_results_stats.txt`` next to the PDFs.

The report is organised part by part; each part states its data source, the
calculation, and the result, followed by an appendix listing every pair's |r|
so any number can be re-derived by hand.

Usage:
    python -m plotting.paper_stats                # main task -> outputs/stats/
    uv run python paper_stats.py --task robust_1_media
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import skew

from common.paths import display
from plotting import figures as v4

N_TOPICS = 9
DEFAULT_N_BOOT = 2000
SEED = 0

# Sample sizes reported in the methods: label -> source whose prepared data
# defines N.
SAMPLE_SIZE_SOURCES: Dict[str, str] = {
    "ANES (US survey)": "anes",
    "WVS (CN survey)": "wvs",
    "EVS (EU survey, resampled)": "evs_resample2",
    "EVS (EU survey, no replacement)": "evs",
    "EVS (UK / England survey)": "en_evs",
    "Twitter (US)": "twitter",
    "Twitter (EU)": "eutwitter",
    "Weibo (CN)": "weibo",
}

MEDIA = (("Survey", "survey"), ("Social media", "social"))
RULE = "=" * 78


# -- data -----------------------------------------------------------------------


def _pair_table(src: str, year: str = "average"):
    """One row per topic pair: topic_combination, pearson, |r|, intersection."""
    return v4.load_pairwise_correlation(src, year=year).drop_duplicates(subset=["topic_combination"])


def _fisher(r: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(r, 0.0, 1 - 1e-12))


def _eta_sq(groups: Sequence[np.ndarray]) -> Tuple[float, float, float]:
    allv = np.concatenate(groups)
    grand = allv.mean()
    ss_total = float(((allv - grand) ** 2).sum())
    ss_between = float(sum(len(g) * (g.mean() - grand) ** 2 for g in groups))
    return ss_between, ss_total, ss_between / ss_total if ss_total > 0 else float("nan")


def _similarity_pairs(src: str, embedding_type: str, year: str = "average") -> Tuple[np.ndarray, np.ndarray]:
    """(semantic similarity, |r|) per topic pair, merged exactly as Fig 2g, h."""
    corr = v4.load_pairwise_correlation(src, year=year)
    sem = v4.load_semantic_similarity(src, embedding_type=embedding_type)
    merged = (
        corr.merge(sem[["topic_combination", "similarity"]], on="topic_combination", how="left")
        .dropna(subset=["similarity"])
        .drop_duplicates(subset=["topic_combination"])
    )
    return merged["similarity"].to_numpy(float), merged["correlation"].to_numpy(float)


def _gap_closed(xu, yu, xo, yo, ref) -> Tuple[float, float]:
    """(gap remaining at equal similarity, share of raw gap eliminated).

    Both societies' OLS lines are evaluated at the reference similarities;
    the remaining gap is the mean difference of the two lines there.
    """
    bu, au = np.polyfit(xu, yu, 1)
    bo, ao = np.polyfit(xo, yo, 1)
    raw = yu.mean() - yo.mean()
    remaining = float(((au + bu * ref) - (ao + bo * ref)).mean())
    return remaining, 1.0 - remaining / raw


def _spectral_path(src: str, matrix: str = "cov") -> Path:
    stem = v4._SPECTRAL_STEM_OVERRIDE.get(src, src)
    return v4.dimension_results_dir(matrix) / f"{stem}-none-summary.json"


def _sample_size(src: str) -> Optional[Tuple[int, str]]:
    if src not in v4.SOCIAL_SOURCES:
        path = v4.OPINION_DIR / f"individual_opinion_{src}.parquet"
        if not path.exists():
            return None
        return len(v4.load_individual_opinion(src)), f"rows of {display(path)}"
    stem = v4._SPECTRAL_STEM_OVERRIDE.get(src, src)
    files = sorted(v4.dimension_results_dir("cov").glob(f"{stem}-*.csr-loadings.json"))
    if not files:
        return None
    ns = [json.load(open(f))["N"] for f in files]
    if len(ns) == 1:
        return ns[0], f'"N" in {display(files[0])}'
    return sum(ns), (
        f'sum of "N" over {len(ns)} yearly files {display(v4.dimension_results_dir("cov"))}/{stem}-<year>.csr-loadings.json '
        f"({', '.join(f'{n:,}' for n in ns)}); NOT unique users - "
        "users active in several years are counted more than once"
    )


# -- formatting -----------------------------------------------------------------


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    cols = [header, *rows]
    widths = [max(len(str(r[i])) for r in cols) for i in range(len(header))]
    fmt = lambda r: "  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip()
    return [fmt(header), fmt(["-" * w for w in widths]), *[fmt(r) for r in rows]]


def _part(title: str, source: Sequence[str], calc: Sequence[str], result: Sequence[str]) -> List[str]:
    out = ["", RULE, title, RULE, "Source:"]
    out += [f"  {s}" for s in source]
    out += ["Calculation:"] + [f"  {c}" for c in calc]
    out += ["Result:"] + [f"  {r}" for r in result]
    return out


def _ci(lo: float, hi: float, d: int = 3) -> str:
    return f"[{lo:.{d}f}, {hi:.{d}f}]"


# -- report ---------------------------------------------------------------------


def build_report(plotter: "v4.Plotter", n_boot: int = DEFAULT_N_BOOT, header: str = "") -> str:
    regions = plotter.regions
    rng = np.random.default_rng(SEED)
    csv = lambda src: display(v4.CORRELATION_DIR / f"network_analysis_{src}.csv")
    pair_source = [
        f"{csv(getattr(r, attr))}  ({r.code} {medium.lower()})"
        for medium, attr in MEDIA for r in regions
    ] + [
        ('Rows with year == "average" (each user pooled over all years)' if plotter.social_year == "average"
         else f'Social media: rows with year == "{plotter.social_year}" (opinions of that year only); surveys: "average"')
        + "; only the 9 topics of each source",
        "(figures.RESTRICTED_TOPICS) -> 36 topic pairs. |r| = abs(\"Correlation (Pearson)\").",
        "Loaded with figures.load_pairwise_correlation (same as Fig 2).",
    ]
    boot_note = (
        f"95% CI: percentile bootstrap, {n_boot:,} resamples of the 36 topic pairs "
        f"(with replacement), each source resampled independently; numpy seed {SEED}."
    )

    tables: Dict[Tuple[str, str], object] = {}
    absr: Dict[Tuple[str, str], np.ndarray] = {}
    boot_mean: Dict[Tuple[str, str], np.ndarray] = {}
    boot_zmean: Dict[Tuple[str, str], np.ndarray] = {}
    for r in regions:
        for medium, attr in MEDIA:
            key = (r.code, medium)
            df = _pair_table(getattr(r, attr), plotter.correlation_year(getattr(r, attr)))
            tables[key] = df
            a = df["correlation"].to_numpy(float)
            absr[key] = a
            idx = rng.integers(0, len(a), size=(n_boot, len(a)))
            boot_mean[key] = a[idx].mean(axis=1)
            boot_zmean[key] = _fisher(a)[idx].mean(axis=1)

    q = lambda arr: tuple(float(v) for v in np.quantile(arr, [0.025, 0.975]))
    lines = [
        f"Paper statistics report  {header}".rstrip(),
        "Generated by paper_stats.py from the same loaders as plotting/figures.py.",
        "Sources: " + ", ".join(
            f"{r.code} survey={r.survey} / social={r.social}" for r in regions
        ),
    ]

    # Part 1 - example pair (Fig 2a, b)
    codes = {r.code for r in regions}
    pair = plotter._select_tutorial_pair() if {"US", "EU"} <= codes else None
    if pair is not None:
        usa, usb = pair["us_offline"]
        eua, eub = pair["eu_offline"]
        sca, scb = pair["social_codes"]
        rows = [
            ["US", "Survey", plotter.us_survey, f"{usa} x {usb}", f"{pair['us_off']:+.3f}", f"{abs(pair['us_off']):.3f}"],
            ["EU", "Survey", plotter.eu_survey, f"{eua} x {eub}", f"{pair['eu_off']:+.3f}", f"{abs(pair['eu_off']):.3f}"],
            ["US", "Social media", plotter.twitter, f"{sca} x {scb}", f"{pair['us_on']:+.3f}", f"{abs(pair['us_on']):.3f}"],
            ["EU", "Social media", plotter.eutwitter, f"{sca} x {scb}", f"{pair['eu_on']:+.3f}", f"{abs(pair['eu_on']):.3f}"],
        ]
        lines += _part(
            "PART 1. Example topic pair (Fig 2a, b)",
            [csv(plotter.us_survey), csv(plotter.eu_survey), csv(plotter.twitter), csv(plotter.eutwitter),
             'year == "average"; topic matching via figures.OFFLINE_ONLINE_TOPIC_MAP.'],
            [
                "Candidates: every pair of concepts present in both the US and EU topic sets.",
                "For each, look up signed r in the 4 sources; score = d_off / (d_on + 0.05), where",
                "d_off = | |r_US,survey| - |r_EU,survey| |, d_on = same for social media.",
                "The highest-scoring pair is the one drawn in Fig 2a, b (Plotter._select_tutorial_pair).",
                "Figure legends show |r| to 3 decimals.",
            ],
            _table(["Country", "Medium", "Source", "Topics", "signed r", "|r|"], rows)
            + [f"Selection score = {pair['score']:.3f} (d_off = {pair['delta_off']:.3f}, d_on = {pair['delta_on']:.3f})"],
        )

    # Part 2 - per-source distribution of |r|
    rows = []
    for medium, attr in MEDIA:
        for r in regions:
            key = (r.code, medium)
            a = absr[key]
            lo, hi = q(boot_mean[key])
            rows.append([
                r.code, medium, getattr(r, attr), str(len(a)), f"{a.mean():.3f}", _ci(lo, hi),
                f"{np.median(a):.3f}", f"{a.std(ddof=1):.3f}", f"{a.min():.3f}", f"{a.max():.3f}",
                f"{skew(a, bias=False):.3f}",
            ])
    lines += _part(
        "PART 2. Pairwise |r| per source: mean, CI, median, skewness (Fig 2e, f)",
        pair_source,
        [
            "Over the 36 pair |r| values of each source:",
            "mean; median; SD (ddof=1); min; max;",
            "skewness = bias-corrected sample skewness, scipy.stats.skew(bias=False).",
            boot_note + " CI is of the mean.",
        ],
        _table(["Country", "Medium", "Source", "Pairs", "Mean", "95% CI", "Median", "SD", "Min", "Max", "Skew"], rows),
    )

    # Part 3 - ratios
    if "US" in codes:
        rows = []
        for medium, _ in MEDIA:
            us = ("US", medium)
            for r in regions:
                if r.code == "US":
                    continue
                other = (r.code, medium)
                lo, hi = q(boot_mean[us] / boot_mean[other])
                rows.append([
                    f"US / {r.code}", medium, f"{absr[us].mean():.3f}", f"{absr[other].mean():.3f}",
                    f"{absr[us].mean() / absr[other].mean():.3f}", _ci(lo, hi),
                ])
        lines += _part(
            "PART 3. Ratio of mean |r|, US to other societies",
            ["Same 36-pair |r| values as Part 2."],
            [
                "ratio = mean|r|(US) / mean|r|(other), within the same medium.",
                boot_note + " In each resample the ratio of the two resampled means is taken.",
            ],
            _table(["Comparison", "Medium", "Mean US", "Mean other", "Ratio", "95% CI"], rows),
        )

    # Part 4 - dispersion and range
    disp: Dict[str, float] = {}
    boot_disp: Dict[str, np.ndarray] = {}
    rows = []
    for medium, _ in MEDIA:
        keys = [(r.code, medium) for r in regions]
        zmeans = np.array([_fisher(absr[k]).mean() for k in keys])
        means = np.array([absr[k].mean() for k in keys])
        disp[medium] = float(zmeans.std(ddof=0))
        boot_disp[medium] = np.stack([boot_zmean[k] for k in keys]).std(axis=0, ddof=0)
        lo, hi = q(boot_disp[medium])
        hi_code, lo_code = keys[means.argmax()][0], keys[means.argmin()][0]
        rows.append([
            medium,
            ", ".join(f"{k[0]}={z:.3f}" for k, z in zip(keys, zmeans)),
            f"{disp[medium]:.3f}", _ci(lo, hi),
            f"{means.max() - means.min():.3f} ({hi_code} {means.max():.3f} - {lo_code} {means.min():.3f})",
        ])
    lo, hi = q(boot_disp["Survey"] / boot_disp["Social media"])
    lines += _part(
        "PART 4. Cross-societal dispersion and range of mean |r|",
        ["Same 36-pair |r| values as Part 2."],
        [
            "Fisher z of each pair: z = arctanh(|r|).",
            "Dispersion (per medium) = population SD (ddof=0; the societies are fixed cases) across",
            "societies of each society's mean z: D = sqrt(sum_s (z_s - mean z)^2 / n_societies).",
            "Dispersion ratio = dispersion(Survey) / dispersion(Social media).",
            "Range (original correlation scale) = max - min across societies of mean |r|.",
            boot_note + " Dispersion and its ratio are recomputed in each resample.",
        ],
        _table(["Medium", "Mean z per society", "Dispersion", "95% CI", "Range of mean |r|"], rows)
        + [f"Dispersion ratio (survey / social media) = "
           f"{disp['Survey'] / disp['Social media']:.3f}, 95% CI {_ci(lo, hi)}"],
    )

    # Part 5 - eta^2
    rows = []
    for medium, _ in MEDIA:
        keys = [(r.code, medium) for r in regions]
        ss_b, ss_t, eta = _eta_sq([_fisher(absr[k]) for k in keys])
        rows.append([medium, str(sum(len(absr[k]) for k in keys)), f"{ss_b:.3f}", f"{ss_t:.3f}",
                     f"{eta:.3f} ({100 * eta:.1f}%)"])
    lines += _part(
        "PART 5. Variance decomposition: share of variation due to society (eta^2)",
        ["Same 36-pair |r| values as Part 2."],
        [
            "Pool the Fisher z = arctanh(|r|) of all pairs of one medium across societies.",
            "eta^2 = SS_between / SS_total with society as the grouping factor:",
            "SS_between = sum_g n_g (mean_g - grand mean)^2; SS_total = sum (z - grand mean)^2.",
            "Descriptive only: no permutation test or p-value.",
        ],
        _table(["Medium", "Pairs pooled", "SS_between", "SS_total", "eta^2"], rows),
    )

    # Part 6 - spectral dimension (Fig 3)
    rows = []
    sources = []
    for medium, attr in MEDIA:
        for r in regions:
            src = getattr(r, attr)
            path = _spectral_path(src, plotter.spectral_matrix)
            if not path.exists():
                rows.append([r.code, medium, src, "missing", "", "", "", "", "", ""])
                continue
            summ = v4.load_spectral_summary(src, plotter.spectral_matrix)
            pr, er = float(np.mean(summ["PR"])), float(np.mean(summ["eRank"]))
            sources.append(display(path))
            boot = v4.load_spectral_bootstrap(src, plotter.spectral_matrix)
            if boot is None:
                pr_ci = norm_ci = er_ci = "no bootstrap"
            else:
                sources.append(display(path.with_name(path.name.replace("-summary", "-bootstrap"))))
                pr_ci = f"[{boot['PR_lo']:.3f}, {boot['PR_hi']:.3f}]"
                norm_ci = f"[{boot['PR_lo'] / N_TOPICS:.3f}, {boot['PR_hi'] / N_TOPICS:.3f}]"
                er_ci = f"[{boot['eRank_lo']:.3f}, {boot['eRank_hi']:.3f}]"
            rows.append([r.code, medium, src, str(len(summ["PR"])), f"{pr:.3f}", pr_ci,
                         f"{pr / N_TOPICS:.3f}", norm_ci, f"{er:.3f}", er_ci])
    lines += _part(
        "PART 6. Dimensionality: participation ratio and exponential rank (Fig 3)",
        sources + ["Keys \"PR\" and \"eRank\" (one entry per year in the source)."],
        [
            "Values computed upstream (dimensions.py) from the eigenvalues of the 9-topic pairwise-available",
            ("correlation matrix (covariance rescaled to unit variances)" if plotter.spectral_matrix == "corr"
             else "covariance matrix") + ": PR = (sum lambda)^2 / sum lambda^2; eRank = exp(entropy of lambda / sum lambda).",
            "Where a source has several yearly entries (Weibo), the mean is taken, as in Fig 3.",
            f"Normalized PR = PR / {N_TOPICS} (number of topics).",
            "95% CI: percentile bootstrap from spectral_bootstrap.py (rows = respondents / users",
            "resampled with replacement within each file; PR and eRank recomputed in each resample;",
            f"these are the Fig 3 error bars). Normalized CI = CI / {N_TOPICS}.",
        ],
        _table(["Country", "Medium", "Source", "Entries", "PR", "PR 95% CI", "PR/9", "PR/9 95% CI",
                "eRank", "eRank 95% CI"], rows),
    )

    # Part 7 - pairwise-complete overlap
    rows = []
    for medium, attr in MEDIA:
        for r in regions:
            n = tables[(r.code, medium)]["intersection"].to_numpy(int)
            rows.append([r.code, medium, getattr(r, attr), f"{n.min():,}", f"{int(np.median(n)):,}", f"{n.max():,}"])
    lines += _part(
        "PART 7. Pairwise-complete overlap counts across the 36 pairs",
        [s for s in pair_source] + ['Column "Intersection" = respondents/users with both topics observed.'],
        ["Minimum, median, maximum of Intersection over the 36 pairs of each source."],
        _table(["Country", "Medium", "Source", "Minimum", "Median", "Maximum"], rows),
    )

    # Part 8 - sample sizes
    rows, notes = [], []
    for label, src in SAMPLE_SIZE_SOURCES.items():
        res = _sample_size(src)
        if res is None:
            rows.append([label, src, "missing"])
            continue
        rows.append([label, src, f"{res[0]:,}"])
        notes.append(f"{src}: {res[1]}")
    lines += _part(
        "PART 8. Sample sizes",
        notes,
        [
            "Surveys: number of respondents (rows) in the prepared individual-level data.",
            "Social media: users in the matrix used for the Fig 3 spectral analysis.",
        ],
        _table(["Dataset", "Source", "N"], rows),
    )

    # Part 9 - gap eliminated by semantic similarity
    if "US" in codes:
        rows_main, rows_alt, sources = [], [], []
        rng9 = np.random.default_rng(SEED)
        for medium, attr in MEDIA:
            us_region = next(r for r in regions if r.code == "US")
            xu, yu = _similarity_pairs(getattr(us_region, attr), plotter.embedding_type,
                                       plotter.correlation_year(getattr(us_region, attr)))
            for r in regions:
                if r.code == "US":
                    continue
                xo, yo = _similarity_pairs(getattr(r, attr), plotter.embedding_type,
                                           plotter.correlation_year(getattr(r, attr)))
                raw = yu.mean() - yo.mean()
                bo, ao = np.polyfit(xo, yo, 1)
                remaining, share = _gap_closed(xu, yu, xo, yo, xu)
                boot = []
                for _ in range(n_boot):
                    iu = rng9.integers(0, len(xu), len(xu))
                    io = rng9.integers(0, len(xo), len(xo))
                    boot.append(_gap_closed(xu[iu], yu[iu], xo[io], yo[io], xu[iu])[1])
                lo, hi = q(np.array(boot))
                rows_main.append([
                    f"US - {r.code}", medium, f"{xu.mean():.3f}", f"{xo.mean():.3f}",
                    f"{yu.mean():.3f}", f"{yo.mean():.3f}", f"{raw:.3f}",
                    f"{(ao + bo * xu).mean():.3f}", f"{remaining:.3f}",
                    f"{100 * share:.1f}%", f"[{100 * lo:.1f}%, {100 * hi:.1f}%]",
                ])
                shared = np.linspace(max(xu.min(), xo.min()), min(xu.max(), xo.max()), 200)
                alt = [_gap_closed(xu, yu, xo, yo, ref) for ref in (xu, shared, np.r_[xu, xo])]
                rows_alt.append([f"US - {r.code}", medium] + [
                    f"{rem:.3f} ({100 * sh:.1f}%)" for rem, sh in alt
                ])
        for medium, attr in MEDIA:
            for r in regions:
                sources.append(
                    f"{csv(getattr(r, attr))} + similarity of {getattr(r, attr)} "
                    f"({plotter.embedding_type} embedding)"
                )
        lines += _part(
            "PART 9. How much of the US gap in mean |r| is eliminated by semantic similarity",
            sources + [
                "Pairs merged on topic_combination exactly as in Fig 2g, h",
                "(figures.load_pairwise_correlation + load_semantic_similarity).",
            ],
            [
                "Raw gap = mean|r|(US) - mean|r|(other), same medium.",
                "Fit each society's OLS line |r| = a + b * similarity (the Fig 2g, h lines).",
                "Main estimate (\"align the other society to the U.S. similarity distribution\"):",
                "  evaluate both lines at the U.S. pairs' similarity values;",
                "  gap at equal similarity = mean(US line - other line) at those values",
                "  (= mean|r|(US) - mean of the other society's line at U.S. similarities);",
                "  share eliminated = 1 - gap at equal similarity / raw gap.",
                boot_note + " Lines are refit in each resample.",
                "Sensitivity: the same calculation with other reference similarity values -",
                "  U.S. pairs (main), evenly spaced over the range both societies cover,",
                "  and all pairs of both societies pooled.",
                "Note: with a small raw gap (social media) the share is unstable and its CI wide.",
            ],
            _table(
                ["Comparison", "Medium", "Mean sim US", "Mean sim other", "Mean |r| US",
                 "Mean |r| other", "Raw gap", "Other line at US sims", "Gap at equal sim",
                 "Eliminated", "95% CI"],
                rows_main,
            )
            + ["", "Sensitivity - gap at equal similarity (share eliminated), by reference:"]
            + _table(["Comparison", "Medium", "US pairs (main)", "Shared range", "Pooled pairs"], rows_alt),
        )

    # Appendix - every pair
    lines += ["", RULE, "APPENDIX. |r| of every topic pair (for re-verification)", RULE]
    for medium, attr in MEDIA:
        for r in regions:
            df = tables[(r.code, medium)].sort_values("correlation", ascending=False)
            lines += ["", f"{r.code} {medium} ({getattr(r, attr)})"]
            lines += _table(
                ["Pair", "signed r", "|r|", "Fisher z", "N (intersection)"],
                [[t, f"{p:+.3f}", f"{a:.3f}", f"{np.arctanh(min(a, 1 - 1e-12)):.3f}", f"{n:,}"]
                 for t, p, a, n in zip(df["topic_combination"], df["pearson"], df["correlation"], df["intersection"])],
            )
    return "\n".join(lines) + "\n"


def write_report(
    plotter: "v4.Plotter",
    path: Path,
    n_boot: int = DEFAULT_N_BOOT,
    header: str = "",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(plotter, n_boot=n_boot, header=header))
    return path


def main(task: str = "main", n_boot: int = DEFAULT_N_BOOT) -> Path:
    """Write the stats report for one task without re-drawing the figures."""
    plotter = v4.Plotter(**v4.task_plotter_kwargs(task))
    path = v4.STATS_DIR / plotter._output_path(task, "stats").with_suffix(".txt").name
    return write_report(plotter, path, n_boot=n_boot, header=f"(task: {task})")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
