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
from scipy.stats import skew, spearmanr

from analysis.correlation import semantic_adjustment as sa
from analysis.correlation.pattern_agreement import repairing_test
from common.paths import display
from plotting import figures as v4

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


def _spectral_path(src: str, matrix: str = "cov", year: Optional[str] = None) -> Path:
    return v4.spectral_summary_path(src, matrix, year)


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
    csv = lambda src: display(v4.CORRELATION_DIR / f"network_analysis_{v4.parent_source(src)}.csv")
    n_topics = len(v4.RESTRICTED_TOPICS[regions[0].survey])
    n_pairs = n_topics * (n_topics - 1) // 2
    pair_source = [
        f"{csv(getattr(r, attr))}  ({r.code} {medium.lower()})"
        for medium, attr in MEDIA for r in regions
    ] + [
        ('Rows with year == "average" (each user pooled over all years)' if plotter.social_year == "average"
         else f'Social media: rows with year == "{plotter.social_year}" (opinions of that year only); surveys: "average"')
        + f"; only the {n_topics} topics of each source",
        f"(figures.RESTRICTED_TOPICS) -> {n_pairs} topic pairs. |r| = abs(\"Correlation (Pearson)\").",
        "Loaded with figures.load_pairwise_correlation (same as Fig 2).",
    ]
    boot_note = (
        f"95% CI: percentile bootstrap, {n_boot:,} resamples of the {n_pairs} topic pairs "
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
            f"Over the {n_pairs} pair |r| values of each source:",
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
            [f"Same {n_pairs}-pair |r| values as Part 2."],
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
        [f"Same {n_pairs}-pair |r| values as Part 2."],
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
        [f"Same {n_pairs}-pair |r| values as Part 2."],
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
            year = plotter.spectral_year(src)
            path = _spectral_path(src, plotter.spectral_matrix, year)
            if not path.exists():
                rows.append([r.code, medium, src, "missing", "", "", "", "", "", ""])
                continue
            summ = v4.load_spectral_summary(src, plotter.spectral_matrix, year)
            pr, er = float(np.mean(summ["PR"])), float(np.mean(summ["eRank"]))
            k = len(v4.RESTRICTED_TOPICS[src])
            sources.append(display(path))
            boot = v4.load_spectral_bootstrap(src, plotter.spectral_matrix, year)
            if boot is None:
                pr_ci = norm_ci = er_ci = "no bootstrap"
            else:
                sources.append(f"bootstrap of {src}" + (f" ({year})" if year else ""))
                pr_ci = f"[{boot['PR_lo']:.3f}, {boot['PR_hi']:.3f}]"
                norm_ci = f"[{boot['PR_lo'] / k:.3f}, {boot['PR_hi'] / k:.3f}]"
                er_ci = f"[{boot['eRank_lo']:.3f}, {boot['eRank_hi']:.3f}]"
            rows.append([r.code, medium, src, str(len(summ["PR"])), f"{pr:.3f}", pr_ci,
                         f"{pr / k:.3f}", norm_ci, f"{er:.3f}", er_ci])
    lines += _part(
        "PART 6. Dimensionality: participation ratio and exponential rank (Fig 3)",
        sources + ["Keys \"PR\" and \"eRank\" (one entry per year in the source)."],
        [
            "Values computed upstream (dimensions.py) from the eigenvalues of the 9-topic pairwise-available",
            ("correlation matrix (covariance rescaled to unit variances)" if plotter.spectral_matrix == "corr"
             else "covariance matrix") + ": PR = (sum lambda)^2 / sum lambda^2; eRank = exp(entropy of lambda / sum lambda).",
            ("Where a source has several yearly entries (Weibo), the mean is taken, as in Fig 3."
             if plotter.social_year == "average" else
             f"Social media: the {plotter.social_year} matrix only (<stem>-{plotter.social_year}.csr-none.json; "
             f"bootstrap <stem>-none-{plotter.social_year}-bootstrap.json, or the main file if it covers only that year)."),
            f"Normalized PR = PR / K, K = number of topics of the source ({n_topics} here).",
            "95% CI: percentile bootstrap from spectral_bootstrap.py (rows = respondents / users",
            "resampled with replacement within each file; PR and eRank recomputed in each resample;",
            f"these are the Fig 3 error bars). Normalized CI = CI / K.",
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
        f"PART 7. Pairwise-complete overlap counts across the {n_pairs} pairs",
        [s for s in pair_source] + ['Column "Intersection" = respondents/users with both topics observed.'],
        [f"Minimum, median, maximum of Intersection over the {n_pairs} pairs of each source."],
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

    # Part 9 - mean |r| with semantic similarity held fixed (Fig 2G)
    ref = plotter.semantic_reference()
    sem = plotter.semantic_adjustment()
    sem_boot_note = (
        f"95% CI: percentile bootstrap, {plotter.n_bootstrap:,} resamples of each source's topic pairs "
        f"(with replacement, independently); numpy seed {plotter.SEMANTIC_SEED}; weights / regressions "
        "recomputed in each resample (the Fig 2G and Fig 2s intervals)."
    )
    sem_sources = [
        f"{csv(getattr(r, attr))} + similarity of {getattr(r, attr)} ({plotter.embedding_type} embedding)"
        for medium, attr in MEDIA for r in regions
    ] + ["Pairs merged on topic_combination (figures.similarity_pairs)."]
    rows_main, rows_gap, rows_alt = [], [], []
    for medium, attr in MEDIA:
        ref_src = getattr(ref, attr)
        xr, yr = v4.similarity_pairs(ref_src, plotter.embedding_type, plotter.correlation_year(ref_src))
        for r in regions:
            src = getattr(r, attr)
            x, y = v4.similarity_pairs(src, plotter.embedding_type, plotter.correlation_year(src))
            lo, hi = sem["adjusted_ci"][src]
            ess = "" if r is ref else f"{sa.effective_sample_size(sa.density_ratio_weights(xr, x)):.1f}"
            rows_main.append([
                r.code, medium, src, f"{x.mean():.3f}", f"{sem['raw'][src]:.3f}",
                f"{sem['adjusted'][src]:.3f}", _ci(lo, hi), ess,
            ])
            if r is ref:
                continue
            d = sa.gap_decomposition(xr, yr, x, y)
            rows_gap.append([
                f"{ref.code} - {r.code}", medium, f"{d['raw_gap']:.3f}",
                f"{d['similarity_part']:+.3f}", f"{d['society_part']:.3f}",
                f"{100 * d['society_part'] / d['raw_gap']:.0f}%",
            ])
            strata = plotter.semantic_adjustment("strata")
            lo_s, hi_s = strata["adjusted_ci"][src]
            rows_alt.append([
                r.code, medium, f"{sem['adjusted'][src]:.3f}", _ci(lo, hi),
                f"{strata['adjusted'][src]:.3f}", _ci(lo_s, hi_s),
            ])
    support = plotter.semantic_adjustment("support")
    rows_sup = []
    for medium, attr in MEDIA:
        for r in regions:
            src = getattr(r, attr)
            lo_b, hi_b = plotter.semantic_support(src)
            x, _ = plotter.semantic_pairs(src, support=True)
            w = plotter.semantic_weights(src, support=True)
            m_w = w @ x
            lo, hi = support["adjusted_ci"][src]
            rows_sup.append([
                r.code, medium, f"[{lo_b:.3f}, {hi_b:.3f}]", f"{len(x)}/{len(plotter.semantic_pairs(src)[0])}",
                f"{m_w:.3f}", f"{np.sqrt(w @ (x - m_w) ** 2):.3f}",
                "" if r is ref else f"{sa.effective_sample_size(w):.1f}",
                f"{sem['adjusted'][src]:.3f}", f"{support['adjusted'][src]:.3f}", _ci(lo, hi),
            ])
    lines += _part(
        f"PART 9. Mean |r| with semantic similarity held fixed at the {ref.code} distribution (Fig 2G)",
        sem_sources,
        [
            f"Within each medium, reweight every other society's {n_pairs} pairs so their semantic similarities are",
            f"distributed like the {ref.code} pairs' (DiNardo-Fortin-Lemieux reweighting):",
            f"  weight_j = f_{ref.code}(sim_j) / f_other(sim_j), both densities Gaussian KDEs (Scott bandwidth),",
            "  weights normalised to sum to 1; adjusted mean |r| = sum_j weight_j * |r|_j.",
            f"The {ref.code} value is its raw mean (it is the reference). ESS = Kish effective number of pairs",
            f"  (sum w)^2 / sum w^2, out of {n_pairs}: how far the weights are from uniform.",
            sem_boot_note,
            f"Gap decomposition: raw gap = mean|r|({ref.code}) - mean|r|(other) =",
            "  similarity mix (adjusted - raw mean of the other society; what the gap would change by if the",
            f"  other society's topics were as similar as the {ref.code} topics, with sign) +",
            f"  society ({ref.code} - adjusted; the gap in |r| between topic pairs of equal similarity).",
            f"Weighting, not matching: with {n_pairs} pairs per society, matching would let a few pairs stand in for many.",
            f"{ref.code} pairs beyond the other society's similarity range cannot be reproduced by any weighting, so",
            f"  the reweighted similarity distribution is close to, not identical with, the {ref.code} one (Part 11).",
            f"Sensitivity - quintile strata: {ref.code} similarity quintiles, each weighted by its {ref.code} share",
            f"  (other pairs outside the {ref.code} range dropped); same bootstrap.",
            "Robustness - common support (Fig 2s support): within each medium keep only the pairs whose similarity",
            "  lies in the range every society covers [largest minimum, smallest maximum], for all societies",
            f"  including the {ref.code}; then the same density-ratio reweighting to the kept {ref.code} pairs. The {ref.code}",
            "  value is the mean |r| of its kept pairs. Bounds fixed from the observed pairs; bootstrap resamples",
            "  drawn from the kept pairs. Reweighted mean / SD of similarity: weighted moments of the kept pairs",
            f"  (the {ref.code} row gives the target).",
        ],
        _table(["Country", "Medium", "Source", "Mean sim", "Raw mean |r|", "Adjusted mean |r|", "95% CI",
                "ESS"], rows_main)
        + ["", "Gap decomposition (main method):"]
        + _table(["Comparison", "Medium", "Raw gap", "Similarity mix", "Society", "Society share"], rows_gap)
        + ["", "Sensitivity - adjusted mean |r| by method:"]
        + _table(["Country", "Medium", "Density ratio (main)", "95% CI", "Quintile strata", "95% CI"],
                 rows_alt)
        + ["", "Robustness - common support:"]
        + _table(["Country", "Medium", "Support", "Pairs kept", "Reweighted sim mean", "Reweighted sim SD",
                  "ESS", "Main adjusted |r|", "Common-support |r|", "95% CI"], rows_sup),
    )

    # Part 10 - Shapley decomposition (text only, no figure panel)
    rows, rows_alt = [], []
    for medium, attr in MEDIA:
        sources = [getattr(r, attr) for r in regions]
        data = [v4.similarity_pairs(s, plotter.embedding_type, plotter.correlation_year(s)) for s in sources]
        groups = np.concatenate([np.full(len(x), i) for i, (x, _) in enumerate(data)])
        x_all = np.concatenate([x for x, _ in data])
        r_all = np.concatenate([y for _, y in data])
        d = sem["decomposition"][medium]
        ci = sem["decomposition_ci"][medium]
        rows.append([
            medium, str(len(r_all)), f"{100 * d['society']:.1f}%", _ci(100 * ci["society"][0], 100 * ci["society"][1], 1),
            f"{100 * d['similarity']:.1f}%", _ci(100 * ci["similarity"][0], 100 * ci["similarity"][1], 1),
            f"{100 * d['total']:.1f}%", f"{100 * d['unexplained']:.1f}%",
        ])
        raw = sa.shapley_r2(r_all, groups, x_all)
        rows_alt.append([
            medium,
            f"{100 * d['society_alone']:.1f}% / {100 * (d['total'] - d['similarity_alone']):.1f}%",
            f"{100 * d['similarity_alone']:.1f}% / {100 * (d['total'] - d['society_alone']):.1f}%",
            f"{100 * raw['society']:.1f}% / {100 * raw['similarity']:.1f}%",
        ])
    lines += _part(
        "PART 10. Variance decomposition: society vs semantic similarity (Shapley R^2; text, no figure)",
        ["Same pairs as Part 9."],
        [
            "Pool the pairs of one medium across societies; outcome Fisher z = arctanh(|r|) (as in Part 5).",
            "Linear regression on society (dummies) and semantic similarity (linear). R^2 of each subset:",
            "  R^2(society), R^2(similarity), R^2(both).",
            "Shapley share of a factor = mean of its R^2 gain over the two entry orders, e.g.",
            "  society = [R^2(society) + R^2(both) - R^2(similarity)] / 2; the two shares add up to R^2(both).",
            "Unexplained = 1 - R^2(both) (topic-pair differences not captured by society or similarity).",
            "Society alone = eta^2 of Part 5.",
            sem_boot_note,
            "Sensitivity: each factor entered first / last (the two bounds the Shapley share averages),",
            "  and the same decomposition on |r| instead of Fisher z.",
        ],
        _table(["Medium", "Pairs", "Society", "95% CI", "Similarity", "95% CI", "R^2 both", "Unexplained"], rows)
        + ["", "Sensitivity:"]
        + _table(["Medium", "Society first / last", "Similarity first / last", "|r| scale: society / similarity"],
                 rows_alt),
    )

    # Part 11 - supplementary Figs 2s (reweighting in detail)
    rows_sim, rows_ols = [], []
    for medium, attr in MEDIA:
        for r in regions:
            src = getattr(r, attr)
            x, y = plotter.semantic_pairs(src)
            w = plotter.semantic_weights(src)
            q25, q50, q75 = np.quantile(x, [0.25, 0.5, 0.75])
            m_w = w @ x
            rows_sim.append([
                r.code, medium, f"{x.mean():.3f}", f"{x.std():.3f}", f"{x.min():.3f}", f"{q25:.3f}",
                f"{q50:.3f}", f"{q75:.3f}", f"{x.max():.3f}", f"{m_w:.3f}", f"{np.sqrt(w @ (x - m_w) ** 2):.3f}",
                "" if r is ref else f"{sa.effective_sample_size(w):.1f}",
            ])
            for label, weights in (("raw", None), ("reweighted", w)):
                if r is ref and label == "reweighted":
                    continue
                fit = sa.weighted_ols(x, y, weights)
                rho = spearmanr(x, y) if weights is None else None
                rows_ols.append([
                    r.code, medium, label, f"{fit['intercept']:.3f}", f"{fit['slope']:+.3f}", f"{fit['se']:.3f}",
                    f"{fit['p']:.3f}", f"{fit['r2']:.3f}",
                    f"{rho.statistic:+.2f}" if rho else "", f"{rho.pvalue:.2f}" if rho else "", str(len(x)),
                ])
    lines += _part(
        f"PART 11. Reweighting in detail (supplementary Figs 2s, OLS and LOESS versions)",
        ["Same pairs and weights as Part 9."],
        [
            "Similarity distribution of each source's pairs (Figs 2s a, b): mean, SD (ddof=0), quantiles; and the",
            f"  weighted mean / SD under the Part 9 weights (close to the {ref.code} values when the reweighting",
            f"  works; the {ref.code} row is the reference). ESS as in Part 9.",
            "Pair |r| raw vs reweighted (Figs 2s c, d): the means are Part 9's raw and adjusted mean |r|.",
            "|r| = intercept + slope * similarity (Figs 2s e-h, OLS version), statsmodels WLS: equal weights",
            "  (raw pairs) or the Part 9 weights (reweighted); slope SE and two-sided p from the t distribution",
            "  (WLS treats the weights as known); weighted R^2.",
            "Spearman rho of |r| and similarity over each source's pairs (raw pairs only; scipy.stats.spearmanr,",
            "  two-sided p) - the rank correlations quoted in the main text.",
            f"The LOESS version draws a local-linear fit (tricube kernel over the nearest {plotter.loess_frac:.0%} of",
            "  the pairs, kernel times the pair weights, no robustness iterations) in place of the lines;",
            "  descriptive only, no numbers reported.",
        ],
        ["Similarity distribution:"]
        + _table(["Country", "Medium", "Mean", "SD", "Min", "Q25", "Median", "Q75", "Max",
                  "Reweighted mean", "Reweighted SD", "ESS"], rows_sim)
        + ["", "OLS |r| ~ similarity, raw and reweighted pairs:"]
        + _table(["Country", "Medium", "Pairs used", "Intercept", "Slope", "SE", "p", "R^2", "Spearman rho", "p",
                  "N"], rows_ols),
    )

    # Part 12 - topic-specific agreement between the U.S. and European sources
    if {"US", "EU"} <= codes:
        concepts = plotter._us_eu_concept_pairs()
        shared = sorted({c for e in concepts for c in e["social_codes"]})
        us_off = {e["social_codes"][0]: e["us_offline"][0] for e in concepts}
        us_off.update({e["social_codes"][1]: e["us_offline"][1] for e in concepts})
        eu_off = {e["social_codes"][0]: e["eu_offline"][0] for e in concepts}
        eu_off.update({e["social_codes"][1]: e["eu_offline"][1] for e in concepts})

        def pair_values(src, rename=None):
            df = v4.load_pairwise_correlation(src, year=plotter.correlation_year(src))
            back = {v: k for k, v in (rename or {}).items()}
            return {frozenset((back.get(str(a), str(a)), back.get(str(b), str(b)))): float(r)
                    for a, b, r in zip(df["tid1"], df["tid2"], df["pearson"])}

        rows = []
        for medium, us_src, eu_src, us_map, eu_map in (
            ("Survey", plotter.us_survey, plotter.eu_survey, us_off, eu_off),
            ("Social media", plotter.twitter, plotter.eutwitter, None, None),
        ):
            res = repairing_test(pair_values(us_src, us_map), pair_values(eu_src, eu_map), shared,
                                 n_perm=5000, seed=SEED)
            rows.append([
                medium, f"{us_src} / {eu_src}", str(res["n_pairs"]), f"{res['diff']:.3f}",
                f"{res['diff_random']:.3f}", f"{res['p_diff']:.3f}", f"{res['corr']:.2f}",
                f"{res['corr_random']:.2f}", f"{res['p_corr']:.3f}",
            ])
        lines += _part(
            "PART 12. Topic-specific agreement of U.S. and European pair correlations (random re-pairing)",
            [f"Same {n_pairs}-pair |r| values as Part 2; the {len(shared)} topic concepts shared by the U.S. and European",
             "sources (figures.OFFLINE_ONLINE_TOPIC_MAP: Twitter/X codes; survey items mapped to them)",
             f"-> {len(shared) * (len(shared) - 1) // 2} pairs."],
            [
                f"For each medium, the {len(shared) * (len(shared) - 1) // 2} concept pairs' |r| in the U.S. source vs the European source:",
                "  mean |difference| and the Pearson correlation of the two patterns (matched).",
                f"Random re-pairing: relabel the European concepts by a random permutation of the {len(shared)} concepts and",
                f"  recompute both quantities; 5,000 permutations, numpy seed {SEED}. p = share of permutations at least",
                "  as close as the matched pairing (difference <= matched; correlation >= matched).",
                "If the sources were alike only in level (all pairs compressed to a common value), random re-pairing",
                "  would agree as well as the matched pairing.",
            ],
            _table(["Medium", "Sources", "Pairs", "Mean |diff|", "Random mean |diff|", "p",
                    "Pattern r", "Random pattern r", "p"], rows),
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
