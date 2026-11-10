"""Identical survey items in both countries (two-country SI check; task identical_items_2c).

ROBUSTNESS ONLY. WVS wave 7 fielded one questionnaire in the United States
(2017) and China (2018). Two item sets are scored for both countries with
exactly the same code:

* ``cn_items`` - the nine Chinese WVS topics of the main analysis.
* ``us_items`` - the eight U.S.-style WVS items of the same-instrument check
  (robustness/wvs_us.py; common.topics.TOPICS_8["wvs_us8"]) less confidence in
  the WHO (Q88), which China did not field: seven items.

Chinese coding (SI "Topic measures"), rebuilt from the raw file; every topic
reproduces data/survey/wvs.dta exactly except gender equality, whose
five-component score is reproduced for 89% of respondents (the stored file
was prepared outside this repository). China is therefore recomputed here with
the rebuilt code, so both countries share it by construction.

    LGBT          Q182 1-10 -> 5..1                 Corrup   Q112 1-10 -> 5..1
    Marriage      Q185, Q186 1-10 -> 5..1, mean, half-up rounding
    Childbearing  Q37 1-5 reversed                  Econ     Q50 1-10 -> 1..5
    Work          Q142 1-4 -> 1, 2.33, 3.67, 5      Foreign  Q34 1-5 reversed
    Environment   Q111 env 1, other 3, growth 5
    GenderEqual   Q29 (1-4 -> 5, 3.67, 2.33, 1), Q30, Q31 (1-4 reversed),
                  Q33, Q35 (1-5 reversed); mean, half-up rounding

For each item set and country: mean |r| over all topic pairs (pairwise
complete, as analysis.correlation.survey_correlation) with a percentile
bootstrap over topic pairs (2,000 draws, seed 0, each country resampled
independently), the U.S.-Chinese gap and ratio with intervals from the same
draws; PR and eRank of the pairwise-available covariance (main estimator)
with a respondent bootstrap (2,000 draws, seed 20261008). Stats file and SI
figure:

    python -m robustness.identical_items --raw <WVS_Cross-National_Wave_7_csv>
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import sparse

from common.topics import CN_SURVEY_TOPICS, TOPICS_8
from robustness.wvs_us import ITEMS as US_ITEMS, recode, ten_to_five

FOUR_TO_FIVE = {1: 1.0, 2: 2.3333, 3: 3.6667, 4: 5.0}
CN_COLUMNS = ["Q182", "Q112", "Q185", "Q186", "Q37", "Q50", "Q142", "Q34", "Q111",
              "Q29", "Q30", "Q31", "Q33", "Q35"]
US8 = [str(t) for t in TOPICS_8["wvs_us8"]]
CN9 = [str(t) for t in CN_SURVEY_TOPICS]
COUNTRIES = {"US": "USA", "CN": "CHN"}
N_BOOT = 2000


def _pos(s: pd.Series) -> pd.Series:
    return s.where(s > 0).astype(float)


def _half_up(m: pd.Series) -> pd.Series:
    return np.floor(m + 0.5)


def cn_item_scores(df: pd.DataFrame) -> pd.DataFrame:
    """The nine Chinese topics, coded as in the main analysis."""
    gender = pd.concat([
        (5 - _pos(df["Q29"])).map(FOUR_TO_FIVE),
        5 - _pos(df["Q30"]), 5 - _pos(df["Q31"]),
        6 - _pos(df["Q33"]), 6 - _pos(df["Q35"]),
    ], axis=1).mean(axis=1)
    marriage = pd.concat([ten_to_five(df["Q185"], True), ten_to_five(df["Q186"], True)], axis=1).mean(axis=1)
    out = pd.DataFrame({
        "Corrup": ten_to_five(df["Q112"], reverse=True),
        "GenderEqual": _half_up(gender),
        "Marriage": _half_up(marriage),
        "Childbearing": 6 - _pos(df["Q37"]),
        "LGBT": ten_to_five(df["Q182"], reverse=True),
        "Environment": recode(df["Q111"], "environment"),
        "Econ": ten_to_five(df["Q50"]),
        "Work": _pos(df["Q142"]).map(FOUR_TO_FIVE),
        "Foreign": 6 - _pos(df["Q34"]),
    })
    return out[CN9]


def us_item_scores(df: pd.DataFrame) -> pd.DataFrame:
    """The eight U.S.-style WVS items of the same-instrument check."""
    return pd.DataFrame({t: recode(df[US_ITEMS[t][0]], US_ITEMS[t][1]) for t in US8})[US8]


SCORERS = {"cn_items": cn_item_scores, "us_items": us_item_scores}
DROPPED: Dict[str, List[str]] = {}


def load(raw: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """{item set: {country: respondents x topics}}: adults who answered any topic of the set."""
    cols = sorted(set(CN_COLUMNS) | {c for c, _ in US_ITEMS.values()})
    df = pd.read_csv(raw, usecols=["B_COUNTRY_ALPHA", "Q262", *cols], low_memory=False)
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for item_set, scorer in SCORERS.items():
        scored = {code: scorer(df[(df["B_COUNTRY_ALPHA"] == alpha) & (df["Q262"] >= 18)].reset_index(drop=True))
                  for code, alpha in COUNTRIES.items()}
        # Items not fielded in one country (China did not ask Q88, confidence
        # in the WHO) are dropped in both, so the item sets stay identical.
        keep = [t for t in scored["US"].columns if all(X[t].notna().any() for X in scored.values())]
        DROPPED[item_set] = [t for t in scored["US"].columns if t not in keep]
        out[item_set] = {c: X[keep].dropna(how="all").reset_index(drop=True) for c, X in scored.items()}
    return out


def validate_china(raw: str) -> Dict[str, float]:
    """Share of Chinese respondents whose rebuilt score equals data/survey/wvs.dta."""
    from common.paths import SURVEY_DIR

    df = pd.read_csv(raw, usecols=["B_COUNTRY_ALPHA", *CN_COLUMNS], low_memory=False)
    built = cn_item_scores(df[df["B_COUNTRY_ALPHA"] == "CHN"].reset_index(drop=True))
    stored = pd.read_stata(SURVEY_DIR / "wvs.dta", convert_categoricals=False)
    return {t: float((np.isclose(built[t], stored[t], atol=1e-3) | (built[t].isna() & stored[t].isna())).mean())
            for t in CN9}


def pair_table(X: pd.DataFrame) -> pd.DataFrame:
    """Signed Pearson r and pairwise-complete n of every topic pair."""
    rows = []
    for a, b in itertools.combinations(X.columns, 2):
        ok = X[[a, b]].dropna()
        rows.append({"pair": f"{a} x {b}", "r": float(np.corrcoef(ok[a], ok[b])[0, 1]), "n": len(ok)})
    return pd.DataFrame(rows)


def spectral(X: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 20261008) -> Dict[str, object]:
    """PR and eRank of the pairwise-available covariance, with a respondent bootstrap."""
    from analysis.dimension.spectral_bootstrap import spectral_metrics

    M = sparse.csr_matrix(X.fillna(0.0).to_numpy(float))   # scores are 1-5, so 0 = missing
    B = M.copy()
    B.data = np.ones_like(B.data)
    n = M.shape[0]
    point = spectral_metrics(M, B, np.ones(n))
    rng = np.random.default_rng(seed)
    reps = np.array([spectral_metrics(M, B, np.bincount(rng.integers(0, n, n), minlength=n).astype(float))
                     for _ in range(n_boot)])
    return {"PR": point[0], "eRank": point[1], "PR_ci": np.percentile(reps[:, 0], [2.5, 97.5]),
            "eRank_ci": np.percentile(reps[:, 1], [2.5, 97.5]), "K": X.shape[1]}


def analyse(data: Dict[str, Dict[str, pd.DataFrame]], n_boot: int = N_BOOT) -> Dict[str, Dict]:
    res = {}
    for item_set, by_country in data.items():
        rng = np.random.default_rng(0)
        absr, boot = {}, {}
        tables = {c: pair_table(X) for c, X in by_country.items()}
        for c, t in tables.items():
            a = t["r"].abs().to_numpy()
            absr[c] = a
            boot[c] = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
        gap = absr["US"].mean() - absr["CN"].mean()
        res[item_set] = {
            "tables": tables,
            "n": {c: len(X) for c, X in by_country.items()},
            "mean": {c: float(a.mean()) for c, a in absr.items()},
            "mean_ci": {c: np.percentile(b, [2.5, 97.5]) for c, b in boot.items()},
            "gap": float(gap), "gap_ci": np.percentile(boot["US"] - boot["CN"], [2.5, 97.5]),
            "ratio": float(absr["US"].mean() / absr["CN"].mean()),
            "ratio_ci": np.percentile(boot["US"] / boot["CN"], [2.5, 97.5]),
            "spectral": {c: spectral(X, n_boot) for c, X in by_country.items()},
        }
    return res


LABEL = {"cn_items": "Chinese WVS items", "us_items": "U.S.-style WVS items"}


def report(res: Dict[str, Dict], check: Dict[str, float], raw: str) -> str:
    L = ["Paper statistics report  (task: identical_items_2c)",
         "Generated by robustness/identical_items.py.", "",
         "SOURCE: " + raw + " (WVS wave 7, v6.0); B_COUNTRY_ALPHA USA (2017) / CHN (2018); Q262 >= 18;",
         "  respondents with at least one topic of the item set; negative codes missing; no weights.",
         "CODING: module docstring. Rebuilt Chinese scores vs data/survey/wvs.dta (share identical):",
         "  " + ", ".join(f"{t} {v:.3f}" for t, v in check.items()),
         "CALCULATION: |r| = |Pearson r| on pairwise-complete respondents; mean over all pairs.",
         "  Mean, gap (US - CN) and ratio (US / CN): percentile bootstrap, 2,000 resamples of each country's",
         "  topic pairs (independently; numpy seed 0). PR / eRank: pairwise-available covariance + ridge 1e-6",
         "  (analysis.dimension.spectral_bootstrap.spectral_metrics); 2,000 respondent resamples, seed 20261008.", ""]
    for item_set, r in res.items():
        k = r["spectral"]["US"]["K"]
        L += [f"=== {LABEL[item_set]} (K = {k}, {k * (k - 1) // 2} pairs)"
              + (f"; dropped (not fielded in both countries): {', '.join(DROPPED[item_set])}" if DROPPED.get(item_set) else ""),
              f"  {'Country':8s}{'N':>7s}  {'mean |r|':>9s}  {'95% CI':16s}  {'PR':>6s} {'95% CI':16s} {'PR/K':>6s}  "
              f"{'eRank':>6s} {'95% CI':16s} {'eRank/K':>7s}"]
        for c in ("US", "CN"):
            sp = r["spectral"][c]
            L.append(f"  {c:8s}{r['n'][c]:7d}  {r['mean'][c]:9.3f}  [{r['mean_ci'][c][0]:.3f}, {r['mean_ci'][c][1]:.3f}]"
                     f"    {sp['PR']:6.3f} [{sp['PR_ci'][0]:.3f}, {sp['PR_ci'][1]:.3f}] {sp['PR'] / k:6.3f}  "
                     f"{sp['eRank']:6.3f} [{sp['eRank_ci'][0]:.3f}, {sp['eRank_ci'][1]:.3f}] {sp['eRank'] / k:7.3f}")
        sU, sC = r["spectral"]["US"], r["spectral"]["CN"]
        L += [f"  Gap in mean |r| (US - CN) = {r['gap']:.3f} [{r['gap_ci'][0]:.3f}, {r['gap_ci'][1]:.3f}];"
              f" ratio US/CN = {r['ratio']:.3f} [{r['ratio_ci'][0]:.3f}, {r['ratio_ci'][1]:.3f}]",
              f"  Gap in PR/K = {abs(sU['PR'] - sC['PR']) / k:.3f}; gap in eRank/K = {abs(sU['eRank'] - sC['eRank']) / k:.3f}",
              ""]
        for c in ("US", "CN"):
            t = r["tables"][c].assign(a=lambda d: d["r"].abs()).sort_values("a", ascending=False)
            L.append(f"  Signed r, {c} ({LABEL[item_set]}):")
            L += [f"    {p:34s} {v:+.3f}  n={n:,}" for p, v, n in zip(t["pair"], t["r"], t["n"])]
            L.append("")
    return "\n".join(L) + "\n"


def figure(res: Dict[str, Dict], path: Path) -> Path:
    """Mean |r|, PR/K and eRank/K of both countries on each item set (dash layout)."""
    import matplotlib.pyplot as plt
    from plotting import figures as v4

    plt.rcParams["font.family"] = "Times New Roman"
    color = {"US": v4._TWITTER_COLOR, "CN": v4._WEIBO_COLOR}
    sets = list(res)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    panels = (("Mean |r|", lambda r, c: (r["mean"][c], r["mean_ci"][c])),
              ("PR / K", lambda r, c: (r["spectral"][c]["PR"] / r["spectral"][c]["K"], r["spectral"][c]["PR_ci"] / r["spectral"][c]["K"])),
              ("eRank / K", lambda r, c: (r["spectral"][c]["eRank"] / r["spectral"][c]["K"], r["spectral"][c]["eRank_ci"] / r["spectral"][c]["K"])))
    for ax, (title, get) in zip(axes, panels):
        for x, s in enumerate(sets):
            for c, dx in (("US", -0.12), ("CN", 0.12)):
                v, ci = get(res[s], c)
                ax.plot([x + dx - 0.08, x + dx + 0.08], [v, v], color=color[c], lw=2.6, label=c if x == 0 else None)
                ax.plot([x + dx, x + dx], ci, color=color[c], lw=1.2)
                ax.text(x + dx + 0.1, v, f"{v:.3f}", fontsize=8, color=color[c], va="center")
        ax.set_xticks(range(len(sets)), [f"{LABEL[s]}\n(K = {res[s]['spectral']['US']['K']})" for s in sets])
        ax.set_xlim(-0.5, len(sets) - 0.4)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        if title != "Mean |r|":
            ax.invert_yaxis()   # as main-text Fig 3: higher = fewer dimensions
    axes[0].legend(frameon=False, fontsize=9)
    for ax, letter in zip(axes, "ABC"):
        ax.text(-0.12, 1.05, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", fontstyle="italic")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def run(raw: str, n_boot: int = N_BOOT) -> List[Path]:
    from common.paths import FIGURE_DIR, STATS_DIR, date_prefix

    check = validate_china(raw)
    res = analyse(load(raw), n_boot)
    name = f"{date_prefix()}_identical_items_2c_results"
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    stats = STATS_DIR / f"{name}_stats.txt"
    stats.write_text(report(res, check, raw))
    fig = figure(res, FIGURE_DIR / f"{name}_fig.pdf")
    print(stats, fig, sep="\n")
    return [stats, fig]


if __name__ == "__main__":
    import fire

    fire.Fire(run)
