"""Effective dimensionality (PR / eRank / srank) of each respondent x topic matrix.

1. transform: opinions -> data/dimension/csr/<stem>/<year>.csr.npz
   (rows = respondents/users, columns = the source's 9 topics, missing = absent)
2. calculate: dimensions.estimate() per csr file ->
   data/dimension/results/<matrix>/<stem>-<year>.csr-none.json, -loadings.json and the
   <stem>-none-summary.json that the figures read.

Stems: survey file names (anes, evs_resample2, ...), twitter-<us|eu|en|eu_nen>, weibo.

Usage (from the repository root):
    python -m analysis.dimension.dimension_pipeline transform            # surveys
    python -m analysis.dimension.dimension_pipeline transform --social   # + Twitter/Weibo (cluster)
    python -m analysis.dimension.dimension_pipeline calculate anes
    python -m analysis.dimension.dimension_pipeline calculate twitter --mode eu
    python -m analysis.dimension.dimension_pipeline calculate_all
Then bootstrap intervals: python -m analysis.dimension.spectral_bootstrap all
"""

import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
import json
import os
import fire
import pandas as pd
import numpy as np
from scipy import sparse


twitter_using_topics = [
    "abo",
    "gun",
    "clc",
    "sxo",
    "vac",
    "dpp",
    "soc",
    "minwage",
    "ubi",
]
eu_twitter_using_topics = [
    "abo",
    "swe",
    "clc",
    "sxo",
    "vac",
    "dpp",
    "soc",
    "minwage",
    "ubi",
]
weibo_using_topics = [7, 9, 11, 12, 10, 13, 15, 14, 0]
us_survey_using_topics = [
    "Abortion",
    "Gun",
    "Climate",
    "LGBT",
    "VACC",
    "DeathPenalty",
    "Media",
    "MinimumWage",
    "UBI",
]
cn_survey_using_topics = [
    "Corrup",
    "GenderEqual",
    "Marriage",
    "Childbearing",
    "LGBT",
    "Environment",
    "Econ",
    "Work",
    "Foreign",
]
eu_survey_using_topics = [
    "LGBT",
    "Abortion",
    "DeathPenalty",
    "SocialMedia",
    "Egalitarian",
    "Environment",
    "Prostitution",
    "HealthCare",
    "UnemplyAid",
]

DISTANCE_SPACES = ("spectral", "raw", "bagged")

from config import cfg
from common.paths import CSR_DIR, SURVEY_DIR, dimension_results_dir, report_dir

# Per-year results (<stem>-<year>.csr-none.json, -loadings.json), the
# <stem>-none-summary.json read by the figures, and run logs.

# Survey stems; each reads data/survey/<stem>.dta.
SURVEY_STEMS = [
    "anes", "anes_media", "wvs", "wvs_media",
    "evs", "evs_media", "evs_resample2", "evs_media_resample2",
    "en_evs", "en_evs_media", "eu_nen_evs",
]
TWITTER_MODES = ["us", "eu", "en", "eu_nen"]

original_data_map = {stem: str(SURVEY_DIR / f"{stem}.dta") for stem in SURVEY_STEMS}
# Cleaned social-media opinions (cleaning stage, on the clusters):
#   <TWITTER_OPINION_DIR>/<mode>/merged-<topic>.parquet
#   <WEIBO_OPINION_DIR>/<topic_id>/avg_opinion.parquet
original_data_map["twitter"] = cfg.TWITTER_OPINION_DIR
original_data_map["weibo"] = cfg.WEIBO_OPINION_DIR

# data/dimension/csr/<stem>/<year>.csr.npz; twitter stems are twitter-<mode>.
npz_data_folder = CSR_DIR
os.makedirs(npz_data_folder, exist_ok=True)


def csr_folder(dataset_type: str, mode: str = "us") -> Path:
    if dataset_type == "twitter":
        return npz_data_folder / f"twitter-{mode}"
    return npz_data_folder / dataset_type


# ---------------- Topic-label lookups for loading reports ----------------
# Mirrors the column ordering used when each dataset's npz was written.
_DATASET_TOPICS: Dict[str, list] = {
    "anes": us_survey_using_topics,
    "anes_media": us_survey_using_topics,
    "wvs": cn_survey_using_topics,
    "wvs_media": cn_survey_using_topics,
    "evs": eu_survey_using_topics,
    "evs_media": eu_survey_using_topics,
    "evs_resample2": eu_survey_using_topics,
    "evs_media_resample2": eu_survey_using_topics,
    "en_evs": eu_survey_using_topics,
    "en_evs_media": eu_survey_using_topics,
    "eu_nen_evs": eu_survey_using_topics,
    "weibo": weibo_using_topics,
}

_TWITTER_TOPICS_BY_MODE: Dict[str, list] = {
    "us": twitter_using_topics,
    "eu": eu_twitter_using_topics,
    "en": eu_twitter_using_topics,
    "eu_nen": eu_twitter_using_topics,
}


def _topics_for(dataset_type: str, mode: str = "us") -> List[str]:
    """Return the column-aligned topic names for one dataset/mode."""
    if dataset_type == "twitter":
        names = _TWITTER_TOPICS_BY_MODE[mode]
    else:
        names = _DATASET_TOPICS[dataset_type]
    return [str(t) for t in names]


def _loading_report(csr_path: Path, topic_names: List[str], standardize: bool = False) -> Dict[str, Any]:
    """Build a JSON-serializable PCA loading report for one CSR file.

    Re-runs pairwise covariance eigendecomp (cheap for L<=20) and returns
    per-PC eigenvalue, variance explained, cumulative variance, signed
    loadings keyed by topic (in column order), and the top-3 contributors
    by |loading|. Components are sorted by eigenvalue descending.
    """
    from analysis.dimension.dimensions import (
        pairwise_cov_eigendecomp,
    )  # local import to avoid heavy import at module load

    csr = sparse.load_npz(csr_path)
    lam, V = pairwise_cov_eigendecomp(csr, standardize=standardize)

    order = np.argsort(lam)[::-1]
    lam = lam[order]
    V = V[:, order]

    L = csr.shape[1]
    if len(topic_names) != L:
        # Fall back to generic names so the report still serializes safely.
        topic_names = [f"topic_{i}" for i in range(L)]

    pos_total = float(np.sum(lam[lam > 0]))
    cumulative = 0.0
    components: List[Dict[str, Any]] = []
    for k in range(len(lam)):
        eig = float(lam[k])
        var_frac = float(eig / pos_total) if (pos_total > 0 and eig > 0) else 0.0
        cumulative += var_frac
        loadings = V[:, k].astype(float).tolist()
        order_by_abs = sorted(
            range(len(loadings)), key=lambda i: abs(loadings[i]), reverse=True
        )
        components.append(
            {
                "pc": k + 1,
                "eigenvalue": eig,
                "variance_explained": var_frac,
                "cumulative_variance_explained": cumulative,
                "loadings": {topic_names[i]: loadings[i] for i in range(L)},
                "top_contributors": [
                    {"topic": topic_names[i], "loading": loadings[i]}
                    for i in order_by_abs[:3]
                ],
            }
        )

    return {
        "topics": list(topic_names),
        "N": int(csr.shape[0]),
        "L": int(L),
        "components": components,
    }


def _plot_loadings_heatmap(report: Dict[str, Any], out_path: Path, title: str) -> None:
    """Save a topics × PCs signed-loading heatmap to PDF.

    Cells annotated with loading values; diverging RdBu colormap centered at 0.
    """
    import matplotlib.pyplot as plt  # local import: only needed when plotting

    topics = report["topics"]
    components = report["components"]
    L = len(topics)
    K = len(components)
    if L == 0 or K == 0:
        return

    M = np.zeros((L, K))
    for k, comp in enumerate(components):
        for i, t in enumerate(topics):
            M[i, k] = comp["loadings"][t]

    fig_w = max(5.0, 0.75 * K + 2.5)
    fig_h = max(4.0, 0.42 * L + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = max(abs(M.min()), abs(M.max()), 0.1)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(K))
    ax.set_xticklabels(
        [f"PC{c['pc']}\n{c['variance_explained'] * 100:.1f}%" for c in components]
    )
    ax.set_yticks(range(L))
    ax.set_yticklabels(topics)
    ax.set_title(title)

    for i in range(L):
        for k in range(K):
            v = M[i, k]
            color = "white" if abs(v) > 0.55 * vmax else "black"
            ax.text(
                k, i, f"{v:+.2f}", ha="center", va="center", color=color, fontsize=8
            )

    fig.colorbar(im, ax=ax, label="signed loading")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_variance_scree(
    report: Dict[str, Any],
    scalars: Dict[str, Any],
    out_path: Path,
    title: str,
) -> None:
    """Save a scree + cumulative-variance plot to PDF."""
    import matplotlib.pyplot as plt  # local import: only needed when plotting

    components = report["components"]
    K = len(components)
    if K == 0:
        return
    var_frac = np.array([c["variance_explained"] for c in components]) * 100.0
    cum = np.array([c["cumulative_variance_explained"] for c in components]) * 100.0

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    xs = np.arange(1, K + 1)
    ax.bar(xs, var_frac, color="#4477AA", alpha=0.85, label="variance explained")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_xticks(xs)
    ax.set_ylim(0, max(100.0, float(var_frac.max()) * 1.1))

    ax2 = ax.twinx()
    ax2.plot(xs, cum, color="#CC6677", marker="o", label="cumulative")
    ax2.set_ylabel("Cumulative variance (%)")
    ax2.set_ylim(0, 105)

    subtitle_bits = []
    for key, fmt in (
        ("PR", "PR={:.2f}"),
        ("eRank", "eRank={:.2f}"),
        ("srank", "srank={:.2f}"),
    ):
        val = scalars.get(key)
        if val is not None and np.isfinite(val):
            subtitle_bits.append(fmt.format(val))
    if subtitle_bits:
        ax.set_title(f"{title}\n{', '.join(subtitle_bits)}")
    else:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _format_loading_summary(
    dataset_type: str,
    mode: str,
    distance_space: str,
    year_reports: List[Tuple[str, Dict[str, Any], Dict[str, Any]]],
) -> str:
    """Human-readable per-dataset summary across years.

    year_reports: list of (year_label, loading_report, dimensions_estimated).
    """
    lines: List[str] = []
    header = f"=== Loading report: {dataset_type}"
    if dataset_type == "twitter":
        header += f"/{mode}"
    header += f" (distance_space={distance_space}) ==="
    lines.append(header)

    if year_reports:
        topics = year_reports[0][1]["topics"]
        lines.append(f"\nTopics (L={len(topics)}): {', '.join(topics)}")

    for year_label, report, scalars in year_reports:
        lines.append(f"\n--- {year_label} (N={report['N']:,}) ---")
        scal_bits = []
        for key, fmt in (
            ("PR", "PR={:.2f}"),
            ("eRank", "eRank={:.2f}"),
            ("srank", "srank={:.2f}"),
        ):
            val = scalars.get(key)
            if val is not None and np.isfinite(val):
                scal_bits.append(fmt.format(val))
        if scal_bits:
            lines.append(", ".join(scal_bits))

        for comp in report["components"]:
            lines.append(
                f"  PC{comp['pc']} | eigval={comp['eigenvalue']:+.4f} | "
                f"var={comp['variance_explained'] * 100:5.1f}% | "
                f"cum={comp['cumulative_variance_explained'] * 100:5.1f}%"
            )
            sorted_items = sorted(
                comp["loadings"].items(), key=lambda kv: abs(kv[1]), reverse=True
            )
            for topic, loading in sorted_items:
                lines.append(f"      {loading:+.3f}  {topic}")

    return "\n".join(lines) + "\n"


def convert_and_save_survey_data(input_file, output_dir, restricted_topics):
    """
    读取 Stata 文件，转换为稀疏矩阵格式并保存

    Parameters:
    -----------
    input_file : str
        输入的 .dta 文件路径
    output_dir : str
        输出目录
    """
    name = Path(input_file).stem
    df = pd.read_stata(input_file)
    if "evs" in name:
        if name.startswith("eu_nen_evs"):
            df = df[df["country"] != "Great Britain"]
        elif name.startswith("en_evs"):
            df = df[df["country"] == "Great Britain"]
        else:
            pass
    if "poli_free" in df.columns:
        df = df.drop(columns=["poli_free"])
        print("已删除 poli_free 列")

    if "year" not in df.columns:
        df["year"] = 2020

    numeric_cols = (
        df[restricted_topics].select_dtypes(include=[np.number]).columns.tolist()
    )
    for year, df_year in df.groupby("year"):
        X = df_year[numeric_cols].to_numpy()
        r, c = np.where(np.isfinite(X))
        data = X[r, c].astype(float)
        csr = sparse.csr_matrix((data, (r, c)), shape=X.shape)
        output_file = output_dir / f"{year}.csr.npz"

        sparse.save_npz(output_file, csr)
        print(
            f"saved {output_file} with shape={csr.shape} nnz={csr.nnz} density={csr.nnz / (csr.shape[0] * csr.shape[1]):.4f}"
        )


def convert_and_save_twitter_data(mode="us"):
    """
    将不同 topic 的 twitter 数据按年份合并成一个数据框。

    逻辑：
    1. 遍历年份 2016-2023
    2. 对于每一年，遍历所有 topic，提取对应年份列并重命名为 topic 名
    3. 只保留所有 topic 都有数据的年份
    4. 对于某一年，将所有 topic 的数据按 author_id 进行 outer merge
    5. 给每个年份的 df 加上 year 列
    6. 将所有年份的 df concat 起来

    返回:
        DataFrame: 合并后的数据框，列包括：
            - 每个 topic 的 opinion 列（如 abo, gun, clc 等）
            - year 列
            - average 列（所有 topic 的平均值）
    """
    twitter_data_dir = Path(original_data_map["twitter"]) / mode
    years = list(range(2016, 2024))
    mode_available_topic_map = {
        "us": twitter_using_topics,
        "eu": eu_twitter_using_topics,
        "en": eu_twitter_using_topics,
        "eu_nen": eu_twitter_using_topics,
    }
    available_topics = mode_available_topic_map[mode]
    all_year_dfs = []
    for year in years:
        year_str = str(year)
        year_topic_data = {}
        for topic in available_topics:
            file_path = twitter_data_dir / f"merged-{topic}.parquet"
            df = pd.read_parquet(file_path)
            # 如果该 topic 有对应年份的列，提取并重命名为 topic 名
            if year_str in df.columns:
                cur_topic_df = (
                    df[[year_str]]
                    .dropna(subset=[year_str])
                    .rename(columns={year_str: topic})
                )
                year_topic_data[topic] = cur_topic_df
            else:
                break

        if len(year_topic_data) != len(available_topics):
            print(f"[WARN] Year {year} has less than {len(available_topics)} topics")
            continue
        year_df = None
        for topic, topic_df in year_topic_data.items():
            if year_df is None:
                year_df = topic_df
            else:
                year_df = year_df.merge(
                    topic_df, left_index=True, right_index=True, how="outer"
                )

        year_df["year"] = year
        all_year_dfs.append(year_df)
        print(f"[INFO] Processed year {year}: {len(year_df)} authors")

    merged_df = pd.concat(all_year_dfs, ignore_index=False)

    numeric_cols = (
        merged_df[available_topics].select_dtypes(include=[np.number]).columns.tolist()
    )
    for year, df_year in merged_df.groupby("year"):
        X = df_year[numeric_cols].to_numpy()
        r, c = np.where(np.isfinite(X))
        data = X[r, c].astype(float)
        csr = sparse.csr_matrix((data, (r, c)), shape=X.shape)

        output_dir = csr_folder("twitter", mode)
        os.makedirs(output_dir, exist_ok=True)
        output_file = output_dir / f"{year}.csr.npz"
        sparse.save_npz(output_file, csr)
        print(
            f"saved {output_file} with shape={csr.shape} nnz={csr.nnz} density={csr.nnz / (csr.shape[0] * csr.shape[1]):.4f}"
        )


def convert_and_save_weibo_data():
    """
    Convert and save weibo data into npz files.
    """
    weibo_data_dir = Path(original_data_map["weibo"])
    years = list(range(2016, 2024))
    all_year_dfs = []
    for year in years:
        year_str = str(year)
        year_topic_data = {}
        for topic in weibo_using_topics:
            file_path = weibo_data_dir / str(topic) / "avg_opinion.parquet"
            df = pd.read_parquet(file_path)
            # 如果该 topic 有对应年份的列，提取并重命名为 topic 名
            if year_str in df.columns:
                cur_topic_df = (
                    df[[year_str]]
                    .dropna(subset=[year_str])
                    .rename(columns={year_str: topic})
                )
                # drop the row where the value is NaN
                year_topic_data[topic] = cur_topic_df
            else:
                break

        if len(year_topic_data) != len(weibo_using_topics):
            print(f"[WARN] Year {year} has less than {len(weibo_using_topics)} topics")
            continue
        year_df = None
        for topic, topic_df in year_topic_data.items():
            if year_df is None:
                year_df = topic_df
            else:
                year_df = year_df.merge(
                    topic_df, left_index=True, right_index=True, how="outer"
                )

        year_df["year"] = year
        all_year_dfs.append(year_df)
        print(f"[INFO] Processed year {year}: {len(year_df)} authors")

    merged_df = pd.concat(all_year_dfs, ignore_index=False)

    numeric_cols = (
        merged_df[weibo_using_topics]
        .select_dtypes(include=[np.number])
        .columns.tolist()
    )
    for year, df_year in merged_df.groupby("year"):
        X = df_year[numeric_cols].to_numpy()
        r, c = np.where(np.isfinite(X))
        data = X[r, c].astype(float)
        csr = sparse.csr_matrix((data, (r, c)), shape=X.shape)

        output_dir = csr_folder("weibo")
        os.makedirs(output_dir, exist_ok=True)
        output_file = output_dir / f"{year}.csr.npz"
        sparse.save_npz(output_file, csr)
        print(
            f"saved {output_file} with shape={csr.shape} nnz={csr.nnz} density={csr.nnz / (csr.shape[0] * csr.shape[1]):.4f}"
        )


def transform_data(surveys=True, social=False) -> None:
    """Write data/dimension/csr/<stem>/<year>.csr.npz.

    Surveys are cheap and read data/survey/*.dta. Social media (``--social``)
    needs the cleaned opinion folders on the cluster (config); its csr files
    are normally taken as precomputed inputs.
    """
    if surveys:
        for survey_data_name in SURVEY_STEMS:
            restricted_topics = (
                us_survey_using_topics
                if "anes" in survey_data_name
                else (
                    cn_survey_using_topics
                    if "wvs" in survey_data_name
                    else eu_survey_using_topics
                )
            )
            input_file = original_data_map[survey_data_name]
            if not os.path.exists(input_file):
                print(f"[SKIP] {survey_data_name}: {input_file} not found")
                continue
            output_dir = csr_folder(survey_data_name)
            os.makedirs(output_dir, exist_ok=True)
            convert_and_save_survey_data(input_file, output_dir, restricted_topics)

    if social:
        for mode in TWITTER_MODES:
            convert_and_save_twitter_data(mode=mode)
        convert_and_save_weibo_data()
    print("All data transformed and saved.")


def calculate(
    dataset_type: str,
    mode: str = "us",
    distance_space: str = "none",
    plot: bool = False,
    matrix: str = "cov",
) -> Dict[str, Any]:
    """
    Run dimensions.estimate() on survey data files.

    Args:
        dataset_type: Which dataset type to analyze (folder_map的key)
        distance_space: Distance space for analysis (spectral, raw, bagged, none). Default is "none".
        plot: If True, also save per-year PDF visualizations (loadings heatmap + scree).
        matrix: "cov" (pairwise-available covariance, the main analysis) or
            "corr" (the same matrix rescaled to correlations). Results go to
            data/dimension/results/<matrix>/ with the same file names, so the
            two never overwrite each other.

    Returns:
        Dictionary with keys: PR, eRank, srank, each containing a list of results.
    """
    from analysis.dimension.dimensions import estimate

    work_folder = dimension_results_dir(matrix)
    work_folder.mkdir(parents=True, exist_ok=True)
    standardize = matrix == "corr"
    tag = distance_space or "none"
    loadings_tag = "loadings"

    npz_folder = csr_folder(dataset_type, mode)
    if not npz_folder.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {npz_folder}")

    # 转换 distance_space 参数
    distance_space_value = None if distance_space == "none" else distance_space

    # 查找所有 npz 文件
    npz_files = sorted(npz_folder.glob("*.npz"))
    if not npz_files:
        print(f"[WARN] No .npz files found in: {npz_folder}")
        return {"PR": [], "eRank": [], "srank": []}

    print(f"[INFO] Dataset type: {dataset_type}")
    if dataset_type == "twitter":
        print(f"[INFO] Mode: {mode}")
    print(f"[INFO] Distance space: {distance_space_value}; matrix: {matrix}")
    print(f"[INFO] Found {len(npz_files)} files to process")

    # 初始化结果字典
    all_results = {
        "PR": [],
        "eRank": [],
        "srank": [],
    }

    # 收集每年的 loading 报告，便于在跑完后生成汇总文本
    topic_names = _topics_for(dataset_type, mode)
    year_reports: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []

    # 简单的 logging 设置
    log_file = work_folder / f"{dataset_type}-{tag}.log"
    if dataset_type == "twitter":
        log_file = work_folder / f"{dataset_type}-{mode}-{tag}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file, encoding="utf-8", mode="w")],
        force=True,
    )
    logger = logging.getLogger()

    # 遍历所有文件并计算
    for i, input_file in enumerate(npz_files, 1):
        try:
            logger.info(f"[{i}/{len(npz_files)}] Processing: {input_file.name}")
            print(f"[{i}/{len(npz_files)}] Processing: {input_file.name}")

            # 调用 estimate
            estimated = estimate(input_file, distance_space=distance_space_value, standardize=standardize)

            # 保存单个文件的 json 结果
            json_file = (
                work_folder
                / f"{dataset_type}-{input_file.stem}-{tag}.json"
            )
            if dataset_type == "twitter":
                json_file = (
                    work_folder
                    / f"{dataset_type}-{mode}-{input_file.stem}-{tag}.json"
                )
            with open(json_file, "w") as f:
                json.dump(estimated, f, indent=2)

            # 收集结果
            all_results["PR"].append(estimated.get("PR"))
            all_results["eRank"].append(estimated.get("eRank"))
            all_results["srank"].append(estimated.get("srank"))

            # NEW: compute and save per-year PCA loading report
            try:
                report = _loading_report(input_file, topic_names, standardize=standardize)
                loadings_file = (
                    work_folder / f"{dataset_type}-{input_file.stem}-{loadings_tag}.json"
                )
                if dataset_type == "twitter":
                    loadings_file = (
                        work_folder
                        / f"{dataset_type}-{mode}-{input_file.stem}-{loadings_tag}.json"
                    )
                with open(loadings_file, "w") as f:
                    json.dump(report, f, indent=2)
                year_reports.append((input_file.stem, report, estimated))
                logger.info(
                    f"[{i}/{len(npz_files)}] Loadings saved: {loadings_file.name}"
                )

                if plot:
                    title_prefix = dataset_type + (
                        f"/{mode}" if dataset_type == "twitter" else ""
                    )
                    plot_stem_prefix = (
                        f"{dataset_type}-{mode}-{input_file.stem}"
                        if dataset_type == "twitter"
                        else f"{dataset_type}-{input_file.stem}"
                    )
                    plot_folder = report_dir("dimension_loadings")
                    heatmap_pdf = (
                        plot_folder / f"{plot_stem_prefix}-loadings-heatmap.pdf"
                    )
                    scree_pdf = plot_folder / f"{plot_stem_prefix}-scree.pdf"
                    _plot_loadings_heatmap(
                        report, heatmap_pdf, f"{title_prefix} — {input_file.stem}"
                    )
                    _plot_variance_scree(
                        report,
                        estimated,
                        scree_pdf,
                        f"{title_prefix} — {input_file.stem}",
                    )
                    logger.info(
                        f"[{i}/{len(npz_files)}] Plots saved: {heatmap_pdf.name}, {scree_pdf.name}"
                    )
            except Exception as e:
                logger.error(f"Loading report failed for {input_file.name}: {e!r}")
                print(f"[ERROR] Loading report failed for {input_file.name}: {e!r}")

            logger.info(f"[{i}/{len(npz_files)}] Completed: {input_file.name}")
            print(f"[{i}/{len(npz_files)}] Completed: {input_file.name}")

        except Exception as e:
            logger.error(f"Error processing {input_file.name}: {e!r}")
            logger.error(traceback.format_exc())
            print(f"[ERROR] Failed to process {input_file.name}: {e!r}")
            # 错误时添加 None 以保持列表长度一致
            all_results["PR"].append(None)
            all_results["eRank"].append(None)
            all_results["srank"].append(None)

    # 保存汇总结果
    summary_file = (
        work_folder / f"{dataset_type}-{tag}-summary.json"
    )
    if dataset_type == "twitter":
        summary_file = (
            work_folder
            / f"{dataset_type}-{mode}-{tag}-summary.json"
        )
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"[INFO] Summary saved to: {summary_file}")
    logger.info(f"All tasks completed. Summary saved to: {summary_file}")

    # NEW: write a human-readable loading summary aggregating all years
    if year_reports:
        # Sort by year (stem); fall back to original order if non-numeric.
        def _stem_key(item):
            try:
                return (0, int(item[0]))
            except (TypeError, ValueError):
                return (1, item[0])

        year_reports_sorted = sorted(year_reports, key=_stem_key)
        loading_summary_text = _format_loading_summary(
            dataset_type, mode, tag, year_reports_sorted
        )
        loading_summary_file = (
            work_folder
            / f"{dataset_type}-{tag}-loadings-summary.txt"
        )
        if dataset_type == "twitter":
            loading_summary_file = (
                work_folder
                / f"{dataset_type}-{mode}-{tag}-loadings-summary.txt"
            )
        with open(loading_summary_file, "w") as f:
            f.write(loading_summary_text)
        print(f"[INFO] Loading summary saved to: {loading_summary_file}")
        logger.info(f"Loading summary saved to: {loading_summary_file}")

    return all_results


# Every source behind Fig 3 (main + robustness): (dataset_type, mode).
FIG3_SOURCES = [(stem, "us") for stem in SURVEY_STEMS] + [
    ("twitter", mode) for mode in TWITTER_MODES
] + [("weibo", "us")]


def calculate_all(surveys_only: bool = False, plot: bool = False, matrix: str = "cov") -> None:
    """Run calculate() for every Fig 3 source whose csr folder exists.

    ``--matrix corr`` runs the correlation-matrix variant (robustness check;
    writes to data/dimension/results/corr/ instead of results/cov/).
    """
    for dataset_type, mode in FIG3_SOURCES:
        if surveys_only and dataset_type in ("twitter", "weibo"):
            continue
        if not csr_folder(dataset_type, mode).exists():
            print(f"[SKIP] no csr folder for {dataset_type}/{mode}")
            continue
        calculate(dataset_type, mode=mode, plot=plot, matrix=matrix)


if __name__ == "__main__":
    fire.Fire(
        {
            "transform": transform_data,
            "calculate": calculate,
            "calculate_all": calculate_all,
        }
    )
