"""
Pairwise topic correlations for the surveys (ANES / EVS / WVS and variants).

    python -m analysis.correlation.survey_correlation all
    -> data/correlation/network_analysis_<survey>.csv

处理调查数据，计算话题之间的相关性

输入文件：.dta文件（Stata格式）
- 必须包含year列（年份）
- 其他列都是话题opinion，取值为1-5
- 每一行是一个个体（会自动生成独立的id）

输出文件：CSV文件
表头：year,Topic1,Topic2,tid1,tid2,Intersection,Correlation (Pearson),P-Value (Pearson)

Topic1和Topic2是两个话题，需要穷尽所有可能的组合
Intersection：对应year在这两个话题上都发表了意见的个体数目
Correlation (Pearson)：这两个话题的correlation
P-Value (Pearson)：这两个话题的correlation的p-value

year包括了原始数据中的所有year，以及一个单独的average，代表所有年份（不区分year）
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import itertools
import os
import fire

from common.paths import CORRELATION_DIR, SURVEY_DIR

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

restricted_topics = {
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
}


def process_survey(input_path, output_path):
    """
    处理调查数据，计算话题之间的相关性

    参数:
        input_path: 输入的.dta文件路径
        output_path: 输出的CSV文件路径
    """
    # 读取Stata文件
    input_path = str(input_path)
    name = os.path.basename(input_path).split(".")[0]
    df = pd.read_stata(input_path)
    print(df.head())
    print(df.columns)
    # 生成唯一id
    if "evs" in name:
        print(len(df))
        if "eu_nen" in name:
            df = df[df["country"] != "Great Britain"]
            print(len(df))
        elif name.startswith("en_"):
            df = df[df["country"] == "Great Britain"]
            print(len(df))
        else:
            test_df = df[df["country"] == "Great Britain"]
            print(len(test_df))
    df = df.reset_index().rename(columns={"index": "id"})
    if "year" not in df.columns:
        df["year"] = 2020
    # 话题列（去掉id和year）
    topic_cols = restricted_topics[name]
    # topic_cols = [col for col in df.columns if col not in ["id", "year"]]
    # 所有年份
    years = sorted(df["year"].dropna().unique().tolist())
    years.append("average")
    # years = ["average"]
    results = []
    # 穷举所有话题组合
    for topic1, topic2 in itertools.combinations(topic_cols, 2):
        for year in years:
            if year == "average":
                subdf = df
            else:
                subdf = df[df["year"] == year]
            # 只保留两个话题都不缺失的行
            valid = subdf[[topic1, topic2]].dropna()
            intersection = len(valid)
            if intersection > 1:
                try:
                    corr, pval = pearsonr(valid[topic1], valid[topic2])
                except Exception:
                    corr, pval = np.nan, np.nan
            else:
                corr, pval = np.nan, np.nan
            results.append(
                {
                    "year": year,
                    "Topic1": topic1,
                    "Topic2": topic2,
                    "tid1": topic1,
                    "tid2": topic2,
                    "Intersection": intersection,
                    "Correlation (Pearson)": corr,
                    "P-Value (Pearson)": pval,
                }
            )
    outdf = pd.DataFrame(results)
    # 确保输出目录存在
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    outdf.to_csv(output_path, index=False)
    print(f"处理完成，结果已保存到: {output_path}")


def process_all():
    datas = [
        "anes",
        "anes_media",
        "wvs",
        "wvs_media",
        "evs",
        "evs_media",
        "evs_resample2",
        "evs_media_resample2",
        "en_evs",
        "en_evs_media",
        "eu_nen_evs",
    ]
    for data_name in datas:
        input_path = str(SURVEY_DIR / f"{data_name}.dta")
        output_path = str(CORRELATION_DIR / f"network_analysis_{data_name}.csv")
        process_survey(input_path, output_path)


if __name__ == "__main__":
    fire.Fire({"single": process_survey, "all": process_all})
