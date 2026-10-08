"""Pairwise topic correlations + shuffle baseline for Weibo users.

Input (cleaning stage, on the cluster):
    <WEIBO_OPINION_DIR>/<topic_id>/avg_opinion.parquet   user x year opinions
                                                         (columns "2016".."2023", "average")
    data/reference/weibo_topics.csv                      topic_id -> topic name
Output:
    data/correlation/network_analysis_weibo.csv
    columns: year, tid1, tid2, Topic1, Topic2, Intersection, Correlation (Pearson),
             P-Value (Pearson), shuffled_pearson, shuffled_pearson_std

Same procedure as twitter_correlation (opinions clipped to [-2, 2], users
with opinions on both topics, ``shuffle_times`` shuffles; 1000 for Weibo),
over all 15 Weibo topics; the paper uses 9 of them.

    python -m analysis.correlation.weibo_correlation
"""

import os
import warnings
from itertools import combinations

import fire
import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr

from analysis.correlation.twitter_correlation import (
    calculate_correlation,
    calculate_shuffle_correlation,
)
from common.paths import CORRELATION_DIR, REFERENCE_DIR
from config import cfg

warnings.filterwarnings("ignore", category=RuntimeWarning)

TOPIC_LIST = ["0", "1", "2", "4", "5", "6", "7", "9", "10", "11", "12", "13", "14", "15", "16"]
TARGET_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, "average"]
SHUFFLE_TIMES = 1000


def load_topic_names():
    opinion_df = pd.read_csv(REFERENCE_DIR / "weibo_topics.csv")
    return dict(zip(opinion_df["topic_id"].astype(str), opinion_df["topic"]))


def pearson_confidence_interval(r, n, alpha=0.001):
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / np.sqrt(n - 3)
    z_critical = norm.ppf(1 - alpha / 2)
    z_ci = (z - z_critical * se, z + z_critical * se)
    return np.tanh(z_ci[0]), np.tanh(z_ci[1])


def run(shuffle_times=SHUFFLE_TIMES, source_folder=None, output_dir=None):
    source_folder = source_folder or cfg.WEIBO_OPINION_DIR
    topic_dict = load_topic_names()
    all_data = {}
    for topic in TOPIC_LIST:
        file_name = os.path.join(source_folder, topic, "avg_opinion.parquet")
        if os.path.exists(file_name):
            all_data[topic] = pd.read_parquet(file_name, engine="fastparquet")
    topic_list = [t for t in TOPIC_LIST if t in all_data]

    final_result = []
    for year in TARGET_YEARS:
        year_all_topic = {
            topic: np.clip(all_data[topic][str(year)].dropna(), -2, 2) for topic in topic_list
        }
        for topic1, topic2 in combinations(topic_list, 2):
            s1 = year_all_topic[topic1]
            s2 = year_all_topic[topic2]
            intersection_set = list(set(s1.index) & set(s2.index))
            s1_filtered = s1.loc[intersection_set]
            s2_filtered = s2.loc[intersection_set]

            pearson_corr = pearson_p = shuffled_pearson = shuffled_pearson_std = None
            if len(intersection_set) > 1:  # correlation needs at least 2 users
                pearson_corr, pearson_p = calculate_correlation(s1_filtered, s2_filtered)
                shuffled_pearson, shuffled_pearson_std = calculate_shuffle_correlation(
                    s1_filtered, s2_filtered, exp_times=shuffle_times
                )
            final_result.append((
                year, topic1, topic2, topic_dict[topic1], topic_dict[topic2],
                len(intersection_set), pearson_corr, pearson_p,
                shuffled_pearson, shuffled_pearson_std,
            ))

    df = pd.DataFrame(final_result, columns=[
        "year", "tid1", "tid2", "Topic1", "Topic2", "Intersection",
        "Correlation (Pearson)", "P-Value (Pearson)",
        "shuffled_pearson", "shuffled_pearson_std",
    ])
    output_dir = output_dir or CORRELATION_DIR
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "network_analysis_weibo.csv")
    df.to_csv(path, index=False)
    print(f"saved {path}")
    return path


if __name__ == "__main__":
    fire.Fire(run)
