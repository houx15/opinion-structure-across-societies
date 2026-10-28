"""Pairwise topic correlations + shuffle baseline for Twitter users of one region.

Input (cleaning stage, on the cluster):
    <TWITTER_OPINION_DIR>/merged-<topic>.parquet   user x year opinions (index = author id,
                                                   columns "2016".."2023", "average")
    <LOCATION_DIR>/<mode>_user_ids.json            users located in the region
Output:
    data/correlation/network_analysis_<src>.csv    src = twitter / eutwitter / entwitter / eu_nentwitter
    columns: year, Topic1, Topic2, Intersection, Correlation (Pearson),
             P-Value (Pearson), shuffled_pearson, shuffled_pearson_std

For every year (and the pooled "average"), users with an opinion on both
topics are kept, opinions are clipped to [-2, 2], and Pearson's r is compared
with the mean / sd of r over ``shuffle_times`` independent shuffles of the two
opinion vectors.

    python -m analysis.correlation.twitter_correlation run --mode us
    python -m analysis.correlation.twitter_correlation all
"""

import json
import os
from itertools import combinations

import fire
import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr

from common.paths import CORRELATION_DIR
from config import cfg

MODES = {"us": "twitter", "eu": "eutwitter", "en": "entwitter", "eu_nen": "eu_nentwitter"}

US_TOPICS = ["abo", "gun", "clc", "sxo", "vac", "soc", "dpp", "minwage", "ubi"]
# Europe: sex-work legalization replaces gun control (EVS asks about prostitution).
EU_TOPICS = ["abo", "swe", "clc", "sxo", "vac", "soc", "dpp", "minwage", "ubi"]
TOPICS = {"us": US_TOPICS, "eu": EU_TOPICS, "en": EU_TOPICS, "eu_nen": EU_TOPICS}

TARGET_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, "average"]
SHUFFLE_TIMES = 10000


def load_keep_ids(mode):
    path = os.path.join(cfg.LOCATION_DIR, f"{mode}_user_ids.json")
    with open(path, "r") as f:
        return set(int(x) for x in json.load(f))


def load_topic_opinions(topics, source_folder=None):
    source_folder = source_folder or cfg.TWITTER_OPINION_DIR
    all_data = {}
    for topic in topics:
        file_name = os.path.join(source_folder, f"merged-{topic}.parquet")
        df = pd.read_parquet(file_name, engine="fastparquet")
        df.index = pd.to_numeric(df.index, errors="raise").astype("int64")
        all_data[topic] = df
    return all_data


def calculate_correlation(opinions1, opinions2):
    pearson_corr, pearson_p = pearsonr(opinions1, opinions2)
    return pearson_corr, pearson_p


def calculate_shuffle_correlation(opinions1, opinions2, exp_times=1000):
    opinions1 = np.array(opinions1)
    opinions2 = np.array(opinions2)
    shuffled_pearsons = []
    for _ in range(exp_times):
        np.random.shuffle(opinions1)
        np.random.shuffle(opinions2)
        pearson_corr, _ = calculate_correlation(opinions1, opinions2)
        shuffled_pearsons.append(pearson_corr)
    return np.mean(shuffled_pearsons), np.std(shuffled_pearsons)


def pearson_confidence_interval(r, n, alpha=0.05):
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / np.sqrt(n - 3)
    z_critical = norm.ppf(1 - alpha / 2)
    z_ci = (z - z_critical * se, z + z_critical * se)
    return np.tanh(z_ci[0]), np.tanh(z_ci[1])


def run(mode="us", shuffle_times=SHUFFLE_TIMES, source_folder=None, output_dir=None):
    """Correlations for one region; returns the output csv path."""
    if mode not in MODES:
        raise ValueError(f"Invalid mode: {mode}. Known: {list(MODES)}")
    keep_ids = load_keep_ids(mode)
    topic_list = TOPICS[mode]
    all_data = load_topic_opinions(topic_list, source_folder)

    final_result = []
    for year in TARGET_YEARS:
        year_all_topic = {}
        cur_year_topics = []
        for topic in topic_list:
            if str(year) not in all_data[topic].columns:
                continue
            year_topic = all_data[topic][str(year)].dropna()
            year_all_topic[topic] = np.clip(year_topic, -2, 2)
            cur_year_topics.append(topic)

        for topic1, topic2 in combinations(cur_year_topics, 2):
            s1 = year_all_topic[topic1]
            s2 = year_all_topic[topic2]
            intersection_set = set(s1.index) & set(s2.index) & keep_ids
            s1_filtered = s1.loc[list(intersection_set)]
            s2_filtered = s2.loc[list(intersection_set)]

            pearson_corr = pearson_p = shuffled_pearson = shuffled_pearson_std = None
            if len(intersection_set) > 1:  # correlation needs at least 2 users
                pearson_corr, pearson_p = calculate_correlation(s1_filtered, s2_filtered)
                shuffled_pearson, shuffled_pearson_std = calculate_shuffle_correlation(
                    s1_filtered, s2_filtered, exp_times=shuffle_times
                )
            print(f"[{mode}] {year} {topic1}-{topic2}: n={len(intersection_set)} r={pearson_corr}")
            final_result.append((
                year, topic1, topic2, len(intersection_set), pearson_corr, pearson_p,
                shuffled_pearson, shuffled_pearson_std,
            ))

    df = pd.DataFrame(final_result, columns=[
        "year", "Topic1", "Topic2", "Intersection", "Correlation (Pearson)",
        "P-Value (Pearson)", "shuffled_pearson", "shuffled_pearson_std",
    ])
    output_dir = output_dir or CORRELATION_DIR
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"network_analysis_{MODES[mode]}.csv")
    df.to_csv(path, index=False)
    print(f"saved {path}")
    return path


def run_all(shuffle_times=SHUFFLE_TIMES):
    return [run(mode, shuffle_times=shuffle_times) for mode in MODES]


if __name__ == "__main__":
    fire.Fire({"run": run, "all": run_all})
