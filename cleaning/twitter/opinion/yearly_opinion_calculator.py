import fire
import glob
import time
import json

import os

import pandas as pd
import numpy as np

from cleaning.settings import *

YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
original_dir = TWITTER_DIR
output_dir = OPINION_DIR
os.makedirs(output_dir, exist_ok=True)


def convert_data_to_opinions(topic, year):
    folder = os.path.join(original_dir, f"{folder_map[topic]}-opinion")
    files = glob.glob(os.path.join(folder, f"{year}-*-text.pickle4"))
    # print(files)
    if len(files) == 0:
        print(f"No files found for {topic} in {year}")
        return

    output_file = os.path.join(output_dir, f"opinion-{topic}-{year}.parquet")

    all_data = pd.DataFrame()

    for file in files:
        data = pd.read_pickle(file)
        data = data[data["relevance"] == 1]
        data = data[["author_id", "opinion"]].copy()
        all_data = pd.concat([all_data, data], ignore_index=True)

    # 将all_data的author_id设置为int
    all_data["author_id"] = all_data["author_id"].astype(int)

    user_opinions_series = (
        all_data.groupby("author_id")["opinion"].apply(list).reset_index()
    )
    # 检查是否有重复的author_id
    if user_opinions_series["author_id"].duplicated().any():
        # 打印重复索引的用户ID
        duplicate_indices = user_opinions_series["author_id"][
            user_opinions_series["author_id"].duplicated()
        ]
        print(f"Duplicate indices found in {file}: {duplicate_indices.tolist()}")
        raise ValueError(f"Duplicate index found in {file}. Skipping...")
        return
    user_opinions_series.to_parquet(output_file, engine="fastparquet")


def calculate_user_opinion_dataframe(topic):
    """
    计算用户在每年的平均 opinion，并生成一个大的 DataFrame。

    参数:
        topic (str): 主题名称，用于定位文件路径。

    返回:
        pd.DataFrame: 每行是一个用户，每列是该用户在某年的平均 opinion，最后一列是整体平均 opinion。
    """
    output_file = os.path.join(output_dir, f"merged-{topic}.parquet")
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping...")
        return

    yearly_dfs = []
    valid_years = []

    # 遍历每年的文件
    for year in YEARS:
        yearly_file = os.path.join(output_dir, f"opinion-{topic}-{year}.parquet")

        # 检查文件是否存在
        if not os.path.exists(yearly_file):
            print(f"File {yearly_file} not found. Skipping...")
            continue

        # 读取年度数据
        yearly_data = pd.read_parquet(yearly_file, engine="fastparquet")
        yearly_data = yearly_data.set_index("author_id")

        yearly_opinion = yearly_data["opinion"]

        yearly_count_opinion = np.array([len(row) for row in yearly_opinion])
        yearly_sum_opinion = np.array([sum(row) for row in yearly_opinion])

        yearly_avg_opinion = yearly_sum_opinion / yearly_count_opinion

        # 将结果合并到 DataFrame，列名为年份
        yearly_data[str(year)] = yearly_avg_opinion
        yearly_data[f"{year}_count"] = yearly_count_opinion
        yearly_data[f"{year}_sum"] = yearly_sum_opinion
        yearly_data = yearly_data.drop(columns=["opinion"])

        # 检查yearly_data是否有重复索引
        if yearly_data.index.duplicated().any():
            # 打印重复索引的用户ID
            duplicate_indices = yearly_data.index[yearly_data.index.duplicated()]
            print(
                f"Duplicate indices found in {yearly_file}: {duplicate_indices.tolist()}"
            )
            raise ValueError(f"Duplicate index found in {yearly_file}. Skipping...")
            continue

        yearly_dfs.append(yearly_data)
        valid_years.append(year)

    # 合并所有年度数据
    user_opinion_df = pd.concat(yearly_dfs, axis=1, join="outer")
    del yearly_dfs
    del yearly_data

    sum_cols = [f"{year}_sum" for year in valid_years]
    count_cols = [f"{year}_count" for year in valid_years]

    # 计算每个用户的总 opinion 和总 count
    user_opinion_df["sum"] = user_opinion_df[sum_cols].sum(axis=1)
    user_opinion_df["count"] = user_opinion_df[count_cols].sum(axis=1)

    user_opinion_df["average"] = user_opinion_df["sum"] / user_opinion_df["count"]
    user_opinion_df.dropna(subset=["average"], inplace=True)
    user_opinion_df.drop(columns=sum_cols, inplace=True)
    user_opinion_df = user_opinion_df.drop(columns=["sum", "count"])
    # 将结果保存到 Parquet 文件
    output_file = os.path.join(output_dir, f"merged-{topic}.parquet")
    user_opinion_df.to_parquet(output_file, engine="fastparquet")
    print(f"User opinion DataFrame saved to {output_file}")


def run_all_year(topic):
    for year in YEARS:
        convert_data_to_opinions(topic, year)


def convert_all_topics():
    for topic in folder_map.keys():
        if topic in ["abo", "clc", "gun", "sxo", "chn", "dru"]:
            continue
        run_all_year(topic)


def merge_all_topics():
    for topic in folder_map.keys():
        if topic in ["abo", "clc", "gun", "sxo", "chn", "dru"]:
            continue
        calculate_user_opinion_dataframe(topic)


if __name__ == "__main__":
    fire.Fire(
        {
            "convert": convert_all_topics,
            "convert_topic": run_all_year,
            "merge": merge_all_topics,
        }
    )
