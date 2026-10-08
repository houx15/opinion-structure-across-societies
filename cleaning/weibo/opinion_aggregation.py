import fire
import glob
import time
import json

import os

import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from cleaning.settings import *


YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]


def convert_data_to_opinions(topic, year):
    topic = str(topic)
    data_path = os.path.join(WEIBO_TOPIC_KEYWORD_DIR, topic)
    parquet_files = glob.glob(os.path.join(data_path, f"{year}-*.parquet"))

    output_dir = os.path.join(WEIBO_OPINION_DIR, topic)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, f"opinion-{year}.parquet")

    all_data = pd.DataFrame()

    for parquet_file in parquet_files:
        print(f"Processing file: {parquet_file}")
        pfile = pq.ParquetFile(parquet_file)
        all_columns = pfile.schema.names
        if "opinion" not in all_columns:
            print(f"File {parquet_file} does not have opinion column.")
            continue
        df = pd.read_parquet(parquet_file, engine="fastparquet", columns=["user_id", "opinion"])
        # drop na opinions
        df = df.dropna(subset=["opinion"])

        all_data = pd.concat([all_data, df], ignore_index=True)
    user_opinions_series = all_data.groupby("user_id")["opinion"].apply(list).reset_index()
    user_opinions_series.to_parquet(output_file, engine="fastparquet")


def merge_yearly_opinion_data(topic):
    """
    合并多年的用户 opinion 数据到一个文件。
    
    参数:
        topic (str): 主题名称，用于定位文件路径。
    
    返回:
        None: 将合并后的数据保存为一个 Parquet 文件。
    """
    topic = str(topic)
    # 定义输入和输出路径
    input_dir = os.path.join(WEIBO_OPINION_DIR, topic)
    output_file = os.path.join(input_dir, f"merged_opinion.parquet")

    # 初始化空列表存储年度数据
    all_years_data = []

    # 遍历每年的文件
    for year in YEARS:
        yearly_file = os.path.join(input_dir, f"opinion-{year}.parquet")
        
        # 检查文件是否存在
        if not os.path.exists(yearly_file):
            print(f"File {yearly_file} not found. Skipping...")
            continue
        
        # 读取年度数据
        yearly_data = pd.read_parquet(yearly_file, engine="fastparquet")
        
        yearly_data["opinion"] = yearly_data["opinion"].apply(lambda x: x if isinstance(x, list) else [x])
        
        # 添加到总数据列表
        all_years_data.append(yearly_data)

    # 合并所有年度数据
    if all_years_data:
        merged_data = pd.concat(all_years_data, ignore_index=True)

        # 按 user_id 分组并合并 opinion 列为列表
        result = merged_data.groupby("user_id")["opinion"].apply(lambda x: sum(x, [])).reset_index()

        # 保存合并后的数据到 Parquet 文件
        result.to_parquet(output_file, engine="fastparquet")
        print(f"Merged data saved to {output_file}")
    else:
        print("No data to merge.")


def log(message):
    with open("log.txt", "a") as f:
        f.write(message + "\n")


def calculate_weighted_average(user_opinion_df, years):
    """
    计算用户的加权平均 opinion。
    
    参数:
        user_opinion_df (pd.DataFrame): 包含用户每年的 opinion 和 count 列的 DataFrame。
        years (list): 年份列表，用于定位列名。
    
    返回:
        pd.Series: 每个用户的加权平均 opinion。
    """
    # 初始化加权总和和总计数
    weighted_sum = pd.Series(0, index=user_opinion_df.index)
    total_count = pd.Series(0, index=user_opinion_df.index)
    
    # 遍历每年的数据
    for year in years:
        opinion_col = year  # 当前年份的 opinion 列
        count_col = f"{year}_count"  # 当前年份的 count 列
        
        # 计算加权总和和总计数
        start_time = time.time()
        weighted_sum += user_opinion_df[opinion_col] * user_opinion_df[count_col]
        total_count += user_opinion_df[count_col]
        log(f"Weighted sum and count time for {year}: {time.time() - start_time:.2f} seconds")
    
    # 计算加权平均
    weighted_average = weighted_sum / total_count
    
    # 将加权平均结果添加到 DataFrame 的最后一列
    user_opinion_df["weighted_average"] = weighted_average
    
    return user_opinion_df


def calculate_user_opinion_dataframe(topic):
    """
    计算用户在每年的平均 opinion，并生成一个大的 DataFrame。
    
    参数:
        topic (str): 主题名称，用于定位文件路径。
    
    返回:
        pd.DataFrame: 每行是一个用户，每列是该用户在某年的平均 opinion，最后一列是整体平均 opinion。
    """
    # 定义输入路径
    topic = str(topic)
    input_dir = os.path.join(WEIBO_OPINION_DIR, topic)
    
    yearly_dfs = []

    # 遍历每年的文件
    for year in YEARS:
        yearly_file = os.path.join(input_dir, f"opinion-{year}.parquet")
        
        # 检查文件是否存在
        if not os.path.exists(yearly_file):
            print(f"File {yearly_file} not found. Skipping...")
            continue
        
        # 读取年度数据
        yearly_data = pd.read_parquet(yearly_file, engine="fastparquet")
        print(yearly_data.head())
        yearly_data['user_id'] = pd.to_numeric(yearly_data['user_id'], errors='coerce')
        # drop na user_id
        yearly_data = yearly_data.dropna(subset=["user_id"])
        print(yearly_data["user_id"])
        # raise Exception("Debugging")
        log(f"Yearly data for {year} loaded, shape: {yearly_data.shape}")

        start_time = time.time()
        yearly_data = yearly_data.set_index("user_id")
        log(f"Set index time: {time.time() - start_time:.2f} seconds")

        yearly_opinion = yearly_data["opinion"]
        
        # 计算每个用户的平均 opinion
        start_time = time.time()
        yearly_count_opinion = np.array([len(row) for row in yearly_opinion])
        yearly_sum_opinion = np.array([sum(row) for row in yearly_opinion])
        # print shape
        print(yearly_count_opinion.shape)
        print(yearly_sum_opinion.shape)
        log(f"Count time: {time.time() - start_time:.2f} seconds")
        

        yearly_avg_opinion = yearly_sum_opinion / yearly_count_opinion

        # 将结果合并到 DataFrame，列名为年份
        yearly_data[str(year)] = yearly_avg_opinion
        yearly_data[f"{year}_count"] = yearly_count_opinion
        yearly_data[f"{year}_sum"] = yearly_sum_opinion
        yearly_data = yearly_data.drop(columns=["opinion"])

        yearly_dfs.append(yearly_data)
    
    # 合并所有年度数据
    user_opinion_df = pd.concat(yearly_dfs, axis=1, join="outer")
    del yearly_dfs
    del yearly_data

    sum_cols = [f"{year}_sum" for year in YEARS]
    count_cols = [f"{year}_count" for year in YEARS]
    
    # 计算每个用户的总 opinion 和总 count
    user_opinion_df["sum"] = user_opinion_df[sum_cols].sum(axis=1)
    user_opinion_df["count"] = user_opinion_df[count_cols].sum(axis=1)

    user_opinion_df["average"] = user_opinion_df["sum"] / user_opinion_df["count"]
    user_opinion_df.dropna(subset=["average"], inplace=True)
    user_opinion_df.drop(columns=sum_cols, inplace=True)
    user_opinion_df = user_opinion_df.drop(columns=["sum", "count"])
    # 将结果保存到 Parquet 文件
    output_file = os.path.join(input_dir, f"avg_opinion.parquet")
    user_opinion_df.to_parquet(output_file, engine="fastparquet")
    print(f"User opinion DataFrame saved to {output_file}")


def run_all_year(topic):
    for year in YEARS:
        convert_data_to_opinions(topic, year)

def convert_all_topics():
    for topic in range(17):
        if topic in [3, 4, 8]:
            continue
        topic = str(topic)
        run_all_year(topic)


def merge_all_topic():
    for topic in range(17):
        if topic in [3, 4, 8]:
            continue
        topic = str(topic)
        calculate_user_opinion_dataframe(topic)


def clean_unexpected_users(topic,):
    topic = str(topic)
    data_path = os.path.join(WEIBO_TOPIC_KEYWORD_DIR, topic)
    output_dir = os.path.join(WEIBO_OPINION_DIR, "debug")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, f"opinion-{topic}.parquet")
    all_data = pd.DataFrame()
    for year in YEARS:
        parquet_files = glob.glob(os.path.join(data_path, f"{year}-*.parquet"))
        for parquet_file in parquet_files:
            data = pd.read_parquet(parquet_file, engine="fastparquet")
            data['convert_user_id'] = pd.to_numeric(data['user_id'], errors='coerce')
            # 筛选出convert user id 为na的
            na_user = data[data['convert_user_id'].isna()]
            na_user_df = na_user[['user_id', 'weibo_id']].copy()
            # if empty
            if na_user_df.empty:
                continue
            na_user_df['date'] = parquet_file.split("/")[-1].split(".")[0]
            all_data = pd.concat([all_data, na_user_df], ignore_index=True)
    all_data.to_parquet(output_file, engine="fastparquet")


def merge_unexpected_users():
    """
    合并所有存在问题的users
    输出：tweet_id, user_id, date
    额外输出一个列表，包含所有的dates
    """
    data_path = os.path.join(WEIBO_OPINION_DIR, "debug")
    output_file = os.path.join(data_path, "all_unexpected_users.parquet")
    all_bug_dates = set()
    all_data = pd.DataFrame()

    all_bug_files = glob.glob(os.path.join(data_path, "opinion-*.parquet"))
    for bug_file in all_bug_files:
        topic_id = bug_file.split("/")[-1].split("-")[1].split(".")[0]
        data = pd.read_parquet(bug_file, engine="fastparquet")
        data["topic_id"] = topic_id
        all_data = pd.concat([all_data, data], ignore_index=True)

        # 获取日期
        unique_dates = data['date'].unique()
        all_bug_dates.update(unique_dates)

    all_data.to_parquet(output_file, engine="fastparquet")
    print(f"All unexpected users saved to {output_file}")
    all_bug_dates = list(all_bug_dates)
    with open(os.path.join(data_path, "all_bug_dates.json"), "w") as f:
        json.dump(all_bug_dates, f)


def recover_unexpected_users():
    original_data_folder = WEIBO_TOPIC_KEYWORD_DIR
    data_path = os.path.join(WEIBO_OPINION_DIR, "debug")
    recover_file = os.path.join(data_path, "result.parquet")
    all_users = os.path.join(data_path, "all_unexpected_users.parquet")

    all_data = pd.read_parquet(all_users, engine="fastparquet")
    all_data.set_index("weibo_id", inplace=True)
    recover_data = pd.read_parquet(recover_file, engine="fastparquet")
    recover_data.set_index("weibo_id", inplace=True)

    # merge, keep all_data
    all_data = all_data.merge(recover_data, how="left", left_index=True, right_index=True)
    all_data.dropna(subset=["new_user_id"], inplace=True)

    # 按照topic_id, date分组，从而将新的文件对应为源文件
    for (topic, date), group_df in all_data.groupby(['topic_id', 'date']):
        # 获取源文件路径
        original_file = os.path.join(original_data_folder, topic, f"{date}.parquet")
        if not os.path.exists(original_file):
            print(f"Original file {original_file} does not exist.")
            continue
        
        # 读取源文件
        original_data = pd.read_parquet(original_file, engine="fastparquet")

        # groupd_df中对应的new_user_id，替换同weibo_id的user_id
        for _, row in group_df.iterrows():
            weibo_id = row.name
            new_user_id = row["new_user_id"]
            original_data.loc[original_data["weibo_id"] == weibo_id, "user_id"] = new_user_id
        
        # 保存修改后的源文件

        original_data.to_parquet(original_file, engine="fastparquet")


if __name__ == "__main__":
    fire.Fire({
        "clean_all": clean_unexpected_users,
        "merge_bug": merge_unexpected_users,
        "recover_user": recover_unexpected_users,

        "convert_yearly": convert_data_to_opinions,
        "convert": run_all_year,
        "convert_all": convert_all_topics,
        
        "merge": calculate_user_opinion_dataframe,
        "merge_all": merge_all_topic,
    })
