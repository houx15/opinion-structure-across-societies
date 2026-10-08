import os

topics = ["5", "6", "9", "10", "12", "13", "14", "15", "16"]
YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

import pandas as pd
from cleaning.settings import *


opinion_dir = WEIBO_OPINION_DIR

def get_frequent_users(topic):
    """
    计算每个用户在每年的平均 opinion，并生成一个大的 DataFrame。
    
    参数:
        topic (str): 主题名称，用于定位文件路径。
    
    返回:
        pd.DataFrame: 每行是一个用户，每列是该用户在某年的平均 opinion，最后一列是整体平均 opinion。
    """
    # 定义输入路径
    topic = str(topic)
    input_dir = os.path.join(opinion_dir, topic)

    avg_df = pd.read_parquet(os.path.join(input_dir, "avg_opinion.parquet"))

    count_columns = [f"{year}_count" for year in YEARS]
    avg_df["count"] = avg_df[count_columns].sum(axis=1)

    # 排序，取前10000个用户的user_id，按行写入txt
    avg_df = avg_df.sort_values(by="count", ascending=False).head(10000)
    print(avg_df)
    user_ids = avg_df.index.tolist()
    with open(os.path.join(input_dir, "frequent_users.txt"), "w") as f:
        for user_id in user_ids:
            f.write(f"{user_id}\n")
    print(f"Frequent users for topic {topic} saved to {input_dir}/frequent_users.txt")


if __name__ == "__main__":
    for topic in topics:
        get_frequent_users(topic)