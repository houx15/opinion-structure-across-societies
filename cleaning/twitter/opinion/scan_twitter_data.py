import os
"""
筛选 Twitter 数据，根据 mode（us/eu）筛选出指定用户 ID 的数据
"""

import json
from pathlib import Path
import pandas as pd
import glob
import fire
from cleaning.settings import *

original_dir = OPINION_DIR


def filter_twitter_data(mode="en", output_dir=None):
    """
    筛选 Twitter 数据，只保留指定用户 ID 的数据

    Args:
        mode: "us" 或 "eu"，指定筛选模式
        output_dir: 输出目录，如果为 None 则使用默认路径
    """
    # 加载用户 ID 列表
    if mode == "us":
        user_ids_file = os.path.join(BASE_DIR, "us_user_ids.json")
    elif mode == "eu":
        user_ids_file = os.path.join(BASE_DIR, "eu_user_ids.json")
    elif mode == "en":
        user_ids_file = os.path.join(BASE_DIR, "en_user_ids.json")
    elif mode == "eu_nen":
        user_ids_file = os.path.join(BASE_DIR, "eu_nen_user_ids.json")
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'us' or 'eu' or 'en'")

    print(f"[INFO] Loading user IDs from {user_ids_file}")
    with open(user_ids_file, "r") as f:
        keep_ids = json.load(f)
        # 转为 int 并转为 set 以提高查找效率
        keep_ids = set(int(x) for x in keep_ids)
    print(f"[INFO] Loaded {len(keep_ids)} user IDs")

    # 设置输入和输出目录
    twitter_data_dir = Path(original_dir)
    if output_dir is None:
        output_dir = twitter_data_dir / mode
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")

    merged_files = glob.glob(f"{twitter_data_dir}/merged-*.parquet")

    # 处理每个 topic
    for file_path in merged_files:
        df = pd.read_parquet(file_path)

        # 获取原始行数
        original_count = len(df)
        print(f"[INFO] Original rows: {original_count}")

        # 筛选：只保留 index（author_id）在 keep_ids 中的行
        # 确保 index 是 int 类型以便匹配
        df.index = df.index.astype(int)
        filtered_df = df[df.index.isin(keep_ids)]

        filtered_count = len(filtered_df)
        print(
            f"[INFO] Filtered rows: {filtered_count} ({filtered_count/original_count*100:.2f}%)"
        )

        # 保存筛选后的数据
        output_path = output_dir / file_path.split("/")[-1]
        filtered_df.to_parquet(output_path)
        print(f"[INFO] Saved to: {output_path}")
        print()


if __name__ == "__main__":
    fire.Fire(filter_twitter_data)
