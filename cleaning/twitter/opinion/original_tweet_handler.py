"""
input: <ORIGINAL_TWEETS_DIR>/xx.tar.gz
unzip the tar.gz file and access all the pickle4 files
the first xxxx-xx-xx is the date
>>> data = pd.read_pickle("<ORIGINAL_TWEETS_DIR>/vaccine-mandate/2022-08-21_2022-08-21_vaccinemandate_keywords_alpha_beta_9.pickle4")
>>> data.head()
       conversation_id                created_at  ... public_metrics.quote_count  in_reply_to_user_id
0  1561491029425102850  2022-08-21T23:10:47.000Z  ...                          0                  NaN
1  1561491029395652608  2022-08-21T23:10:47.000Z  ...                          0                  NaN
2  1561491021447548930  2022-08-21T23:10:45.000Z  ...                          0                  NaN
3  1561491021183295490  2022-08-21T23:10:45.000Z  ...                          0                  NaN
4  1561371406511874053  2022-08-21T23:10:45.000Z  ...                          0  1059211792579248128

[5 rows x 14 columns]
>>> data.columns
Index(['conversation_id', 'created_at', 'source', 'edit_history_tweet_ids',
       'referenced_tweets', 'text', 'author_id', 'id', 'lang',
       'public_metrics.retweet_count', 'public_metrics.reply_count',
       'public_metrics.like_count', 'public_metrics.quote_count',
       'in_reply_to_user_id'],
      dtype='object')
>>>




output: <TWITTER_DIR>/<topic folder>
text_file: date-text.pickle4
         author_id                                              tweet
tweet_id
4607533     716903  Watching a discussion about abortion.... i'll ...
4522513     621543  Official campaigning is set to start in Portug...
"""

import os
import glob
import pandas as pd
import fire
import gzip
import tarfile

from cleaning.settings import *


handling_targets = ORIGINAL_TOPIC_LIST


def save_dataframes(output_folder, date, dataframes):
    """
    将数据帧保存为pickle文件。

    参数:
        output_folder (str): 输出目录。
        date (str): 日期字符串。
        dataframes (list): 数据帧列表。
    """
    # concat dataframes
    all_data = pd.concat(dataframes)
    # drop repeated index
    all_data = all_data[~all_data.index.duplicated(keep="first")]
    # save to pickle
    output_path = os.path.join(output_folder, f"{date}-text.pickle4")
    all_data.to_pickle(output_path)


def clean_original_tweets(topic):
    """
    清理原始推文数据，提取作者ID和推文内容，并保存为新的pickle文件。

    参数:
        topic (str): 主题名称，用于定位文件路径。
    """
    source_targz_file = os.path.join(ORIGINAL_TWEETS_DIR, f"{topic}.tar.gz")
    output_folder = os.path.join(TWITTER_DIR, f"{topic}-opinion")
    os.makedirs(output_folder, exist_ok=True)
    if not os.path.exists(source_targz_file):
        print(f"File {source_targz_file} not found. Skipping...")
        return
    cur_date = None
    cur_dataframes = []
    with tarfile.open(source_targz_file, "r:gz") as tar:
        for member in sorted(tar.getmembers(), key=lambda x: x.name):
            # 保证同一日期的挨着
            if member.name.endswith(".pickle4"):
                # pickle load 文件
                try:
                    file = tar.extractfile(member)
                    if file is not None:
                        # 读取数据
                        date = member.name.split("/")[-1].split("_")[0]
                        if cur_date is None:
                            cur_date = date
                        elif cur_date != date:
                            save_dataframes(output_folder, cur_date, cur_dataframes)
                            cur_dataframes = []
                            cur_date = date
                        data = pd.read_pickle(file)
                        data = data[["author_id", "id", "text"]].copy()
                        data.rename(columns={"id": "tweet_id"}, inplace=True)
                        data.set_index("tweet_id", inplace=True)
                        cur_dataframes.append(data)
                except Exception as e:
                    print(f"Error processing {member.name}: {e}")
                    continue


def clean_all():
    """
    处理所有主题的原始推文数据。
    """
    for topic in handling_targets:
        print(f"Processing {topic}...")
        clean_original_tweets(topic)
        print(f"Finished processing {topic}.")
    print("All topics processed.")


def line_counter():
    """
    统计每个文件的行数。
    """
    total_count = 0
    for topic in handling_targets:
        topic_count = 0
        folder = os.path.join(TWITTER_DIR, f"{topic}-opinion")
        files = glob.glob(os.path.join(folder, "*.pickle4"))
        for file in files:
            data = pd.read_pickle(file)
            topic_count += len(data)
            total_count += len(data)

        print(f"{topic}: {topic_count} lines")

    print(f"Total count: {total_count}")


sample_count = 10000


def sample_for_labeling():
    """
    从每个主题中随机抽取10000条推文进行标注。
    分层抽样，这10000条从每一个文件里面平均抽取
    存储为 ORIGINAL_TWEETS_DIR/labeling_sample/topic.pickle4
    """
    for topic in handling_targets:
        if os.path.exists(
            os.path.join(ORIGINAL_TWEETS_DIR, "labeling_sample", f"{topic}.parquet")
        ):
            print(f"File already exists for {topic}. Skipping...")
            continue
        folder = os.path.join(TWITTER_DIR, f"{topic}-opinion")
        files = glob.glob(os.path.join(folder, "*.pickle4"))
        topic_date_file = {}
        total_count = 0
        for file in files:
            data = pd.read_pickle(file)
            topic_date_file[file] = data
            total_count += len(data)

        sample_rate = sample_count / total_count

        samples = []
        for file, data in topic_date_file.items():
            sample_data = data.sample(frac=sample_rate, random_state=2025)
            samples.append(sample_data)

        # 将所有的 sample_data 合并
        all_data = pd.concat(samples)
        print(f"Sampled {len(all_data)} lines from {topic}")
        sample_dir = os.path.join(ORIGINAL_TWEETS_DIR, "labeling_sample")
        os.makedirs(sample_dir, exist_ok=True)
        all_data.to_parquet(
            os.path.join(sample_dir, f"{topic}.parquet"),
            engine="fastparquet",
        )


if __name__ == "__main__":
    fire.Fire(
        {
            "all_clean": clean_all,
            "single_clean": clean_original_tweets,
            "count": line_counter,
            "sample": sample_for_labeling,
        }
    )
