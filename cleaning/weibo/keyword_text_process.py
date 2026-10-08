"""
1. remove duplicate ones
2. categorize all the text into different topics
3. generate 10,000 samples for each topic (for training)
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import fire
import json

from collections import defaultdict
from cleaning.settings import *

TEXT_DIR = os.path.join(WEIBO_WORK_DIR, "keyword_text_data")

def handle_retweet(text):
    return text.split("//")[0]

def keyword_grammar_loader():
    topic_info = pd.read_csv(WEIBO_TOPICS_CSV, usecols=["topic_id", "topic", "keyword"])
    # keyword是字符串，用;分隔了多个关键词
    keyword_search_grammar = {}
    for index, row in topic_info.iterrows():
        # 如果keyword是none
        if not row["keyword"] or type(row["keyword"]) != str:
            continue
        keyword_search_grammar[row["topic_id"]] = "|".join(row["keyword"].split(";"))
    return keyword_search_grammar


def deduplicate_parquet(year):
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    date_range = [start_date + timedelta(days=n) for n in range((end_date - start_date).days + 1)]

    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        parquet_path = f"{TEXT_DIR}/{date_str}.parquet"

        if not os.path.exists(parquet_path):
            continue

        df = pd.read_parquet(parquet_path)
        if df.empty:
            print(f"Warning: {parquet_path} is empty, removing...")
            os.remove(parquet_path)
            continue

        df.drop_duplicates(subset='weibo_url', inplace=True)
        df["original_weibo_content"] = df["weibo_content"].apply(handle_retweet)
        df.to_parquet(parquet_path, engine='fastparquet', index=False)


grammar_mapping = keyword_grammar_loader()

def single_file_preprocess(date):
    topic_data_dir = WEIBO_TOPIC_KEYWORD_DIR
    date_str = date.strftime('%Y-%m-%d')
    file_path = f"{TEXT_DIR}/{date_str}.parquet"

    if not os.path.exists(file_path):
        return None
    
    data_count = {}
    
    data = pd.read_parquet(file_path, engine='fastparquet')
    for topic_id, match_grammar in grammar_mapping.items():
        topic_output_dir = f"{topic_data_dir}/{topic_id}"
        if not os.path.exists(topic_output_dir):
            os.makedirs(topic_output_dir)
        topic_data = data[data["original_weibo_content"].str.contains(match_grammar)].copy()
        if topic_data.empty:
            data_count[topic_id] = 0
            continue
        topic_data.loc[:, "topic_id"] = topic_id
        topic_data.to_parquet(f"{topic_output_dir}/{date_str}.parquet", engine='fastparquet', index=False)

        data_count[topic_id] = len(topic_data)
    
    return data_count

sample_count = 10000

def sample(topic_id, topic_date_count):
    """
    topic_date_count:
    key: date yyyy-mm-dd
    value: count of data, int

    weighted sampling based on the count of a particular day
    """
    topic_data_dir = os.path.join(WEIBO_TOPIC_KEYWORD_DIR, str(topic_id))
    if not os.path.exists(topic_data_dir):
        return None

    date_count = sorted(topic_date_count.items(), key=lambda x: x[1], reverse=True)
    total_count = sum([x[1] for x in date_count])
    sample_rate = sample_count / total_count

    samples = []

    for date_str, count in date_count:
        # date_str = date.strftime('%Y-%m-%d')
        if not os.path.exists(f"{topic_data_dir}/{date_str}.parquet"):
            continue
        data = pd.read_parquet(f"{topic_data_dir}/{date_str}.parquet", engine='fastparquet')
        sample_data = data.sample(frac=sample_rate, random_state=2025)
        samples.append(sample_data)
    
    topic_sample = pd.concat(samples)
    sample_dir = os.path.join(WEIBO_WORK_DIR, "topic_keyword_data_sample")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    topic_sample.to_parquet(f"{sample_dir}/{topic_id}.parquet", engine='fastparquet', index=False)

def pipeline(year, action, topic_id = None):
    assert action in ["deduplicate", "preprocess", "sample"]
    if action == "deduplicate":
        deduplicate_parquet(year)
    elif action == "preprocess":
        topic_date_count = defaultdict(dict)

        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        date_range = [start_date + timedelta(days=n) for n in range((end_date - start_date).days + 1)]
        for date in date_range:
            data_count = single_file_preprocess(date)
            if data_count is not None:
                for topic_id, count in data_count.items():
                    topic_date_count[topic_id][date.strftime('%Y-%m-%d')] = count
        sample_dir = os.path.join(WEIBO_WORK_DIR, "topic_keyword_data_sample")
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        with open(f"{sample_dir}/{year}_date_count.json", "w") as f:
            json.dump(topic_date_count, f)
    elif action == "sample":
        # assert topic_id is not None
        topics = pd.read_csv(WEIBO_TOPICS_CSV, usecols=["topic_id"])["topic_id"].tolist()
        if topic_id is not None:
            topics = [topic_id]
        
        for topic_id in topics:
            topic_id = str(topic_id)

            all_topic_date_count = {}
            for year in range(2016, 2024):
                with open(os.path.join(WEIBO_WORK_DIR, "topic_keyword_data_sample", f"{year}_date_count.json")) as f:
                    topic_date_count = json.load(f)
                    cur_topic_date_count = topic_date_count.get(topic_id, {})
                    all_topic_date_count.update(cur_topic_date_count)
            
            if all_topic_date_count == {}:
                continue
            
            sample(topic_id, all_topic_date_count)

if __name__ == '__main__':
    fire.Fire(pipeline)


