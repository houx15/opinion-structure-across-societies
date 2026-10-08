"""Topic keywords for Weibo: noun/verb counts -> TF-IDF per topic and year.

Same procedure as tfidf_twitter: jieba POS tags starting with n / v (length
> 1) in relevant posts with an opinion (BERT relevance == 1) ->
<WEIBO_NOUN_COUNT_DIR>/<topic>_<year>_noun_count.parquet (columns Noun,
Frequency); per year keep the top 1% words by Frequency of each of the 15
topics; TF = Frequency / sum; IDF = log((T + 1) / (df + 1)); TF-IDF = TF * IDF
-> data/tf_idf/tf_idf_weibo/<topic>_<year>_tf_idf.parquet

    python -m analysis.semantic.tfidf_weibo all_years 10
    python -m analysis.semantic.tfidf_weibo merge
    python -m analysis.semantic.tfidf_weibo tf_idf

Runs on the cluster (keyword-matched Weibo posts with BERT predictions).
"""

import fire
import glob
import time

import os

import pandas as pd
import pyarrow.parquet as pq
import numpy as np

from collections import Counter

from collections import defaultdict

from common.paths import TF_IDF_DIR
from config import cfg

NOUN_COUNT_DIR = cfg.WEIBO_NOUN_COUNT_DIR
TFIDF_DIR = TF_IDF_DIR / "tf_idf_weibo"


def year_noun_extractor(topic, year):
    import jieba.posseg as pseg

    topic = str(topic)
    data_path = os.path.join(cfg.WEIBO_TOPIC_KEYWORD_DIR, topic)
    parquet_files = glob.glob(os.path.join(data_path, f"{year}-*.parquet"))
    # tokenizer = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
    # pos_tagger = hanlp.load('CTB5_POS_RNN')
    word_counter = Counter()
    # cleaned_weibo_content
    for file in parquet_files:
        # start_time = time.time()
        df = pd.read_parquet(file)
        if "relevance" not in df.columns or "opinion" not in df.columns:
            print(f"File {file} does not contain 'relevance' or 'opinion' columns.")
            continue
        df = df[(df["relevance"] == 1) & (df["opinion"].notnull())]
        
        # 确保 'cleaned_weibo_content' 列存在
        if 'cleaned_weibo_content' in df.columns:
            contents = df['cleaned_weibo_content'].dropna().tolist()
            
            for content in contents:
                # 分词
                # tokens = tokenizer(content)
                # # 词性标注
                # pos_tags = pos_tagger(tokens)
                
                # # 提取名词
                # nouns = [token for token, pos in zip(tokens, pos_tags) if pos.startswith('n')]
                
                # # 更新词频统计
                # noun_counter.update(nouns)
                # 使用Jieba进行分词和词性标注
                words = pseg.cut(content)
                
                # 提取具有实际含义的词（如名词和动词）
                meaningful_words = [word for word, flag in words if flag.startswith(('n', 'v')) and len(word) > 1]
                
                # 更新词频统计
                word_counter.update(meaningful_words)
        
        # with open("noun_counter.txt", "a") as f:
        #     f.write(f"Processed file: {file}, Time taken: {time.time() - start_time:.2f} seconds, line count: {len(contents)}\n")
    
    return word_counter


def process(topic, year):
    word_counter = year_noun_extractor(topic, year)
    counter_df = pd.DataFrame(word_counter.items(), columns=['Noun', 'Frequency'])
    counter_df.to_parquet(f"{NOUN_COUNT_DIR}/{topic}_{year}_noun_count.parquet", index=False)

def process_all_years(topic):
    for year in range(2016, 2024):
        print(f"Processing {topic} for the year {year}")
        process(topic, year)

def process_four_years(topic, type="1"):
    type = str(type)
    if type == "1":
        for year in range(2016, 2020):
            print(f"Processing {topic} for the year {year}")
            process(topic, year)
    elif type == "2":
        for year in range(2020, 2024):
            print(f"Processing {topic} for the year {year}")
            process(topic, year)

def merge_years(topic):
    all_files = glob.glob(f"{NOUN_COUNT_DIR}/{topic}_20*_noun_count.parquet")
    all_dataframes = []
    
    for file in all_files:
        df = pd.read_parquet(file)
        all_dataframes.append(df)
    
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    # 统计每个词的总频率
    merged_df = merged_df.groupby('Noun', as_index=False).agg({'Frequency': 'sum'})

    # 只保留长度>1的词
    merged_df = merged_df[merged_df['Noun'].str.len() > 1]
    # 排序
    merged_df = merged_df.sort_values(by='Frequency', ascending=False)
    # 打印shape
    print(f"Merged DataFrame shape: {merged_df.shape}")
    # 保存合并后的数据
    merged_df.to_parquet(f"{NOUN_COUNT_DIR}/{topic}_merged_noun_count.parquet", index=False)

def data_analysis(topic):
    merged_data = pd.read_parquet(f"{NOUN_COUNT_DIR}/{topic}_merged_noun_count.parquet")
    # print >10, >100, > 1000的数量
    print(f"Topic {topic} data analysis:")
    print(f"Total: {len(merged_data)}")
    print(f"Count > 10: {len(merged_data[merged_data['Frequency'] > 10])}, Ratio: {len(merged_data[merged_data['Frequency'] > 10]) / len(merged_data)}")
    print(f"Count > 100: {len(merged_data[merged_data['Frequency'] > 100])}, Ratio: {len(merged_data[merged_data['Frequency'] > 100]) / len(merged_data)}")
    print(f"Count > 1000: {len(merged_data[merged_data['Frequency'] > 1000])}, Ratio: {len(merged_data[merged_data['Frequency'] > 1000]) / len(merged_data)}")
    for year in range(2016, 2024):
        if year < 2020 and topic in ["16"]:
            continue
        year_df = pd.read_parquet(f"{NOUN_COUNT_DIR}/{topic}_{year}_noun_count.parquet")
        print(f"Year {year} Count > 1000: {len(year_df[year_df['Frequency'] > 1000])}, Ratio: {len(year_df[year_df['Frequency'] > 1000]) / len(year_df)}")


def tf_idf(year):
    year = str(year)
    topic_dfs = []
    calc_ratio = 0.01

    for topic in range(17):
        if topic in [3, 8]:
            continue
        topic = str(topic)
        if year in ["2016", "2017", "2018", "2019"] and topic in ["16"]:
            continue
        topic_df = pd.read_parquet(f"{NOUN_COUNT_DIR}/{topic}_{year}_noun_count.parquet")
        topic_df['tid'] = topic
        # 截取前calc_ratio的内容
        topic_df = topic_df.nlargest(int(len(topic_df) * calc_ratio), 'Frequency')
        topic_df['TF'] = topic_df['Frequency'] / topic_df['Frequency'].sum()
        topic_dfs.append(topic_df)
    
    total_topics = len(topic_dfs)

    combined_df = pd.concat(topic_dfs, ignore_index=True)

    top_percent_count = defaultdict(int)

    for tid, group in combined_df.groupby('tid'):
        # threshold = np.percentile(group['TF'], 100 * (1-calc_ratio))
        # top_words = group[group['TF'] >= threshold]['Noun']
        for word in group['Noun']:
            top_percent_count[word] += 1
    
    combined_df['IDF'] = combined_df['Noun'].apply(lambda x: np.log((total_topics + 1) / (top_percent_count[x] + 1)))

    combined_df['TF-IDF'] = combined_df['TF'] * combined_df['IDF']

    # result = combined_df.groupby('tid').apply(lambda x: x.nlargest(int(len(x) * calc_ratio), 'TF-IDF'))
    # result.reset_index(drop=True, inplace=True)

    # 将result分topic保存
    for topic, group in combined_df.groupby('tid'):
        # group = group[['Noun', 'Frequency', 'TF', 'IDF', 'TF-IDF']]
        # sort by TF-IDF
        group = group.sort_values(by='TF-IDF', ascending=False)
        os.makedirs(TFIDF_DIR, exist_ok=True)
        group.to_parquet(TFIDF_DIR / f"{topic}_{year}_tf_idf.parquet", index=False)
        print(f"year {year} topic {topic} TF-IDF saved.")
        print(group.head(10))
        # print num with TF-IDF larger than 0
        print(f"TF-IDF > 0: {len(group[group['TF-IDF'] > 0])}, Ratio: {len(group[group['TF-IDF'] > 0]) / len(group)}")

def merge_all():
    for topic in range(17):
        if topic in [3, 8]:
            continue
        topic = str(topic)
        merge_years(topic)
        print(f"Merged {topic} data.")


def analyze_all_topics():
    for topic in range(17):
        if topic in [3, 8]:
            continue
        topic = str(topic)
        data_analysis(topic)


def all_tf_idf():
    for year in range(2016, 2024):
        tf_idf(year)
    
    tf_idf("merged")


if __name__ == "__main__":
    fire.Fire({
        "single_year": process,
        "all_years": process_all_years,
        "four_years": process_four_years,
        "merge": merge_all,
        "analyze": analyze_all_topics,
        "tf_idf": all_tf_idf,
    })