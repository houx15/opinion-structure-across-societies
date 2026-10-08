"""Topic keywords for Twitter: noun/verb counts -> TF-IDF per topic and year.

1. process / process_all: spaCy lemmas of NOUN and VERB tokens (alphabetic,
   length > 1) in relevant tweets (BERT relevance == 1) of users located in
   the region -> <TWITTER_WORD_COUNT_DIR>/<mode>/noun-verb-<topic>-<year>.parquet
2. merge_all: pooled "merged" year.
3. tf_idf: per year, keep each topic's top 1% words by count;
   TF = count / sum(count); IDF = log((T + 1) / (df + 1)) over the T topics of
   the region; TF-IDF = TF * IDF ->
   data/tf_idf/tf_idf_<src>/tf-idf-<topic>-<year>.parquet

    python -m analysis.semantic.tfidf_twitter pipeline --mode us

Runs on the cluster (raw tweets, spaCy en_core_web_sm).
"""

import glob
import json
import os
import time
from collections import Counter, defaultdict

import fire
import numpy as np
import pandas as pd

from common.paths import TF_IDF_DIR
from config import cfg

original_dir = cfg.TWITTER_DIR
output_dir = cfg.TWITTER_WORD_COUNT_DIR
YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

MODE_SOURCE = {"us": "twitter", "eu": "eutwitter", "en": "entwitter", "eu_nen": "eu_nentwitter"}

# Folder of each topic's tweets under TWITTER_DIR (+ "-opinion" after BERT prediction).
folder_map = {
    "abo": "abortion-merge",
    "clc": "climate-change-merge",
    "gun": "gun-merge",
    "sxo": "sexual-orientation-merge",
    "vac": "vaccine-mandate",
    "soc": "social-media",
    "dpp": "death-penalty",
    "minwage": "minimum-wage",
    "swe": "sex-work-legalization",
    "ubi": "universal-basic-income",
}

# Topics entering the IDF of each region (as in the published TF-IDF tables).
TFIDF_TOPICS = {
    "us": ["abo", "clc", "gun", "sxo", "vac", "soc", "dpp", "minwage", "ubi"],
    "eu": ["abo", "clc", "gun", "sxo", "vac", "soc", "dpp", "minwage", "swe", "ubi"],
    "en": ["abo", "clc", "gun", "sxo", "vac", "soc", "dpp", "minwage", "swe", "ubi"],
    "eu_nen": ["abo", "clc", "sxo", "vac", "soc", "dpp", "minwage", "swe", "ubi"],
}


def log(message, mode):
    with open(f"process-{mode}.log", "a") as log_file:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")


def get_keep_ids(mode="en"):
    """Author ids located in the region (cleaning/twitter/location)."""
    if mode not in MODE_SOURCE:
        raise ValueError(f"Invalid mode: {mode}")
    with open(os.path.join(cfg.LOCATION_DIR, f"{mode}_user_ids.json"), "r") as f:
        return set(json.load(f))


def year_noun_verb_extractor(topic, year, mode):
    keep_ids = get_keep_ids(mode)
    topic = str(topic)
    folder = os.path.join(original_dir, f"{folder_map[topic]}-opinion")
    parquet_files = glob.glob(os.path.join(folder, f"{year}-*-text.pickle4"))

    if len(parquet_files) == 0:
        print(f"No files found for topic {topic} in {year}.")
        return None

    word_counter = Counter()
    import spacy

    nlp = spacy.load(
        "en_core_web_sm", disable=["ner", "parser"]
    )  # 只做分词和词性标注，提升速度

    for file in parquet_files:
        # start_time = time.time()
        opinion_df = pd.read_pickle(file)
        df = opinion_df
        if "relevance" not in df.columns:
            print(f"File {file} does not contain 'relevance' column.")
            continue
        if "author_id" not in df.columns:
            print(f"File {file} does not contain author id")
            continue
        before = len(df)
        df = df[df["author_id"].isin(keep_ids)]
        log(f"mode: {mode}, before filter: {before}, after: {len(df)}, (year-{year}, topic-{topic})", mode)
        df = df[(df["relevance"] == 1)]

        # 确保 'tweet' 列存在
        text_column = "tweet" if "tweet" in df.columns else "text"
        contents = df[text_column].dropna().tolist()

        # spaCy批量处理，提升效率
        for doc in nlp.pipe(
            contents, batch_size=1000, n_process=4
        ):  # n_process可并行加速，按需调整
            # 提取名词和动词，且长度大于1
            meaningful_words = [
                token.lemma_.lower()
                for token in doc
                if token.pos_ in {"NOUN", "VERB"}
                and len(token.lemma_) > 1
                and token.is_alpha
            ]
            word_counter.update(meaningful_words)
        # log(f"Processed {file} in {time.time() - start_time:.2f} seconds.")

    return word_counter


def process(topic, year, mode):
    mode_out_dir = os.path.join(output_dir, mode)
    os.makedirs(mode_out_dir, exist_ok=True)
    output_file = os.path.join(mode_out_dir, f"noun-verb-{topic}-{year}.parquet")
    if os.path.exists(output_file):
        print(f"{topic}-{year}-{mode} exists, skip...")
        return
    word_counter = year_noun_verb_extractor(topic, year, mode)
    if word_counter is None:
        return
    df = pd.DataFrame(word_counter.items(), columns=["word", "count"])
    df.to_parquet(output_file, engine="fastparquet", index=False)


def process_all_years(topic, mode):
    for year in YEARS:
        process(topic, year, mode)
        print(f"Processed {topic} - {mode} for year {year}.")


def merge_years(topic, mode):
    mode_out_dir = os.path.join(output_dir, mode)
    all_files = glob.glob(os.path.join(mode_out_dir, f"noun-verb-{topic}-20*.parquet"))
    all_dataframes = []
    for file in all_files:
        df = pd.read_parquet(file, engine="fastparquet")
        all_dataframes.append(df)

    merged_df = pd.concat(all_dataframes, ignore_index=True)
    merged_df = merged_df.groupby("word", as_index=False).agg({"count": "sum"})
    merged_df = merged_df.sort_values(by="count", ascending=False)
    print(f"Merged {len(all_files)} files for topic {topic}, shape: {merged_df.shape}.")

    output_file = os.path.join(mode_out_dir, f"noun-verb-{topic}-merged.parquet")
    merged_df.to_parquet(output_file, engine="fastparquet", index=False)
    print(f"Saved merged file to {output_file}.")


def data_analysis(topic, mode):
    mode_out_dir = os.path.join(output_dir, mode)
    merged_data = pd.read_parquet(
        os.path.join(mode_out_dir, f"noun-verb-{topic}-merged.parquet"),
        engine="fastparquet",
    )
    print(f"Data for topic {topic}:")
    print(f"Total unique words: {merged_data.shape[0]}")
    print(
        f"Count > 100: {merged_data[merged_data['count'] > 100].shape[0]}, ratio: {merged_data[merged_data['count'] > 100].shape[0] / merged_data.shape[0]:.2%}"
    )
    print(
        f"Count > 1000: {merged_data[merged_data['count'] > 1000].shape[0]}, ratio: {merged_data[merged_data['count'] > 1000].shape[0] / merged_data.shape[0]:.2%}"
    )

    for year in YEARS:
        if os.path.exists(
            os.path.join(mode_out_dir, f"noun-verb-{topic}-{year}.parquet")
        ):
            yearly_data = pd.read_parquet(
                os.path.join(mode_out_dir, f"noun-verb-{topic}-{year}.parquet"),
                engine="fastparquet",
            )
            print(f"Year {year}:")
            print(
                f"Count > 1000: {yearly_data[yearly_data['count'] > 1000].shape[0]}, ratio: {yearly_data[yearly_data['count'] > 1000].shape[0] / yearly_data.shape[0]:.2%}"
            )


def tf_idf(year, mode):
    """
    计算每个年份的TF-IDF值，并保存到文件中。
    """
    mode_out_dir = os.path.join(output_dir, mode)
    topic_dfs = []
    calc_ratio = 0.01

    for topic in TFIDF_TOPICS[mode]:
        if os.path.exists(
            os.path.join(mode_out_dir, f"noun-verb-{topic}-{year}.parquet")
        ):
            topic_df = pd.read_parquet(
                os.path.join(mode_out_dir, f"noun-verb-{topic}-{year}.parquet"),
                engine="fastparquet",
            )
            topic_df["tid"] = topic
            topic_df = topic_df.nlargest(int(len(topic_df) * calc_ratio), "count")
            topic_df["TF"] = topic_df["count"] / topic_df["count"].sum()
            topic_dfs.append(topic_df)

    total_topics = len(topic_dfs)
    combined_df = pd.concat(topic_dfs, ignore_index=True)

    top_percent_count = defaultdict(int)

    for tid, group in combined_df.groupby("tid"):
        for word in group["word"]:
            top_percent_count[word] += 1

    combined_df["IDF"] = combined_df["word"].apply(
        lambda x: np.log((total_topics + 1) / (top_percent_count[x] + 1))
    )
    combined_df["TF-IDF"] = combined_df["TF"] * combined_df["IDF"]

    tfidf_dir = TF_IDF_DIR / f"tf_idf_{MODE_SOURCE[mode]}"
    for topic, group in combined_df.groupby("tid"):
        os.makedirs(tfidf_dir, exist_ok=True)
        output_file = os.path.join(tfidf_dir, f"tf-idf-{topic}-{year}.parquet")
        # sort by TF-IDF
        group = group.sort_values(by="TF-IDF", ascending=False)
        group.to_parquet(output_file, engine="fastparquet", index=False)
        print(f"Saved TF-IDF for {topic} in {year} to {output_file}")
        print(
            f"TF-IDF > 0: {group[group['TF-IDF'] > 0].shape[0]}, ratio: {group[group['TF-IDF'] > 0].shape[0] / group.shape[0]:.2%}"
        )
        print(group.head(10))


def merge_all(mode):
    for topic in TFIDF_TOPICS[mode]:
        merge_years(topic, mode)
        print(f"Merged all years for topic {topic}.")
        data_analysis(topic, mode)
        print(f"Data analysis completed for topic {topic}.")


def all_tf_idf(mode):
    for year in YEARS:
        tf_idf(year, mode)
        print(f"TF-IDF calculation completed for year {year}.")
    tf_idf("merged", mode)

def pipeline(mode):
    for topic in TFIDF_TOPICS[mode]:
        process_all_years(topic, mode)

    merge_all(mode)
    all_tf_idf(mode)
    print("pipeline finished")


def topic_pipeline(topic, mode):
    process_all_years(topic, mode)
    merge_years(topic, mode)
    data_analysis(topic, mode)


if __name__ == "__main__":
    fire.Fire(
        {
            "process": process,
            "process_all": process_all_years,
            "merge_all": merge_all,
            "tf_idf": all_tf_idf,
            "pipeline": pipeline,
            "topic_pipe": topic_pipeline
        }
    )
