import fire
import glob

import pandas as pd
import pyarrow.parquet as pq

from datetime import datetime, timedelta

import os

from cleaning.bert.utils import sentence_cleaner
from cleaning.settings import *


def clean_weibo_opinion(topic: str):
    topic = str(topic)
    start_date = datetime(2016, 1, 1)
    end_date = datetime(2023, 12, 31)

    first_withouth_binary_result = None

    date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

    data_path = os.path.join(WEIBO_TOPIC_KEYWORD_DIR, topic)

    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")
        data_file = os.path.join(data_path, f"{date_str}.parquet")
        if not os.path.exists(data_file):
            # print(f"File {data_file} does not exist.")
            continue
        df = pd.read_parquet(data_file, engine="fastparquet")
        # if "cleaned_weibo_content" not in df.columns:
        #     df["cleaned_weibo_content"] = df["weibo_content"].apply(lambda x: sentence_cleaner("weibo", x))
        #     df.to_parquet(data_file)
        
        if "relevance" not in df.columns and first_withouth_binary_result is None:
            first_withouth_binary_result = date_str
            break
    
    with open("first_withouth_binary_result.txt", "a") as f:
        f.write(f"{topic}: {first_withouth_binary_result}\n")


start_date_dict = {
    "0": datetime(2019, 2, 6),
    "1": datetime(2020, 3, 4),
    "4": datetime(2019, 6, 22),
    "11": datetime(2019, 3, 14),
    "14": datetime(2020, 9, 4),
    "16": datetime(2020, 1, 1)
}

class YearDataMerge(object):
    def __init__(self, topic: str, task_type: str):
        self.topic = topic
        self.start_date = datetime(2016, 1, 1)
        # if topic in start_date_dict:
        #     self.start_date = start_date_dict[topic]
        self.end_date = datetime(2023, 12, 31)
        self.date_range = [self.start_date + timedelta(days=i) for i in range((self.end_date - self.start_date).days + 1)]
        self.data_path = os.path.join(WEIBO_TOPIC_KEYWORD_DIR, self.topic)

        self.task_type = task_type
        self.output_dir = os.path.join(self.data_path, "year_data") if task_type == "regression" else os.path.join(self.data_path, "binary_year_data")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.cache_files = []
        self.cumulative_count = 0
        self.cur_year = None
        self.cur_year_count = 0
    
    def save(self):
        output_path = os.path.join(self.output_dir, f"{self.cur_year}-{self.cur_year_count}.parquet")
        if os.path.exists(output_path):
            print(f"{output_path} already exists.")
            raise ValueError(f"{output_path} already exists.")
        combined_df = pd.concat(self.cache_files)
        combined_df.to_parquet(output_path)
        del combined_df

        self.cache_files = []
        self.cumulative_count = 0
        self.cur_year_count += 1
    
    def _should_skip_file(self, all_columns):
        if self.task_type == "regression":
            return "opinion" in all_columns
        elif self.task_type == "binary":
            return "relevance" in all_columns
        return False
    
    def _read_parquet_data(self, data_file, date_str):
        parquet_file = pq.ParquetFile(data_file)
        all_columns = parquet_file.schema.names

        required_columns = ["weibo_id", "cleaned_weibo_content"]

        # if not all(col in all_columns for col in required_columns):
        #     raise ValueError(f"File {data_file} does not have required columns: {required_columns}")
        if self._should_skip_file(all_columns):
            print(f"File {data_file} skipped.")
            return None
        
        columns_to_read = required_columns.copy()
        if self.task_type == "regression":
            if "relevance" not in all_columns:
                raise ValueError(f"File {data_file} does not have 'relevance' column.")
            columns_to_read.append("relevance")
        
        df = pd.read_parquet(data_file, engine="fastparquet", columns=columns_to_read)

        if self.task_type == "regression":
            df = df[df["relevance"] == 1]
        df["date"] = date_str
        return df[required_columns + ["date"]].copy()
    
    def process(self):
        for date in self.date_range:
            if self.cur_year is None:
                self.cur_year = date.year
            if date.year != self.cur_year:
                self.save()
                self.cur_year = date.year
                self.cur_year_count = 0

            date_str = date.strftime("%Y-%m-%d")
            print(f"Processing {date_str}...")
            data_file = os.path.join(self.data_path, f"{date_str}.parquet")
            if not os.path.exists(data_file):
                # print(f"File {data_file} does not exist.")
                continue
            df = self._read_parquet_data(data_file, date_str)
            if df is None:
                continue
            self.cache_files.append(df)
            self.cumulative_count += len(df)
            
            if self.cumulative_count > 840000:
                self.save()
        
        if len(self.cache_files) > 0:
            self.save()


def merge(topic: str, task_type: str):
    topic = str(topic)
    merger = YearDataMerge(topic, task_type)
    merger.process()


def recover_merged_data_to_daily_parquet(topic: str, task_type: str, year: int = None):
    topic = str(topic)
    data_path = os.path.join(WEIBO_TOPIC_KEYWORD_DIR, topic)

    if task_type == "regression":
        year_path = os.path.join(data_path, "year_data")
    elif task_type == "binary":
        year_path = os.path.join(data_path, "binary_year_data")
    else:
        raise ValueError("task_type should be either 'regression' or 'binary'.")
    
    if not os.path.exists(year_path):
        raise ValueError(f"{year_path} does not exist.")
    
    if year is None:
        parquet_files = glob.glob(os.path.join(year_path, "*.parquet"))
    else:
        parquet_files = glob.glob(os.path.join(year_path, f"{year}-*.parquet"))
    if len(parquet_files) == 0:
        raise ValueError(f"No parquet files found in {year_path}.")
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file, engine="fastparquet")
        if task_type == "binary":
            assert "relevance" in df.columns
        if task_type == "regression":
            assert "opinion" in df.columns
        if "date" not in df.columns:
            raise ValueError("The dataframe does not have a 'date' column.")
        
        # get all date unique values
        all_dates = df["date"].unique()
        for date in all_dates:
            date_df = df[df["date"] == date]
            # date_str = date.strftime("%Y-%m-%d")
            daily_file = os.path.join(data_path, f"{date}.parquet")
            if os.path.exists(daily_file):
                daily_df = pd.read_parquet(daily_file, engine="fastparquet")
                # merge date_df to daily_df based on weibo_id
                # date_df only keep the necessary columns, relevance for binary task, and opinion for regression task
                if task_type == "binary":
                    date_df = date_df[["weibo_id", "relevance"]]
                elif task_type == "regression":
                    date_df = date_df[["weibo_id", "opinion"]]
                else:
                    raise ValueError("task_type should be either 'regression' or 'binary'.")
                daily_df = daily_df.merge(date_df, on="weibo_id", how="left")
                daily_df.to_parquet(daily_file)
            else:
                raise ValueError(f"{daily_file} does not exist.")
        
        print(f"Processed {parquet_file} and merged data to daily parquet files.")


def check(topic: str, task_type: str):
    topic = str(topic)
    data_path = os.path.join(WEIBO_TOPIC_KEYWORD_DIR, topic)
    
    parquet_files = glob.glob(os.path.join(data_path, "*.parquet"))

    result = ""
    
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file, engine="fastparquet")
        if task_type == "binary":
            # relevance = na
            assert "relevance" in df.columns
            if df["relevance"].isna().sum() > 0:
                result += f"{parquet_file} has na in relevance\n"
        if task_type == "regression":
            # relevance == 1 & opinion == na
            try:
                assert "opinion" in df.columns
            except AssertionError:
                result += f"{parquet_file} does not have opinion column\n"
                continue
            if df.loc[df["relevance"] == 1, "opinion"].isna().any():
                result += f"{parquet_file} has different count of relevance and opinion\n"
    
    print(result)


if __name__ == "__main__":
    fire.Fire(
        {
            "merge": merge,
            "recover": recover_merged_data_to_daily_parquet,
            "check": check
        }
    )
