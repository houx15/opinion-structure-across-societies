import os
import json
import time

import pandas as pd

import ahocorasick
from datetime import datetime, timedelta

from cleaning.settings import *
from cleaning.weibo.archive_utils import *

import argparse


TEXT_DIR = os.path.join(WEIBO_WORK_DIR, "keyword_text_data")
if not os.path.exists(TEXT_DIR):
    os.makedirs(TEXT_DIR)

def log(text, lid=None):
    output = os.path.join(WEIBO_WORK_DIR, "logs", f"keyword_log_{lid}.txt" if lid is not None else "log.txt")
    with open(output, "a") as f:
        f.write(f"{text}\n")


def get_zipped_fresh_data_file(year, date):
    """
    date should be yyyy-mm-dd format
    """
    return f"{DATA_SOURCE_DIR}/{year}/freshdata/weibo_freshdata.{date}.7z"


def get_unzipped_fresh_data_folder(year):
    return f"{WEIBO_WORK_DIR}/text_working_data/{year}/"


def get_unzipped_fresh_data_file(year, date):
    if date == "2020-06-30":
        return f"{WEIBO_WORK_DIR}/text_working_data/{year}/weibo_2020-06-30.csv"
    elif date in ["2017-01-11", "2016-07-24", "2016-08-09"]:
        return f"{WEIBO_WORK_DIR}/text_working_data/{year}/weibo_log/weibo_freshdata.{date}.csv"
    return f"{WEIBO_WORK_DIR}/text_working_data/{year}/weibo_freshdata.{date}"


def delete_unzipped_fresh_data_file(year, date):
    """
    处理完毕之后需要删除文件
    """
    file_path = get_unzipped_fresh_data_file(year, date)
    # 删除文件
    try:
        os.remove(file_path)
        print(f"文件 {file_path} 已成功删除。")
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在。")
    except Exception as e:
        print(f"删除文件时发生错误: {e}")


def unzip_one_fresh_data_file(year, date):
    """
    date should be yyyy-mm-dd format
    """
    unzipped_file_path = get_unzipped_fresh_data_file(year, date)

    # 检查文件是否已经解压
    if os.path.exists(unzipped_file_path):
        print(f"文件 {unzipped_file_path} 已经存在，跳过解压。")
        return unzipped_file_path

    # 如果文件不存在，则进行解压
    zipped_file_path = get_zipped_fresh_data_file(year, date)
    unzipped_dir = get_unzipped_fresh_data_folder(year)
    result = extract_single_7z_file(
        file_path=zipped_file_path, target_folder=unzipped_dir
    )

    if os.path.exists(unzipped_file_path):
        print(f"文件 {unzipped_file_path} 解压成功。")
        return unzipped_file_path
    else:
        print(f"文件 {unzipped_file_path} 解压失败。")
        return None



def process_chunk_special(chunk, automation, result_set):
    all_ids = set()
    for line in chunk:
        for end_index, (kid, keyword) in automation.iter(line):
            """
            Line format (2020-06-30 dump): quoted CSV, fields: id, crawler_time, crawler_time_stamp, is_retweet, user_id, nick_name, avatar, user_type, weibo_id, weibo_content, zhuan, ping, zhan, url, device, locate, time, time_stamp, r_user_id, r_nick_name, r_user_type, r_weibo_id, r_weibo_content, ..., d
            """
            line_data = line.strip().strip('"').split('","')
            if line_data[8] in all_ids:
                continue
            all_ids.add(line_data[8])
            try:
                weibo_content = line_data[9].replace('\n', ' ') if line_data[3] == "0" else line_data[9].replace('\n', ' ') + '//' + line_data[22].replace('\n', ' ')
                result_set.add((kid,line_data[8],line_data[13],line_data[4],line_data[2],line_data[3],line_data[10],line_data[11],line_data[12],weibo_content))
            except:
                continue

def process_chunk(date, chunk, automation, result_set):
    all_ids = set()
    for line in chunk:
        for end_index, (kid, keyword) in automation.iter(line):
            """
            Line format: '<id>\t{json}' with keys id, crawler_time, is_retweet, user_id, nick_name, user_type, weibo_id, weibo_content, zhuan, ping, zhan, url, time, time_stamp, r_* (retweeted post), ..., d
            """
            line_data = line.strip()
            if date == datetime(2020, 6, 30):
                """
                Line format (2020-06-30 dump): quoted CSV, fields: id, crawler_time, crawler_time_stamp, is_retweet, user_id, nick_name, avatar, user_type, weibo_id, weibo_content, zhuan, ping, zhan, url, device, locate, time, time_stamp, r_user_id, r_nick_name, r_user_type, r_weibo_id, r_weibo_content, ..., d
                """
                line_data = line.split('","')
                if line_data[8] in all_ids:
                    continue
                all_ids.add(line_data[8])
                if len(line_data) < 24:
                    continue
                weibo_content = line_data[9].replace('\n', ' ') if line_data[3] == "0" else line_data[9].replace('\n', ' ') + '//' + line_data[22].replace('\n', ' ')
                result_set.add((kid,line_data[8],line_data[13],line_data[4],line_data[17],line_data[3],line_data[10],line_data[11],line_data[12],weibo_content))
            # 判断date(datetime)是否比2019-08-09晚
            elif date >= datetime(2019, 8, 9):
            # if year >= 2020:
                line_data = line.strip().split("\t")
                try:
                    data = json.loads(line_data[1])
                except IndexError as e:
                    print(f"IndexError occurred: {e}")
                    continue
                except json.JSONDecodeError as e:
                    print(f"JSONDecodeError: {e}")
                    # 打印出错误位置
                    print(f"Error at line {e.lineno}, column {e.colno}")
                    # 打印出错误字符位置
                    print(f"Error at character {e.pos}, {line_data[1][int(e.pos)-20: int(e.pos)+20]}")
                    continue
                if data['weibo_id'] in all_ids:
                    continue
                all_ids.add(data['weibo_id'])
                try:
                    weibo_content = data['weibo_content'].replace('\n', ' ') if data['is_retweet'] == "0" else data['weibo_content'].replace('\n', ' ') + '//' + data['r_weibo_content'].replace('\n', ' ')
                    result_set.add((kid,data['weibo_id'],data['url'],data['user_id'],data['time_stamp'],data['is_retweet'],data['zhuan'],data['ping'],data['zhan'],weibo_content))
                except KeyError:
                    continue
            else:
                line_data = line.split("\t")
                if len(line_data) < 24:
                    continue
                if line_data[8] in all_ids:
                    continue
                all_ids.add(line_data[8])
                weibo_content = line_data[9].replace('\n', ' ') if line_data[3] == "0" else line_data[9].replace('\n', ' ') + '//' + line_data[22].replace('\n', ' ')
                result_set.add((kid,line_data[8],line_data[13],line_data[4],line_data[17],line_data[3],line_data[10],line_data[11],line_data[12],weibo_content))

def keyword_loader():
    topic_info = pd.read_csv(WEIBO_TOPICS_CSV, usecols=["topic_id", "topic", "keyword"])
    # keyword是字符串，用;分隔了多个关键词
    keywords = []
    for index, row in topic_info.iterrows():
        # 如果keyword是none
        if not row["keyword"] or type(row["keyword"]) != str:
            continue
        keywords.extend(row["keyword"].split(";"))
    return keywords

def process_file(date, file_path):
    """
    处理单个文件并完成存储
    """
    keywords = keyword_loader()

    # 初始化 Aho-Corasick 自动机
    automation = ahocorasick.Automaton()

    for idx, keyword in enumerate(keywords):
        # 添加关键词，假设关键词格式为 #keyword#
        automation.add_word(f"{keyword}", (idx, keyword))
    automation.make_automaton()

    # 结果字典
    result_set = set()

    # 读取文件并分块处理
    chunk_size = 500000
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        chunk = []
        for line in file:
            chunk.append(line.strip())
            if len(chunk) == chunk_size:
                process_chunk(date, chunk, automation, result_set)
                chunk = []
        # 处理最后一个不满 chunk_size 的块
        if chunk:
            process_chunk(date, chunk, automation, result_set)
    
    return result_set


def check_processed(date):
    """
    检查指定日期的数据是否已经处理过
    """
    output_parquet_path = f"{TEXT_DIR}/{date}.parquet"
    return os.path.exists(output_parquet_path)


def append_to_parquet(date, results):
    """
    将数据追加到指定年份的 Parquet 文件中。
    :param year: 年份（如 2024）
    :param userids: 用户 ID 列表
    :param results: 结果数据列表
    """
    output_parquet_path = f"{TEXT_DIR}/{date}.parquet"
    columns = ["keyword_id", "weibo_id", "weibo_url", "user_id", "time_stamp", "is_retweet", "zhuan", "ping", "zhan", "weibo_content"]
    df = pd.DataFrame(list(results), columns=columns)

    # # 检查文件是否存在
    # if not os.path.exists(output_parquet_path):
    #     # 如果文件不存在，直接写入（新建文件）
    df.to_parquet(output_parquet_path, engine="fastparquet", index=False)
    # else:
    #     # 如果文件存在，以追加模式写入
    #     df.to_parquet(output_parquet_path, engine="fastparquet", index=False, append=True)
    #     log(f"追加数据到文件：{output_parquet_path}")


def process_year(year, mode, action="extract", force_update=False):
    """
    action:
    extract - 从文本中提取含有关键词的内容
    """
    start_date_options = [datetime(year, 1, 1), datetime(year, 5, 1), datetime(year, 9, 1)]
    end_date_options = [datetime(year, 4, 30), datetime(year, 8, 31), datetime(year, 12, 31)]
    if int(mode) == 3: # run all
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
    else:
        start_date = start_date_options[mode]
        end_date = end_date_options[mode]

    current_date = start_date

    date_range = [start_date + timedelta(days=n) for n in range((end_date - start_date).days + 1)]
    
    for current_date in date_range:
        if check_processed(current_date.strftime("%Y-%m-%d")) and not force_update:
            print(f"数据 {current_date.strftime('%Y-%m-%d')} 已经处理过，跳过。")
            continue
        date_str = current_date.strftime("%Y-%m-%d")
        file_path = unzip_one_fresh_data_file(year, date_str)
        # file_path = f"{WEIBO_WORK_DIR}/text_working_data/{year}/weibo_freshdata.test"
        if file_path is None:
            continue
        start_timestamp = int(time.time())
        if action == "extract":
            results = process_file(current_date, file_path)
            append_to_parquet(date_str, results)

            log(
                f"处理 {date_str} 完成，耗时 {int(time.time()) - start_timestamp} 秒。",
                f"{year}_{mode}",
            )
            print(f"finished {date_str} with {len(results)} records")

        delete_unzipped_fresh_data_file(year, date_str)
        


if __name__ == "__main__":
    # for y in [2016, 2017, 2018, 2019]:
    # add arg parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--mode", type=int, default=1)
    parser.add_argument("--action", type=str, default="extract")
    parser.add_argument("--force_update", type=bool, default=False)
    args = parser.parse_args()
    process_year(args.year, args.mode, args.action, args.force_update)
