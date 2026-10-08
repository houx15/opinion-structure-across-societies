import os
import json
import re

import jsonlines

import pandas as pd
import numpy as np

from openai import OpenAI
from collections import defaultdict

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns
import time

"""
配置工作路径
"""
from cleaning.settings import *

SOURCE_DIR = os.path.join(WEIBO_WORK_DIR, "crossover_topic_text") # 原始文本路径, 在路径中有{topic_id}.txt 的文本文件
EXAMPLE_PATH = os.path.join(WEIBO_WORK_DIR, "labeled-11-20.csv") # 标注数据文件
WORKING_DIR = os.path.join(WEIBO_WORK_DIR, "batch") # 工作目录，存储生成的batch file、output、error, 以及configs

"""
配置plt能够用中文
"""
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



new_topic_mapping = {
    "美国": ["中美间谍活动", "科学家受到美国政府歧视", "特朗普犯罪问题", "南海问题", "中美关系", "疫情中的政治争端", "华为相关事件", "孟晚舟事件", "中美制裁与反制裁", "芯片问题", "特斯拉相关讨论", "TikTok事件", "港台问题"],
    "日本": ["抗日战争与南京大屠杀", "731部队", "日本对俄罗斯制裁", "核污水排海", "疫情与东京奥运会", "中日关系", "安倍晋三遇刺"],
    "俄罗斯": ["俄乌战争", "中俄关系"],
    "外交政策": ["外交策略", "国防军事"],
    "医生": ["医患问题"],
    "中医": ["中医与新冠"],
    "医保": ["医疗保障"],
    "贪腐": ["偷税漏税问题", "北极鲶鱼", "贪腐官员落马"],
    "政府管理": ["辱华行为", "爱国主义教育"],
    "男女平等": ["家庭暴力", "性别平等", "包丽案", "货拉拉女乘客坠亡案", "阿里员工性骚扰事件", "大学生性侵事件", "未成年人性侵事件", "月经羞耻", "女性容貌身材焦虑", "性骚扰"],
    "LGBT": ["LGBT"],
    "婚姻": ["相亲问题", "婚恋状态", "催婚压力", "低俗婚闹", "天价彩礼", "婚姻政策"],
    "生育": ["恐婚恐育", "职场婚育歧视", "生育政策"],
    "能源与环境": ["限电", "河南暴雨", "氯乙烯泄漏", "气候变暖及其后果"],
    "就业问题": ["大学生就业选择", "大学生就业难", "就业形势", "职场文化", "00后整顿职场", "快递与外卖问题", "加班", "就业歧视", "工人权益"],
    "经济": ["存款", "收入水平", "消费水平", "国内经济形势", "经济风险", "消费刺激政策", "恒大事件", "房地产形势", "提前还贷风波", "房地产刺激政策"],
    "疫情防控": ["疫情防控政策", "疫情防控-留学生问题"],
    "新冠疫苗": ["疫苗相关"]
}


ALL_LABELS = ["支持", "中立", "不支持", "无关"]
ALL_LABELS_FIVE = ["非常支持", "支持", "中立", "不支持", "非常不支持", "无关"]




class BatchQwen:
    def __init__(self, topic_id, working_dir, task_name="qwen", force_analysis=False):
        self.topic_id = topic_id
        self.topic_working_dir = os.path.join(working_dir, f"topic-{topic_id}")
        self.force_analysis = force_analysis
        self.task_name = task_name
        if not os.path.exists(self.topic_working_dir):
            os.makedirs(self.topic_working_dir)
 
        if "qwen" in task_name:
            self.client = client = OpenAI(
                api_key=DASHSCOPE_API_KEY,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            self.model = "qwq-plus"
        elif "deepseek" in task_name:
            self.client = OpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
            )
            self.model = "deepseek-chat"
        elif "doubao" in task_name:
            self.client = OpenAI(
                api_key=DOUBAO_API_KEY,
                base_url=DOUBAO_BASE_URL,
            )
            self.model = DOUBAO_END_POINT
        elif "openai" in task_name:
            self.client = OpenAI(
                api_key=OPENAI_API_KEY,
                organization=OPENAI_ORGANIZATION,
                project=OPENAI_PROJECT,
            )
            self.model = "gpt-4o-mini"
        else:
            raise ValueError(f"Invalid model name {task_name}.")

        # self.model = "gpt-4o-mini" if "gpt" in task_name else "qwen-plus"

        self.real_time = True if task_name in realtime_llms else False
        self.manual = True if task_name in manual_llms else False
        
        self.load_configs()
    
    def process(self):
        """
        根据configs里面的status决定下一步行动
        """
        if self.force_analysis and self.configs["status"] == "submitted":
            self.configs["status"] = "retrieved"
        if self.configs["status"] == "init":
            texts = self.load_content()
            if self.topic_id in ["test", "test-5label"]:
                self.batch_file_preparation_for_test()
            else:
                self.batch_file_preparation(texts)
            self.submit_batch()
        elif self.configs["status"] == "generated":
            # 已经生成数据，但没有提交，未报错的情况下不应该出现这种情况
            self.submit_batch()
        elif self.configs["status"] == "submitted":
            # 已经提交，但未取回数据
            self.retrieve_batch()
        elif self.configs["status"] == "retrieved":
            # 已经取回数据，但未处理
            print(f"{self.topic_id} All data retrieved.")
            if self.topic_id in ["test", "test-5label"]:
                self.evaluate()
            else:
                self.analyze_user_opinion()
        elif self.configs["status"] == "error":
            # TODO 出现问题，需要遍历单个batch
            pass

        self.save_configs()

    
    def load_configs(self):
        """
        使用json文件存储
        {
            "batches": [], # 使用列表记录该topic所有的batch key（自主创建）
            "batch_info": {
                "batch_key1": { # 单个batch的信息
                    "task_id": "", # aliyun的task_id
                    "status": "", # init-初始化；generated-已生成文件；submitted-已提交；retrieved-已取回, 在部分报错时可以处理
                    "input_file_id": "", # aliyun的input file id
                }
            },
            "status": "", # init-初始化；generated-已生成文件；submitted-已提交；retrieved-已取回; error-存在问题,需要遍历单个batch的情况
        }
        """
        try:
            with open(os.path.join(self.topic_working_dir, f"configs.json"), "r") as f:
                configs = json.load(f)
        except FileNotFoundError:
            configs = {
                "batches": [],
                "batch_info": {},
                "status": "init"
            }
        
        self.configs = configs
    
    def save_configs(self):
        with open(os.path.join(self.topic_working_dir, f"configs.json"), "w") as f:
            json.dump(self.configs, f)

    
    def save_single_batch(self, batch_key, data):
        # 存储为jsonlines
        with jsonlines.open(os.path.join(self.topic_working_dir, f"{batch_key}.jsonl"), "w") as f:
            f.write_all(data)
        self.configs["batches"].append(batch_key)
        self.configs["batch_info"][batch_key] = {"task_id": None, "status": "generated"}
        self.save_configs()
    
    def prompt_composer(self, topic_id):
        raise NotImplementedError("Please implement the prompt composer.")
    
    def load_content(self):
        raise NotImplementedError("Please implement the load content method.")
    
    def batch_file_preparation(self, texts):
        """
        生成batch file
        """
        prompt = self.prompt_composer(self.topic_id)
        item_count = 0
        file_count = 0
        request_list = []
        all_custom_ids = []

        for text in texts:
            if len(text) == 2:
                manual_id, weibo_content = text
            else:
                manual_id = item_count
                weibo_content = text
            if manual_id in all_custom_ids:
                continue
            single_request = {
                "custom_id": f"{self.topic_id}-{manual_id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"文本：{weibo_content}\n回答："}
                    ],
                    "max_tokens": 2000
                }
            }
            request_list.append(single_request)
            all_custom_ids.append(manual_id)
            item_count += 1

            if item_count % MAX_REQUEST_PER_BATCH == 0:
                self.save_single_batch(f"batch-{file_count}", request_list)
                file_count += 1
                request_list = []
        
        if len(request_list) > 0:
            self.save_single_batch(f"batch-{file_count}", request_list)
            file_count += 1
        
        self.configs["status"] = "generated"
        self.save_configs()

    
    def batch_file_preparation_for_test(self):
        """
        生成batch file
        """
        item_count = 0
        file_count = 0
        request_list = []
        all_custom_ids = []

        example_df = pd.read_csv(EXAMPLE_PATH)
        # 去除 更新版statement-1121 列为空的数据
        example_df = example_df[example_df["更新版statement-1121"].notnull()]
        example_df = example_df[example_df[self.consistency_column] == 1]

        # 遍历example_df中所有话题id
        for topic_id in example_df["话题id"].unique():
            topic_df = example_df[example_df["话题id"] == topic_id]
            prompt = self.fewshot_prompt_composer(topic_id)
            # 遍历topic_df
            for index, row in topic_df.iterrows():
                # 如果是example，跳过
                if row["id"] in self.example_ids:
                    continue
                text = row["文本"]
                custom_id = f"{topic_id}-{row['id']}"
                if custom_id in all_custom_ids:
                    continue
                single_request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": f"文本：{text}\n回答："}
                        ],
                        "max_tokens": 2000
                    }
                }
                all_custom_ids.append(custom_id)
                request_list.append(single_request)
                item_count += 1
        
        if len(request_list) > 0:
            self.save_single_batch(f"batch-{file_count}", request_list)
            file_count += 1
        
        self.configs["status"] = "generated"
        self.save_configs()
    

    def load_processed_ids(self, output_file_path):
        self.processed_ids = set()
        if not os.path.exists(output_file_path):
            return
        with jsonlines.open(output_file_path) as rfile:
            for data in rfile:
                pid = str(int(data["custom_id"].split("-")[1]))
                self.processed_ids.add(pid)
        return
    

    def write_single_line_to_jsonl(self, line, output_file_path):
        with jsonlines.open(output_file_path, "a") as wfile:
            wfile.write(line)
        
    def realtime_run_single_batch(self, batch_file):
        batch_file_path = os.path.join(self.topic_working_dir, f"{batch_file}.jsonl")
        output_file_path = os.path.join(self.topic_working_dir, f"{batch_file}-output.jsonl")
        self.load_processed_ids(output_file_path)

        with open("templog.txt", "a") as f:
            f.write(f"{self.topic_working_dir}, batch_file: {batch_file}, processed_ids: {len(self.processed_ids)}\n")

        line_count = 0
        with jsonlines.open(batch_file_path) as rfile:
            for data in rfile:
                line_count += 1
                pid = data["custom_id"].split("-")[1]
                if pid in self.processed_ids:
                    continue
                if len(self.processed_ids) >= 5000 and self.task_name == "deepseek":
                    return "retrieved"
                # 定义最大重试次数
                max_retries = 1
                retries = 0

                while retries < max_retries:
                    try:
                        # 调用 API
                        response = self.client.chat.completions.create(**data["body"])
                        print("response", response)
                        response_dict = response.to_dict()
                        response_dict["custom_id"] = data["custom_id"]
                        self.write_single_line_to_jsonl(response_dict, output_file_path)
                        self.processed_ids.add(pid)
                        break

                    except Exception as e:
                        retries += 1
                        print(
                            f"Error processing custom_id {data['custom_id']} (Attempt {retries}/{max_retries}): {e}"
                        )

                        # 如果重试次数未达到最大值，等待一段时间后重试
                        if retries < max_retries:
                            time.sleep(2**retries)  # 指数退避机制：2秒、4秒、8秒...
                        else:
                            # 超过最大重试次数，记录错误并跳过
                            # error_data = {
                            #     "custom_id": data["custom_id"],
                            #     "error": str(e),
                            # }
                            # self.write_single_line_to_jsonl(
                            #     error_data, output_file_path
                            # )
                            print(
                                f"Failed to process custom_id {data['custom_id']} after {max_retries} attempts."
                            )

        # 如果processed ids与batchfile的长度一样
        if len(self.processed_ids) == line_count:
            return "retrieved"
        return "processing"
                

    def submit_batch(self):
        """
        提交batch
        """
        for batch_key in self.configs["batches"]:
            if self.configs["batch_info"][batch_key]["status"] == "generated":
                if self.manual is False and self.real_time is False:
                    batch_input_file = self.client.files.create(
                        file=open(os.path.join(self.topic_working_dir, f"{batch_key}.jsonl"), "rb"),
                        purpose="batch"
                    )
                    batch_input_file_id = batch_input_file.id

                    self.configs["batch_info"][batch_key]["input_file_id"] = batch_input_file_id
                    self.configs["batch_info"][batch_key]["status"] = "uploaded"
                    self.save_configs()

                    batch_info = self.client.batches.create(
                        input_file_id=batch_input_file_id,
                        endpoint="/v1/chat/completions",
                        completion_window="24h",
                    )
                    print(batch_info)
                    # TODO 处理报错
                    self.configs["batch_info"][batch_key]["task_id"] = batch_info.id
                    self.configs["batch_info"][batch_key]["status"] = "submitted"
                    self.save_configs()
        
        self.configs["status"] = "submitted"
        self.save_configs()

    def retrieve_batch(self):
        """
        取回batch
        """
        all_retrieved = True
        if self.real_time:
            for batch_key in self.configs["batches"]:
                retrieved = self.realtime_run_single_batch(batch_key)
                if retrieved != "retrieved":
                    all_retrieved = False
        elif self.manual:
            print("Manual model, please check the output file.")
            for batch_key in self.configs["batches"]:
                if not os.path.exists(os.path.join(self.topic_working_dir, f"{batch_key}-output.jsonl")):
                    print(f"Batch {batch_key} not retrieved.")
                    all_retrieved = False
        else:
            for batch_key in self.configs["batches"]:
                if self.configs["batch_info"][batch_key]["status"] == "submitted":
                    batch_info = self.client.batches.retrieve(self.configs["batch_info"][batch_key]["task_id"])
                    print(batch_info)
                    if batch_info.status == "completed":
                        output_file_id = batch_info.output_file_id
                        error_file_id = batch_info.error_file_id
                        if output_file_id is not None:
                            output_file_content = self.client.files.content(output_file_id).text
                            with open(os.path.join(self.topic_working_dir, f"{batch_key}-output.jsonl"), "w") as f:
                                f.write(output_file_content)

                        if error_file_id is not None:
                            error_file_content = self.client.files.content(error_file_id).text
                            with open(os.path.join(self.topic_working_dir, f"{batch_key}-error.jsonl"), "w") as f:
                                f.write(error_file_content)

                        self.configs["batch_info"][batch_key]["status"] = "retrieved"
                        self.save_configs()
                    else:
                        all_retrieved = False
                elif self.configs["batch_info"][batch_key]["status"] == "retrieved":
                    continue
                else:
                    all_retrieved = False
        
        if all_retrieved:
            self.configs["status"] = "retrieved"
            self.save_configs()
    

    def process_prediction_text(self, text):
        """
        returns: relevance (binary), opinion (int)
        e.g.:
        0, None
        1, 0
        1, -1
        1, 1
        """
        raise NotImplementedError("Please implement the process_prediction_text method.")
    
    def evaluate(self):
        raise NotImplementedError("Please implement the evaluate method.")
    
    def analyze_user_opinion(self):
        """
        分析用户的意见
        存储原始数据：user_opinions.json
        key：text id
        value：user id，opinion，relevance，timestamp
        """
        raise NotImplementedError("Please implement the analyze_user_opinion method.")


class TopicBatch(BatchQwen):
    def __init__(self, topic_id):
        super().__init__(topic_id, WORKING_DIR)
        self.example_ids = []
        self.consistency_column = "去掉程度-hyc"
        self.label_column = "consistency2"
    
    def prompt_composer(self, topic_id):
        """
        从example path 获得topic name，statement，example，拼接成prompt
        """
        example_df = pd.read_csv(EXAMPLE_PATH)
        topic_df = example_df[example_df["话题id"] == int(topic_id)]
        if topic_df.shape[0] == 0:
            raise ValueError(f"No example found for the topic {topic_id}. Please check the input topic id.")
        topic_name = topic_df["话题名"].iloc[0]
        statement = topic_df["更新版statement-1121"].iloc[0]

        total_label = ALL_LABELS_FIVE if "5label" in self.topic_id else ALL_LABELS
        prompt = f"""你是一个擅长处理微博数据的助手。我将给你提供一段关于{topic_name}的微博文本，请帮我判断这段微博文本是否支持下面的描述：{statement}。请从下列的选项中选择一个回复：{', '.join(total_label)}。不要输出任何其他内容。\n\n"""

        topic_df = topic_df[topic_df[self.consistency_column] == 1]
        print(f"Found {topic_df.shape[0]} consistent examples for the topic {topic_id}")
        example_content = "下面是一些例子：\n"
        has_example = 0

        for label in total_label:
            label_df = topic_df[topic_df[self.label_column] == label]
            if label_df.shape[0] == 0:
                print(f"For topic {topic_name}, no example found for the label {label}")
                continue
            # 随机选择一个example
            example = label_df.sample(1)
            example_text = example["文本"].iloc[0]
            example_content += f"文本：{example_text}\n"
            example_content += f"回答：{label}\n\n"
            has_example = 1
            self.example_ids.append(example["id"].iloc[0])
        
        if has_example == 1:
            prompt += example_content
        
        return prompt
    
    def load_content(self):
        texts = set()
        with open(os.path.join(SOURCE_DIR, f"{self.topic_id}.txt"), "r") as f:
            for line in f.readlines():
                line_split = line.strip().split(",")
                if len(line_split) < 8:
                    continue
                text = ",".join(line_split[7:])
                texts.add(text)
        return list(texts)

    def process_prediction_text(self, text):
        """
        returns: relevance (binary), opinion (int)
        e.g.:
        0, None
        1, 0
        1, -1
        1, 1
        """
        if "无关" in text:
            return 0, None
        if "5label" in self.topic_id:
            if "非常不支持" in text:
                return 1, -2
            if "非常支持" in text:
                return 1, 2
            if "不支持" in text:
                return 1, -1
            if "支持" in text:
                return 1, 1
            return 1, 0
        if "不支持" in text:
            return 1, -1
        if "支持" in text:
            return 1, 1
        return 1, 0
    
    def evaluate(self):
        prediction = defaultdict(dict)
        if self.topic_id not in ["test", "test-5label"]:
            print("no evaluation for non-test topic")
        
        for batch_key in self.configs["batches"]:
            with jsonlines.open(os.path.join(self.topic_working_dir, f"{batch_key}-output.jsonl"), "r") as f:
                for line in f:
                    tid = line["custom_id"].split("-")[1]
                    topic = line["custom_id"].split("-")[0]
                    prediction_text = line["response"]["body"]["choices"][0]["message"]["content"]
                    relevance, opinion = self.process_prediction_text(prediction_text)
                    prediction[tid] = {"relevance": relevance, "opinion": opinion, "topic": topic}
        
        true = defaultdict(dict)
        example_df = pd.read_csv(EXAMPLE_PATH)
        example_df = example_df[example_df[self.consistency_column] == 1]
        for index, row in example_df.iterrows():
            opinion = row[self.label_column]
            relevance, opinion = self.process_prediction_text(opinion)
            true[str(int(row["id"]))] = {"relevance": relevance, "opinion": opinion, "topic": row["话题id"]}
        
        # 计算relevance的accuracy
        relevance_ground_truth = []
        relevance_prediction = []
        opinion_ground_truth = []
        opinion_prediction = []

        for key, value in prediction.items():
            relevance_ground_truth.append(true[key]["relevance"])
            relevance_prediction.append(value["relevance"])

            if value["relevance"] == 1 and true[key]["relevance"] == 1:
                opinion_ground_truth.append(true[key]["opinion"])
                opinion_prediction.append(value["opinion"])
        

        # 计算relevance的accuracy, recall, precision, F1score
        relevance_accuracy = accuracy_score(relevance_ground_truth, relevance_prediction)
        recall = recall_score(relevance_ground_truth, relevance_prediction)
        precision = precision_score(relevance_ground_truth, relevance_prediction)
        f1 = f1_score(relevance_ground_truth, relevance_prediction)
        # 计算opinion的rmse
        opinion_rmse = np.sqrt(np.mean((np.array(opinion_ground_truth) - np.array(opinion_prediction))**2))

        with open(os.path.join(self.topic_working_dir, "evaluation.txt"), "w") as f:
            f.write(f"Relevance accuracy: {relevance_accuracy}\n")
            f.write(f"Relevance recall: {recall}\n")
            f.write(f"Relevance precision: {precision}\n")
            f.write(f"Relevance F1 score: {f1}\n")
            f.write(f"Opinion RMSE: {opinion_rmse}\n")
    
    def analyze_user_opinion(self):
        """
        分析用户的意见
        存储原始数据：user_opinions.json
        key：text id
        value：user id，opinion，relevance，timestamp
        """
        user_opinions = defaultdict(list)

        opinion_full_result = defaultdict(dict)

        if self.topic_id == "test":
            print("cannot analyze test")

        with open(os.path.join(WEIBO_WORK_DIR, "batch", "id_map.json"), "r") as f:
            topic_to_user_mapping = json.load(f)
            topic_to_user_mapping = topic_to_user_mapping[self.topic_id]
            # print(list(topic_to_user_mapping.keys())[0:5])
        
        for batch_key in self.configs["batches"]:
            with jsonlines.open(os.path.join(self.topic_working_dir, f"{batch_key}-output.jsonl"), "r") as f:
                for line in f:
                    item_count = line["custom_id"].split("-")[1]
                    topic = line["custom_id"].split("-")[0]
                    mapped_text_info = topic_to_user_mapping[item_count]
                    mapped_user = mapped_text_info["user_id"]

                    prediction_text = line["response"]["body"]["choices"][0]["message"]["content"]
                    relevance, opinion = self.process_prediction_text(prediction_text)
                    if relevance == 1:
                        user_opinions[mapped_user].append(opinion)
                    
                    opinion_full_result[mapped_text_info["text_id"]] = {
                        "user_id": mapped_user,
                        "opinion": opinion,
                        "relevance": relevance,
                        "timestamp": mapped_text_info["timestamp"],
                        "is_retweet": mapped_text_info["is_retweet"],
                        "zhuan": mapped_text_info["zhuan"],
                        "ping": mapped_text_info["ping"],
                        "zan": mapped_text_info["zan"],
                        "content": mapped_text_info["content"]
                    }
        
        # 针对每个user计算平均opinion
        user_avg_opinion = {}
        for user, opinions in user_opinions.items():
            user_avg_opinion[user] = np.mean(opinions)
        
        # 分别讲user opinions 和 user avg opinion存到文件夹
        with open(os.path.join(self.topic_working_dir, "user_opinions.json"), "w") as f:
            json.dump(user_opinions, f)
        
        with open(os.path.join(self.topic_working_dir, "user_avg_opinion.json"), "w") as f:
            json.dump(user_avg_opinion, f)
        
        with open(os.path.join(self.topic_working_dir, "opinion_full_result.json"), "w") as f:
            json.dump(opinion_full_result, f)


class PopularTopic(BatchQwen):
    def __init__(self, topic_id):
        super().__init__(topic_id, "popular_batch")
        self.load_statement_and_opinion(topic_id)

    def load_statement_and_opinion(self, topic_id):
        opinion_df = pd.read_csv(WEIBO_TOPICS_CSV)
        topic_df = opinion_df[opinion_df["topic"] == topic_id]
        if topic_df.shape[0] == 0:
            raise ValueError(f"No content found for the topic {topic_id}. Please check the input topic id.")
        self.statement = topic_df["description"].iloc[0]
        self.opinions = topic_df.iloc[0, 2:7]
    
    def prompt_composer(self, topic_id):
        """
        有一个csv new_topic_opinion.csv
        topic	description	opinion-1	opinion-2	opinion-3	opinion-4	opinion-5
        其中topic是topic_id，description是statement，opinion-1到opinion-5是5个不同的opinion（-2, -1, 0, 1, 2）
        先选中topic为topic_id的行，然后拼接成prompt
        """
        statement = self.statement
        opinions = self.opinions
        # print(statement, opinions)
        # raise ValueError("stop here")
        prompt = f"""你是一个擅长处理微博数据的助手。我将给你提供一段关于{topic_id}的微博文本，请帮我判断这段微博文本中{statement}。态度一共有五种可能:
        -2: {opinions[0]}
        -1: {opinions[1]}
        0: {opinions[2]}
        1: {opinions[3]}
        2: {opinions[4]}
        如果这条文本的确与这个话题相关，请从上述五种可能中选择最接近的态度，并直接给我返回对应的数字。
        如果这条文本与这个话题无关，请直接给我返回“无关”。
        请不要输出任何其他内容！\n\n"""

        return prompt

    def load_content(self):
        """
        我的文本存储在new_topic_popular_data/{topic_id}_popular.parquet中
        weibo_id列和text列
        我希望返回一个list，单个元素是一个tuple，第一个元素是id，第二个是text
        """
        texts = []
        df = pd.read_parquet(os.path.join(WEIBO_WORK_DIR, "new_topic_popular_data", f"{self.topic_id}_popular.parquet"))
        for index, row in df.iterrows():
            texts.append((row["weibo_id"], row["text"]))
        return texts

    def process_prediction_text(self, text):
        text = text.split("\n")[0]
        if "无关" in text:
            return 0, None
        try:
            return 1, int(text)
        except ValueError:
            print(f"Error processing text: {text}")
            return 0, None

    def analyze_user_opinion(self):
        """
        分析用户的意见
        存储原始数据：user_opinions.json
        key：text id
        value：user id，opinion，relevance，timestamp
        """
        topic_popular_opinions = []
        
        for batch_key in self.configs["batches"]:
            with jsonlines.open(os.path.join(self.topic_working_dir, f"{batch_key}-output.jsonl"), "r") as f:
                for line in f:

                    prediction_text = line["response"]["body"]["choices"][0]["message"]["content"]
                    relevance, opinion = self.process_prediction_text(prediction_text)
                    if relevance == 1:
                        topic_popular_opinions.append(opinion)
                    
        # 绘图, 绘制这个topic的opinion分布图，存储为pdf（直方图）
        plt.figure(figsize=(10, 6))
        sns.histplot(list(topic_popular_opinions), bins=5, discrete=True)
        plt.xlabel("Opinion")
        plt.ylabel("Count")
        # 添加一个图例，说明x的不同值是什么含义 self.opinions的五个元素分别是-2到2 的含义
        plt.xticks(ticks=[-2, -1, 0, 1, 2], labels=self.opinions)

        plt.title(f"{self.statement} (Popular Weibo Text)")
        plt.savefig(os.path.join(self.topic_working_dir, "opinion_distribution.pdf"), format="pdf")


class LLMTopicLabelling(BatchQwen):
    def __init__(self, topic_id, task_name="deepseek", force_analysis=False):
        super().__init__(topic_id, f"labelling_batch_{task_name}", task_name, force_analysis=force_analysis)
        self.load_statement_and_opinion(topic_id)

    def load_statement_and_opinion(self, topic_id):
        opinion_df = pd.read_csv(WEIBO_TOPICS_CSV)
        topic_df = opinion_df[opinion_df["topic"] == topic_id]
        if topic_df.shape[0] == 0:
            raise ValueError(f"No example found for the topic {topic_id}. Please check the input topic id.")
        self.statement = topic_df["description"].iloc[0]
        self.opinions = topic_df.iloc[0, 2:7]
    
    def prompt_composer(self, topic_id):
        """
        有一个csv new_topic_opinion.csv
        topic	description	opinion-1	opinion-2	opinion-3	opinion-4	opinion-5
        其中topic是topic_id，description是statement，opinion-1到opinion-5是5个不同的opinion（-2, -1, 0, 1, 2）
        先选中topic为topic_id的行，然后拼接成prompt
        """
        statement = self.statement
        opinions = self.opinions
        # print(statement, opinions)
        # raise ValueError("stop here")
        prompt = f"""你是一个擅长处理微博数据的助手。我将给你提供一段关于{topic_id}的微博文本，请帮我判断这段微博文本中{statement}。态度一共有五种可能:
        -2: {opinions[0]}
        -1: {opinions[1]}
        0: {opinions[2]}
        1: {opinions[3]}
        2: {opinions[4]}
        如果这条文本的确与这个话题相关，请从上述五种可能中选择最接近的态度，并直接给我返回对应的数字。
        如果这条文本与这个话题无关，请直接给我返回“无关”。
        请不要输出任何其他内容！\n\n"""

        return prompt

    def load_content(self, return_type="tuple"):
        """
        我的文本存储在new_topic_popular_data/{topic_id}_popular.parquet中
        weibo_id列和text列
        我希望返回一个list，单个元素是一个tuple，第一个元素是id，第二个是text
        """
        texts = []
        df = pd.read_parquet(os.path.join(WEIBO_WORK_DIR, "new_topic_llm_labelling", f"{self.topic_id}_llm.parquet"))
        for index, row in df.iterrows():
            texts.append((row["weibo_id"], row["text"]))
        if return_type == "tuple":
            return texts
        return df

    def process_prediction_text(self, text):
        text = text.split("\n")[0]
        if "无关" in text:
            return 0, None, True
        try:
            return 1, int(text), True
        except ValueError:
            # 尝试从文本中提取数字
            try:
                text = re.findall(r"\d+", text)[0]
                if f"-{text}" in text:
                    return 1, -int(text), True
                return 1, int(text), True
            except:
                pass
            if "中立" in text:
                return 1, 0, True
            if "答案为" in text or "答案是" in text:
                text = text.split("答案")[1]
            if "返回" in text:
                text = text.split("返回")[1]
            if "数字" in text:
                text = text.split("数字")[1]
            if "态度" in text:
                text = text.split("态度")[1]
            if "选择" in text:
                text = text.split("选择")[1]
            if "选" in text:
                text = text.split("选")[1]
            if "是" in text:
                text = text.split("是")[1]
            if "为" in text:
                text = text.split("为")[1]
            text = text.strip("为是。：")
            text = text.replace(" ", "")
            text = text.replace("。", "")
            try:
                return 1, int(text), True
            except ValueError:
                pass
            # 提取“”中间的内容
            reresult = re.findall(r"“(.+?)”", text)
            if len(reresult) > 0:
                text = reresult[0]

                try:
                    return 1, int(text[0]), True
                except:
                    pass
            
            print(f"Error processing text: {text}")
            return 0, None, False

    def analyze_user_opinion(self):
        """
        分析用户的意见
        存储原始数据：user_opinions.json
        key：text id
        value：user id，opinion，relevance，timestamp
        """
        result_map = {}
        processed_list = []
        relevance_list = []
        
        for batch_key in self.configs["batches"]:
            with jsonlines.open(os.path.join(self.topic_working_dir, f"{batch_key}-output.jsonl"), "r") as f:
                for line in f:
                    if "response" not in line:
                        prediction_text = line["choices"][0]["message"]["content"]
                    else:
                        prediction_text = line["response"]["body"]["choices"][0]["message"]["content"]
                    relevance, opinion, processed = self.process_prediction_text(prediction_text)
                    if processed:
                        processed_list.append(processed)
                    if relevance:
                        relevance_list.append(relevance)
                    weibo_id = line["custom_id"].split("-")[1]
                    result_map[weibo_id] = opinion
                    
        original_df = self.load_content(return_type="df")
        original_df = original_df[original_df["weibo_id"].isin(result_map.keys())]
        original_df["opinion"] = original_df["weibo_id"].map(result_map)
        original_df.to_parquet(os.path.join(self.topic_working_dir, f"{self.topic_id}_llm_result.parquet"), engine="fastparquet")

        opinion_list = list(result_map.values())
        with open("result.txt", "a") as f:
            f.write(f"{self.topic_id},50000,{len(result_map)},{len(processed_list)},{len(relevance_list)},{opinion_list.count(-2)},{opinion_list.count(-1)},{opinion_list.count(0)},{opinion_list.count(1)},{opinion_list.count(2)}\n")

class LLMKeywordLabelling(BatchQwen):
    def __init__(self, topic_id, task_name="deepseek", force_analysis=False):
        super().__init__(topic_id, f"keyword_batch_{task_name}", task_name, force_analysis=force_analysis)
        self.load_statement_and_opinion(topic_id)

    def load_statement_and_opinion(self, topic_id):
        opinion_df = pd.read_csv(WEIBO_TOPICS_CSV)
        topic_df = opinion_df[opinion_df["topic_id"] == int(topic_id)]
        if topic_df.shape[0] == 0:
            raise ValueError(f"No example found for the topic {topic_id}. Please check the input topic id.")
        self.statement = topic_df["description"].iloc[0]
        self.opinions = topic_df.iloc[0, 2:7]
    
    def prompt_composer(self, topic_id):
        """
        有一个csv new_topic_opinion.csv
        topic	description	opinion-1	opinion-2	opinion-3	opinion-4	opinion-5
        其中topic是topic_id，description是statement，opinion-1到opinion-5是5个不同的opinion（-2, -1, 0, 1, 2）
        先选中topic为topic_id的行，然后拼接成prompt
        """
        statement = self.statement
        opinions = self.opinions
        # print(statement, opinions)
        # raise ValueError("stop here")
        prompt = f"""你是一个擅长处理微博数据的助手。我将给你提供一段关于{topic_id}的微博文本，请帮我判断这段微博文本中{statement}。态度一共有五种可能:
        -2: {opinions[0]}
        -1: {opinions[1]}
        0: {opinions[2]}
        1: {opinions[3]}
        2: {opinions[4]}
        如果这条文本的确与这个话题相关，请从上述五种可能中选择最接近的态度，并直接给我返回对应的数字。
        如果这条文本与这个话题无关，请直接给我返回“无关”。
        请不要输出任何其他内容！\n\n"""

        return prompt

    def load_content(self, return_type="tuple"):
        texts = []
        df = pd.read_parquet(os.path.join(WEIBO_WORK_DIR, "topic_keyword_data_sample", f"{self.topic_id}.parquet"))
        for index, row in df.iterrows():
            texts.append((row["weibo_id"], row["weibo_content"]))
        if return_type == "tuple":
            return texts
        return df

    def process_prediction_text(self, text):
        text = text.split("\n")[0]
        if "无关" in text:
            return 0, None, True
        try:
            return 1, int(text), True
        except ValueError:
            # 尝试从文本中提取数字
            try:
                text = re.findall(r"\d+", text)[0]
                if f"-{text}" in text:
                    return 1, -int(text), True
                return 1, int(text), True
            except:
                pass
            if "中立" in text:
                return 1, 0, True
            if "答案为" in text or "答案是" in text:
                text = text.split("答案")[1]
            if "返回" in text:
                text = text.split("返回")[1]
            if "数字" in text:
                text = text.split("数字")[1]
            if "态度" in text:
                text = text.split("态度")[1]
            if "选择" in text:
                text = text.split("选择")[1]
            if "选" in text:
                text = text.split("选")[1]
            if "是" in text:
                text = text.split("是")[1]
            if "为" in text:
                text = text.split("为")[1]
            text = text.strip("为是。：")
            text = text.replace(" ", "")
            text = text.replace("。", "")
            try:
                return 1, int(text), True
            except ValueError:
                pass
            # 提取“”中间的内容
            reresult = re.findall(r"“(.+?)”", text)
            if len(reresult) > 0:
                text = reresult[0]

                try:
                    return 1, int(text[0]), True
                except:
                    pass
            
            print(f"Error processing text: {text}")
            return 0, None, False

    def analyze_user_opinion(self):
        """
        分析用户的意见
        存储原始数据：user_opinions.json
        key：text id
        value：user id，opinion，relevance，timestamp
        """
        result_map = {}
        processed_list = []
        relevance_list = []
        
        for batch_key in self.configs["batches"]:
            with jsonlines.open(os.path.join(self.topic_working_dir, f"{batch_key}-output.jsonl"), "r") as f:
                for line in f:
                    if "response" not in line:
                        prediction_text = line["choices"][0]["message"]["content"]
                    else:
                        # print(line["response"]["body"]["choices"][0]["message"])
                        message = prediction_text = line["response"]["body"]["choices"][0]["message"]
                        if "content" not in message:
                            continue
                        prediction_text = message["content"]
                    relevance, opinion, processed = self.process_prediction_text(prediction_text)
                    if processed:
                        processed_list.append(processed)
                    if relevance:
                        relevance_list.append(relevance)
                    weibo_id = line["custom_id"].split("-")[1]
                    result_map[weibo_id] = opinion
                    
        original_df = self.load_content(return_type="df")
        original_df = original_df[original_df["weibo_id"].isin(result_map.keys())]
        original_df["opinion"] = original_df["weibo_id"].map(result_map)
        original_df.to_parquet(os.path.join(self.topic_working_dir, f"{self.topic_id}_llm_result.parquet"), engine="fastparquet")

        opinion_list = list(result_map.values())
        with open("result.txt", "a") as f:
            f.write(f"{self.topic_id},50000,{len(result_map)},{len(processed_list)},{len(relevance_list)},{opinion_list.count(-2)},{opinion_list.count(-1)},{opinion_list.count(0)},{opinion_list.count(1)},{opinion_list.count(2)}\n")




def merge_llm_labelling(topic_id, models=["doubao", "deepseek"]):
    """
    合并doubao和deepseek的标注结果
    """
    keyword_file_prefix = os.path.join(WEIBO_WORK_DIR, "keyword_batch")

    all_model_labellings = []

    for model in models:
        model_labelling = pd.read_parquet(os.path.join(f"{keyword_file_prefix}_{model}", f"topic-{topic_id}", f"{topic_id}_llm_result.parquet"), engine="fastparquet")

        # model_labelling = model_labelling[["weibo_id", "opinion"]]
        model_labelling.rename(columns={"opinion": f"{model}_opinion"}, inplace=True)
        model_labelling.fillna(-99, inplace=True)
        model_labelling = model_labelling[model_labelling[f"{model}_opinion"].isin([-2, -1, 0, 1, 2, -99])]

        all_model_labellings.append(model_labelling)
    

    # merge to one based on weibo_id
    merged_labelling = all_model_labellings[0]

    index = 1
    for model_labelling in all_model_labellings[1:]:
        model_labelling = model_labelling[["weibo_id", f"{models[index]}_opinion"]]
        merged_labelling = merged_labelling.merge(model_labelling, on="weibo_id", how="outer")
        index += 1
    
    opinion_columns = [f"{model}_opinion" for model in models]

    # 定义一个函数，统计每行中出现最多的相同值的数量和对应的值
    def get_agreement_value(row):
        value_counts = row.value_counts()  # 统计每个值的出现次数
        max_count = value_counts.max()  # 找到出现次数最多的值
        if max_count > 1:  # 如果有一致性（即至少有两个相同的值）
            return value_counts.idxmax()  # 返回出现次数最多的值
        else:
            return None  # 如果没有一致性，返回 None

    # 应用到每一行，统计最大一致性数量和对应的值
    merged_labelling["agreement_count"] = merged_labelling[opinion_columns].apply(lambda row: row.value_counts().max(), axis=1)
    merged_labelling["agreement_value"] = merged_labelling[opinion_columns].apply(get_agreement_value, axis=1)

    stat_content = [topic_id, merged_labelling.shape[0]]
    for labelling in all_model_labellings:
        stat_content.append(labelling.shape[0])
    

    lge2_ratio = merged_labelling[merged_labelling["agreement_count"] >= 2].shape[0] / merged_labelling.shape[0]
    lge3_ratio = merged_labelling[merged_labelling["agreement_count"] >= 3].shape[0] / merged_labelling.shape[0]

    
    stat_content.append(lge2_ratio)
    stat_content.append(lge3_ratio)

    for value in [-2, -1, 0, 1, 2, -99]:
        stat_content.append(merged_labelling[merged_labelling["agreement_value"] == value].shape[0])

    with open("result.txt", "a") as f:
        f.write(",".join([str(content) for content in stat_content]))
        f.write("\n")
    
    merged_labelling.to_parquet(os.path.join(BERT_DATASET_DIR, f"{topic_id}_merged.parquet"), engine="fastparquet")

def sample_llm_labelling(topic_id, models=["openai", "qwen", "deepseek"]):
    """
    采样llm标注结果
    """
    merged_labelling = pd.read_parquet(
        os.path.join(BERT_DATASET_DIR, f"{topic_id}_merged.parquet")
    )

    sample_size = 200

    # 在agreement value中筛选200条，1. 抽取40条irrelevant
    irrelevant_size = 40
    relevant_sample = merged_labelling[merged_labelling["agreement_value"] != -99]
    relevant_sample = relevant_sample[relevant_sample["agreement_value"].isin([-2, -1, 0, 1, 2])]
    if relevant_sample.shape[0] < sample_size - irrelevant_size:
        irrelevant_size = sample_size - relevant_sample.shape[0]
    irrelevant_sample = merged_labelling[merged_labelling["agreement_value"] == -99].sample(n=irrelevant_size)

    # 2. 剩余的尽可能-2， -1， 0， 1， 2都有，但是又大致分层

    # 分层抽样，-2， -1， 0， 1， 2 各按自己的占比抽取，最少10条（除非该标签不足10条）
    value_sample_list = []
    remained_sample_size = sample_size - irrelevant_sample.shape[0]
    for value in [-2, -1, 0, 1, 2]:
        value_count = relevant_sample[relevant_sample["agreement_value"] == value].shape[0]
        value_sample_size = int(remained_sample_size * value_count / relevant_sample.shape[0])
        print(value, value_count, value_sample_size, relevant_sample.shape)
        if value_count < 10:
            value_sample = relevant_sample[relevant_sample["agreement_value"] == value]
        else:
            if value_sample_size < 10:
                value_sample = relevant_sample[relevant_sample["agreement_value"] == value].sample(n=10)
            else:
                value_sample = relevant_sample[relevant_sample["agreement_value"] == value].sample(n=value_sample_size)
        value_sample_list.append(value_sample)

    # 3. 合并
    sample = pd.concat([irrelevant_sample] + value_sample_list)
    return sample

def merge_sampling():
    all_samples = []
    for topic in range(17):
        if topic in [3, 8]:
            continue
        topic_id = str(topic)
        sample = sample_llm_labelling(topic_id)
        sample["topic_id"] = topic_id
        # sample["topic"] = labelling_folder_map[topic]
        # sample 乱序
        sample = sample.sample(frac=1)
        all_samples.append(sample)

    # 存储为excel，需要增加topic_id列，topic列
    all_samples = pd.concat(all_samples)
    all_samples.to_excel("all_samples.xlsx")


WEIBO_TOPICS = [str(t) for t in range(17) if t not in (3, 8)]


def label(topic_id, model="deepseek"):
    """LLM opinion labels for the 10,000-post sample of one topic (batch API).

    Re-run to advance the batch job: generate -> submit -> retrieve -> analyse.
    model: openai (gpt-4o-mini) / qwen / deepseek / doubao.
    """
    LLMKeywordLabelling(str(topic_id), task_name=model).process()


def merge(topic_id=None, models=("openai", "qwen", "deepseek")):
    """Merge the models' labels; agreement_value is the majority label.

    Output: <BERT_DATASET_DIR>/<topic>_merged.parquet (BERT training data).
    """
    for t in [topic_id] if topic_id is not None else WEIBO_TOPICS:
        merge_llm_labelling(str(t), list(models))


if __name__ == "__main__":
    import fire

    fire.Fire({
        "label": label,
        "merge": merge,
        "sample": sample_llm_labelling,
        "merge_sampling": merge_sampling,
    })
