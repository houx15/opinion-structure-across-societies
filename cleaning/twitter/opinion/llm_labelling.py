import os
import json
import re

import jsonlines

import pandas as pd
import numpy as np

from openai import OpenAI
import time

from cleaning.settings import *

"""
配置工作路径
"""
SOURCE_DIR = os.path.join(ORIGINAL_TWEETS_DIR, "labeling_sample")


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
            # if self.topic_id in ["test", "test-5label"]:
            #     self.batch_file_preparation_for_test()
            # else:
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
            configs = {"batches": [], "batch_info": {}, "status": "init"}

        self.configs = configs

    def save_configs(self):
        with open(os.path.join(self.topic_working_dir, f"configs.json"), "w") as f:
            json.dump(self.configs, f)

    def save_single_batch(self, batch_key, data):
        # 存储为jsonlines
        with jsonlines.open(
            os.path.join(self.topic_working_dir, f"{batch_key}.jsonl"), "w"
        ) as f:
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
                        {"role": "user", "content": f"Text: {weibo_content}\nAnswer:"},
                    ],
                    "max_tokens": 2000,
                },
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
        output_file_path = os.path.join(
            self.topic_working_dir, f"{batch_file}-output.jsonl"
        )
        self.load_processed_ids(output_file_path)

        with open("templog.txt", "a") as f:
            f.write(
                f"{self.topic_working_dir}, batch_file: {batch_file}, processed_ids: {len(self.processed_ids)}\n"
            )

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
                        file=open(
                            os.path.join(self.topic_working_dir, f"{batch_key}.jsonl"),
                            "rb",
                        ),
                        purpose="batch",
                    )
                    batch_input_file_id = batch_input_file.id

                    self.configs["batch_info"][batch_key][
                        "input_file_id"
                    ] = batch_input_file_id
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
                if not os.path.exists(
                    os.path.join(self.topic_working_dir, f"{batch_key}-output.jsonl")
                ):
                    print(f"Batch {batch_key} not retrieved.")
                    all_retrieved = False
        else:
            for batch_key in self.configs["batches"]:
                if self.configs["batch_info"][batch_key]["status"] == "submitted":
                    batch_info = self.client.batches.retrieve(
                        self.configs["batch_info"][batch_key]["task_id"]
                    )
                    print(batch_info)
                    if (
                        batch_info.status == "completed"
                        or batch_info.status == "expired"
                    ):
                        output_file_id = batch_info.output_file_id
                        error_file_id = batch_info.error_file_id
                        if output_file_id is not None:
                            output_file_content = self.client.files.content(
                                output_file_id
                            ).text
                            with open(
                                os.path.join(
                                    self.topic_working_dir, f"{batch_key}-output.jsonl"
                                ),
                                "w",
                            ) as f:
                                f.write(output_file_content)

                        if error_file_id is not None:
                            error_file_content = self.client.files.content(
                                error_file_id
                            ).text
                            with open(
                                os.path.join(
                                    self.topic_working_dir, f"{batch_key}-error.jsonl"
                                ),
                                "w",
                            ) as f:
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
        raise NotImplementedError(
            "Please implement the process_prediction_text method."
        )

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


class LLMKeywordLabelling(BatchQwen):
    def __init__(self, topic_id, task_name="deepseek", force_analysis=False):
        super().__init__(
            topic_id,
            f"keyword_batch_{task_name}",
            task_name,
            force_analysis=force_analysis,
        )
        self.load_statement_and_opinion(topic_id)

    def load_statement_and_opinion(self, topic_id):
        opinion_df = pd.read_csv(TWITTER_OPINION_CONFIG)
        topic_df = opinion_df[opinion_df["topic_id"] == topic_id]
        if topic_df.shape[0] == 0:
            raise ValueError(
                f"No example found for the topic {topic_id}. Please check the input topic id."
            )
        self.statement = topic_df["description"].iloc[0]
        self.opinions = topic_df.iloc[0, 4:9].tolist()
        self.detailed_opinions = topic_df.iloc[1, 4:9].tolist()
        # print(self.statement, self.opinions)
        # print(self.detailed_opinions)

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
        prompt = f"""You are an assistant skilled at identifying hidden opinions in social media texts. Please analyze the following tweet on "{statement}" and directly provide an answer from the list below: 2, 1, 0, -1, -2, skip.
The numbers correspond to the following tags:
2: {opinions[4]}. {self.detailed_opinions[4]}
1: {opinions[3]}. {self.detailed_opinions[3]}
0: {opinions[2]}. {self.detailed_opinions[2]}
-1: {opinions[1]}. {self.detailed_opinions[1]}
-2: {opinions[0]}. {self.detailed_opinions[0]}
skip: irrelevant or no opinion on {statement}
Make sure your answer is from the list above. Provide ONLY the answer directly, no verbosity!\n\n"""

        return prompt

    def load_content(self, return_type="tuple"):
        texts = []
        df = pd.read_parquet(
            f"{SOURCE_DIR}/{labelling_folder_map[self.topic_id]}.parquet"
        )
        for index, row in df.iterrows():
            texts.append((row.name, row["text"]))
        if return_type == "tuple":
            return texts
        return df

    def process_prediction_text(self, text):
        text = text.split("\n")[0]
        if "skip" in text:
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

            try:
                return 1, int(text), True
            except ValueError:
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
        print("analyzing")
        result_map = {}
        processed_list = []
        relevance_list = []

        for batch_key in self.configs["batches"]:
            with jsonlines.open(
                os.path.join(self.topic_working_dir, f"{batch_key}-output.jsonl"), "r"
            ) as f:
                for line in f:
                    if "response" not in line:
                        prediction_text = line["choices"][0]["message"]["content"]
                    else:
                        # print(line["response"]["body"]["choices"][0]["message"])
                        message = prediction_text = line["response"]["body"]["choices"][
                            0
                        ]["message"]
                        if "content" not in message:
                            continue
                        prediction_text = message["content"]
                    relevance, opinion, processed = self.process_prediction_text(
                        prediction_text
                    )
                    if processed:
                        processed_list.append(processed)
                    if relevance:
                        relevance_list.append(relevance)
                    tweet_id = line["custom_id"].split("-")[1]
                    # tweet_id = int(tweet_id)
                    result_map[tweet_id] = opinion

        original_df = self.load_content(return_type="df")
        original_length = len(original_df)
        original_df = original_df[original_df.index.isin(result_map.keys())]
        # print(original_df)
        original_df["opinion"] = original_df.index.map(result_map)
        original_df.to_parquet(
            os.path.join(self.topic_working_dir, f"{self.topic_id}_llm_result.parquet"),
            engine="fastparquet",
        )

        opinion_list = list(result_map.values())
        with open("result.txt", "a") as f:
            f.write(
                f"{self.topic_id},{self.task_name},{original_length},{len(result_map)},{len(processed_list)},{len(relevance_list)},{opinion_list.count(-2)},{opinion_list.count(-1)},{opinion_list.count(0)},{opinion_list.count(1)},{opinion_list.count(2)}\n"
            )


def merge_llm_labelling(topic_id, models=["doubao", "deepseek"]):
    """
    合并doubao和deepseek的标注结果
    """
    keyword_file_prefix = "keyword_batch"

    all_model_labellings = []

    for model in models:
        model_labelling = pd.read_parquet(
            os.path.join(
                f"{keyword_file_prefix}_{model}",
                f"topic-{topic_id}",
                f"{topic_id}_llm_result.parquet",
            ),
            engine="fastparquet",
        )

        # model_labelling = model_labelling[["weibo_id", "opinion"]]
        model_labelling.rename(columns={"opinion": f"{model}_opinion"}, inplace=True)
        model_labelling.fillna(-99, inplace=True)
        model_labelling = model_labelling[
            model_labelling[f"{model}_opinion"].isin([-2, -1, 0, 1, 2, -99])
        ]

        all_model_labellings.append(model_labelling)

    # merge to one based on weibo_id
    merged_labelling = all_model_labellings[0]

    index = 1
    for model_labelling in all_model_labellings[1:]:
        model_labelling = model_labelling[[f"{models[index]}_opinion"]]
        merged_labelling = merged_labelling.merge(
            model_labelling, left_index=True, right_index=True, how="outer"
        )
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
    merged_labelling["agreement_count"] = merged_labelling[opinion_columns].apply(
        lambda row: row.value_counts().max(), axis=1
    )
    merged_labelling["agreement_value"] = merged_labelling[opinion_columns].apply(
        get_agreement_value, axis=1
    )

    stat_content = [topic_id, merged_labelling.shape[0]]
    for labelling in all_model_labellings:
        stat_content.append(labelling.shape[0])

    lge2_ratio = (
        merged_labelling[merged_labelling["agreement_count"] >= 2].shape[0]
        / merged_labelling.shape[0]
    )
    lge3_ratio = (
        merged_labelling[merged_labelling["agreement_count"] >= 3].shape[0]
        / merged_labelling.shape[0]
    )

    stat_content.append(lge2_ratio)
    stat_content.append(lge3_ratio)

    for value in [-2, -1, 0, 1, 2, -99]:
        stat_content.append(
            merged_labelling[merged_labelling["agreement_value"] == value].shape[0]
        )

    with open("result.txt", "a") as f:
        f.write(",".join([str(content) for content in stat_content]))
        f.write("\n")

    os.makedirs(os.path.join(ORIGINAL_TWEETS_DIR, "training_data"), exist_ok=True)
    output_path = os.path.join(
        ORIGINAL_TWEETS_DIR, "training_data", f"{topic_id}_merged.parquet"
    )

    merged_labelling.to_parquet(output_path, engine="fastparquet")


def sample_llm_labelling(topic_id, models=["openai", "qwen", "deepseek"]):
    """
    采样llm标注结果
    """
    merged_labelling = pd.read_parquet(
        os.path.join(ORIGINAL_TWEETS_DIR, "training_data", f"{topic_id}_merged.parquet")
    )

    sample_size = 200

    # 在agreement value中筛选200条，1. 抽取40条irrelevant
    irrelevant_size = 40
    relevant_sample = merged_labelling[merged_labelling["agreement_value"] != -99]
    relevant_sample = relevant_sample[
        relevant_sample["agreement_value"].isin([-2, -1, 0, 1, 2])
    ]
    if relevant_sample.shape[0] < sample_size - irrelevant_size:
        irrelevant_size = sample_size - relevant_sample.shape[0]
    irrelevant_sample = merged_labelling[
        merged_labelling["agreement_value"] == -99
    ].sample(n=irrelevant_size)

    # 2. 剩余的尽可能-2， -1， 0， 1， 2都有，但是又大致分层

    # 分层抽样，-2， -1， 0， 1， 2 各按自己的占比抽取，最少10条（除非该标签不足10条）
    value_sample_list = []
    remained_sample_size = sample_size - irrelevant_sample.shape[0]
    for value in [-2, -1, 0, 1, 2]:
        value_count = relevant_sample[
            relevant_sample["agreement_value"] == value
        ].shape[0]
        value_sample_size = int(
            remained_sample_size * value_count / relevant_sample.shape[0]
        )
        print(value, value_count, value_sample_size, relevant_sample.shape)
        if value_count < 10:
            value_sample = relevant_sample[relevant_sample["agreement_value"] == value]
        else:
            if value_sample_size < 10:
                value_sample = relevant_sample[
                    relevant_sample["agreement_value"] == value
                ].sample(n=10)
            else:
                value_sample = relevant_sample[
                    relevant_sample["agreement_value"] == value
                ].sample(n=value_sample_size)
        value_sample_list.append(value_sample)

    # 3. 合并
    sample = pd.concat([irrelevant_sample] + value_sample_list)
    sample.to_parquet(
        os.path.join(ORIGINAL_TWEETS_DIR, "training_data", f"{topic_id}_sample.parquet")
    )
    return sample


def merge_sampling():
    all_samples = []
    for topic in labelling_folder_map.keys():
        sample = sample_llm_labelling(topic)
        sample["topic_id"] = topic
        sample["topic"] = labelling_folder_map[topic]
        # sample 乱序
        sample = sample.sample(frac=1)
        all_samples.append(sample)

    # 存储为excel，需要增加topic_id列，topic列
    all_samples = pd.concat(all_samples)
    all_samples.to_excel("all_samples.xlsx")


def label(topic_id=None, model="deepseek"):
    """LLM opinion labels for the 10,000-tweet sample of one topic (all topics if omitted).

    Re-run to advance the batch job: generate -> submit -> retrieve -> analyse.
    model: openai (gpt-4o-mini) / qwen (qwq-plus, a reasoning model) / deepseek / doubao.
    """
    for t in [topic_id] if topic_id is not None else list(labelling_folder_map):
        LLMKeywordLabelling(str(t), task_name=model).process()


def merge(topic_id=None, models=("openai", "qwen", "deepseek")):
    """Merge the models' labels -> <ORIGINAL_TWEETS_DIR>/training_data/<topic>_merged.parquet."""
    for t in [topic_id] if topic_id is not None else list(labelling_folder_map):
        merge_llm_labelling(str(t), list(models))


if __name__ == "__main__":
    import fire

    fire.Fire({
        "label": label,
        "merge": merge,
        "sample": sample_llm_labelling,
        "merge_sampling": merge_sampling,
    })
