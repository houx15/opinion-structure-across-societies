import os
import json
import re

import jsonlines

import pandas as pd
import numpy as np

from openai import OpenAI
import time

from cleaning.settings import *


class LLMBatch:
    def __init__(self, working_dir, model="gpt-5-mini", mode="test"):
        self.working_dir = working_dir
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            # organization=OPENAI_ORGANIZATION,
            # project=OPENAI_PROJECT,
        )
        self.model = model
        self.mode = mode

        self.load_configs()

    def process(self):
        """
        根据configs里面的status决定下一步行动
        """
        if self.configs["status"] == "init":
            data = self.load_content()
            self.batch_file_preparation(data)
            self.submit_batch()
        elif self.configs["status"] == "generated":
            # 已经生成数据，但没有提交，未报错的情况下不应该出现这种情况
            self.submit_batch()
        elif self.configs["status"] == "submitted":
            # 已经提交，但未取回数据
            self.retrieve_batch()
        elif self.configs["status"] == "retrieved":
            # 已经取回数据，但未处理
            self.analyze()

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
            with open(os.path.join(self.working_dir, f"configs.json"), "r") as f:
                configs = json.load(f)
        except FileNotFoundError:
            configs = {"batches": [], "batch_info": {}, "status": "init"}

        self.configs = configs

    def save_configs(self):
        with open(os.path.join(self.working_dir, f"configs.json"), "w") as f:
            json.dump(self.configs, f)

    def save_single_batch(self, batch_key, data):
        # 存储为jsonlines
        with jsonlines.open(
            os.path.join(self.working_dir, f"{batch_key}.jsonl"), "w"
        ) as f:
            f.write_all(data)
        self.configs["batches"].append(batch_key)
        self.configs["batch_info"][batch_key] = {"task_id": None, "status": "generated"}
        self.save_configs()

    def prompt_composer(
        self,
    ):
        return """You are a precise geolocation classifier for European countries.
You will receive a Twitter user's self-filled location text that has been identified as being in Europe.
Your task is to determine which specific European country this location belongs to.

Reply with ONLY the ISO 3166-1 alpha-2 country code (2 uppercase letters) or "unknown" if you cannot determine the country.

Valid European country ISO codes:
AL (Albania), AD (Andorra), AT (Austria), BY (Belarus), BE (Belgium), BA (Bosnia and Herzegovina),
BG (Bulgaria), HR (Croatia), CY (Cyprus), CZ (Czechia), DK (Denmark), EE (Estonia),
FI (Finland), FR (France), DE (Germany), GR (Greece), HU (Hungary), IS (Iceland),
IE (Ireland), IT (Italy), LV (Latvia), LI (Liechtenstein), LT (Lithuania), LU (Luxembourg),
MT (Malta), MD (Moldova), MC (Monaco), ME (Montenegro), NL (Netherlands), MK (North Macedonia),
NO (Norway), PL (Poland), PT (Portugal), RO (Romania), RU (Russia), SM (San Marino),
RS (Serbia), SK (Slovakia), SI (Slovenia), ES (Spain), SE (Sweden), CH (Switzerland),
UA (Ukraine), GB (United Kingdom), VA (Vatican City)

Important instructions:
- Location strings may be standard place names OR highly non-standard entries: misspellings, transliterations, slang, landmarks, emojis, lat/lon coordinates, abbreviations, iconic buildings, or phonetic variants.
- You should try to infer the intended country *only if* there is a plausible, geographically grounded interpretation.
- If the entry does not clearly point to a specific European country, output "unknown".
- For cities, identify the country they belong to (e.g., "Paris" → FR, "London" → GB).

Respond ONLY with the ISO code (2 uppercase letters) or "unknown".
"""

    def load_content(self):
        if os.path.exists(os.path.join(self.working_dir, "original_data.parquet")):
            return pd.read_parquet(
                os.path.join(self.working_dir, "original_data.parquet")
            )
        with open(os.path.join(BASE_DIR, "eu_location_by_country.json"), "r") as f:
            data = json.load(f)

        # data undecided 是一个列表，转化为df，赋值独一无二的id
        df = pd.DataFrame(data["unknown"], columns=["location"])
        df["id"] = df.index
        if self.mode == "test":
            df = df.sample(100)
        df.to_parquet(os.path.join(self.working_dir, "original_data.parquet"))
        return df

    def batch_file_preparation(self, data):
        """
        生成batch file
        """
        prompt = self.prompt_composer()
        item_count = 0
        file_count = 0
        request_list = []
        all_custom_ids = set()

        for index, row in data.iterrows():
            manual_id = row["id"]
            location = row["location"]
            if manual_id in all_custom_ids:
                continue
            single_request = {
                "custom_id": f"{manual_id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"Location: {location}\nAnswer:"},
                    ],
                    # "max_tokens": 120,
                },
            }
            request_list.append(single_request)
            all_custom_ids.add(manual_id)
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
                pid = str(int(data["custom_id"]))
                self.processed_ids.add(pid)
        return

    def write_single_line_to_jsonl(self, line, output_file_path):
        with jsonlines.open(output_file_path, "a") as wfile:
            wfile.write(line)

    def submit_batch(self):
        """
        提交batch
        """
        for batch_key in self.configs["batches"]:
            if self.configs["batch_info"][batch_key]["status"] == "generated":

                batch_input_file = self.client.files.create(
                    file=open(
                        os.path.join(self.working_dir, f"{batch_key}.jsonl"),
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
        for batch_key in self.configs["batches"]:
            if self.configs["batch_info"][batch_key]["status"] == "submitted":
                batch_info = self.client.batches.retrieve(
                    self.configs["batch_info"][batch_key]["task_id"]
                )
                print(batch_info)
                if batch_info.status == "completed" or batch_info.status == "expired":
                    output_file_id = batch_info.output_file_id
                    error_file_id = batch_info.error_file_id
                    if output_file_id is not None:
                        output_file_content = self.client.files.content(
                            output_file_id
                        ).text
                        with open(
                            os.path.join(self.working_dir, f"{batch_key}-output.jsonl"),
                            "w",
                        ) as f:
                            f.write(output_file_content)

                    if error_file_id is not None:
                        error_file_content = self.client.files.content(
                            error_file_id
                        ).text
                        with open(
                            os.path.join(self.working_dir, f"{batch_key}-error.jsonl"),
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
        returns: ISO country code (2 uppercase letters) or "unknown"
        """
        # 移除首尾空白和换行符
        text = text.strip().split("\n")[0].strip().upper()
        
        # 有效的欧洲国家 ISO 代码列表
        valid_iso_codes = {
            "AL", "AD", "AT", "BY", "BE", "BA", "BG", "HR", "CY", "CZ", "DK", "EE",
            "FI", "FR", "DE", "GR", "HU", "IS", "IE", "IT", "LV", "LI", "LT", "LU",
            "MT", "MD", "MC", "ME", "NL", "MK", "NO", "PL", "PT", "RO", "RU", "SM",
            "RS", "SK", "SI", "ES", "SE", "CH", "UA", "GB", "UK", "VA", "GI", "TR", "GE", "AZ", "TR"
        }
        
        # 直接匹配 "unknown"
        if "UNKNOWN" in text or text == "UNK":
            return "unknown"
        
        # 尝试提取 2 个字母的 ISO 代码
        # 先尝试直接匹配
        if text in valid_iso_codes:
            # 将 UK 标准化为 GB
            if text == "UK":
                return "GB"
            if text == "GI":
                return "GB"
            return text
        
        # 尝试从文本中提取 2 个大写字母
        iso_match = re.search(r"\b([A-Z]{2})\b", text)
        if iso_match:
            iso_code = iso_match.group(1)
            if iso_code in valid_iso_codes:
                # 将 UK 标准化为 GB
                if iso_code == "UK":
                    return "GB"
                if iso_code == "GI":
                    return "GB"
                return iso_code
        
        # 尝试提取任何 2 个连续的大写字母
        iso_match = re.search(r"([A-Z]{2})", text)
        if iso_match:
            iso_code = iso_match.group(1)
            if iso_code in valid_iso_codes:
                # 将 UK 标准化为 GB
                if iso_code == "UK":
                    return "GB"
                if iso_code == "GI":
                    return "GB"
                return iso_code
        
        # 如果都找不到，返回 unknown
        print(f"Warning: Could not parse ISO code from text: {text}, returning 'unknown'")
        return "unknown"

    def analyze(self):
        print("analyzing")
        result_map = {}
        processed_list = []

        for batch_key in self.configs["batches"]:
            with jsonlines.open(
                os.path.join(self.working_dir, f"{batch_key}-output.jsonl"), "r"
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
                    result = self.process_prediction_text(prediction_text)
                    manual_id = int(line["custom_id"])
                    result_map[manual_id] = result

        original_df = self.load_content()
        original_df = original_df[original_df.index.isin(result_map.keys())]
        original_df["result"] = original_df.index.map(result_map)
        original_df.to_parquet(
            os.path.join(self.working_dir, f"llm_result.parquet"),
            engine="fastparquet",
        )
        print(original_df)


if __name__ == "__main__":
    batch = LLMBatch(working_dir=os.path.join(BASE_DIR, "llm_location_eu_country"), mode="prod")
    batch.process()
