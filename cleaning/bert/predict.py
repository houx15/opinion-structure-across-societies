import torch
import fire
import time
import glob

import numpy as np
import pandas as pd

from datetime import datetime, timedelta

import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from cleaning.bert.utils import sentence_cleaner

from cleaning.settings import *


class OpinionPredict(object):
    r"""
    Make predictions based on trained models.

    Args:
        task_type (`str`):
            Current task type, 'regression' or 'binary'.
        model_path (`str`):
            Model name or path, e.g. '../model-bert/gun-regression'. The model will be used for tokenization and prediction.
        src_type (`str`, *optional*, defaults to `tweet`):
            Data source. 'tweet' or 'weibo
    """

    def __init__(
        self,
        task_type: str,
        model_path: str,
        src_type: str = "tweet",
        fp16: bool = True,
    ) -> None:
        assert task_type in ["regression", "binary"]
        self.task_type = task_type
        self.src_type = src_type

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False)
        self.model = self.model_init(model_path)

        self.fp16 = fp16

        if self.fp16:
            print(">>>>>>>>>Using fp16.")

    def model_init(self, model_path: str):
        print(">>>>>>>>>initializing model:  {}".format(model_path))

        num_labels_map = {
            "binary": 2,
            "regression": 1,
        }
        torch.cuda.empty_cache()
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels_map[self.task_type],
            output_attentions=True,
            output_hidden_states=False,
        )
        if torch.cuda.is_available():
            model.cuda()
        self.model = model
        # self.model.half()
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False  # freeze the finetuned model. save memory.
        # self.model = torch.compile(
        #     self.model,
        #     mode="max-autotune",
        #     fullgraph=True,
        #     dynamic=False,
        # )
        return self.model

    def predict(
        self,
        texts: list,
        max_length: int = 256,  # None,
        padding: str = "max_length",
        batch: int = 512,
        verbose=print,
    ) -> np.array:
        """
        :param texts: list (or numpy.array, pandas.Series, torch.tensor) of strings.
        :param padding: str. 'longest', 'max_length' (default), 'do_not_pad'.
        :param max_length, batch: int. max length of tweet and number of tweets proceeded in a batch. limited by GPU memory.
            max_length=128 is sufficient for most tweets, and 512 tweeets per batch are recommended for 128-letter tweets on a typical Tesla GPU with 16GB memory.
            :param verbose: function that accepts string input. used to display message.
            e.g., verbose = print (default, display on screen)
            e.g., verbose = logger.info (write to a log file)
            e.g., verbose = lambda message: logger.info(f'filename: {message}') # write to a log file with a header
        :return: 1-dim numpy array

        """

        verbose(
            f"predict(texts={len(texts)}, max_length={max_length}, padding={padding}, batch={batch})"
        )

        torch.cuda.empty_cache()
        try:
            prediction = np.array([])
            for i in range(0, len(texts), batch):
                # encode input texts
                encoding = self.tokenizer(
                    [
                        sentence_cleaner(self.src_type, single_text)
                        for single_text in texts[i : i + batch]
                    ],
                    # texts[i : i + batch],
                    add_special_tokens=True,
                    return_token_type_ids=True,
                    truncation=True,
                    padding=padding,
                    return_attention_mask=True,
                    return_tensors="pt",
                    max_length=max_length,
                )
                if torch.cuda.is_available():
                    # verbose(f'self.model.device={self.model.device}')  # self.model.device=cuda:0
                    for key in encoding.keys():
                        encoding[key] = encoding[key].cuda()
                        # verbose(f'encoding[{key}].device={encoding[key].device}. encoding[{key}].shape={encoding[key].shape}') # encoding[input_ids].device=cuda:0. encoding[input_ids].shape=torch.Size([4, 128])

                # calculate the encoded input with frozen model
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs = self.model(
                        input_ids=encoding["input_ids"],
                        attention_mask=encoding["attention_mask"],
                        token_type_ids=encoding["token_type_ids"],
                    ).logits.detach()
                # verbose(f'type(outputs)={type(outputs)}, outputs.device={outputs.device}') #  type(outputs)=<class 'torch.Tensor'>, outputs.device=cuda:0
                if torch.cuda.is_available():
                    outputs = (
                        outputs.cpu()
                    )  # copy the tensor to host memory before converting it to numpy. otherwise we will get an error "can't convert cuda:0 device type tensor to numpy"
                    # verbose(f'type(outputs)={type(outputs)}, outputs.device={outputs.device}')  # type(outputs)=<class 'torch.Tensor'>, outputs.device=cpu
                del encoding
                verbose(f"outputs.numpy().shape={outputs.numpy().shape}")
                # verbose(f'outputs.numpy()={outputs.numpy()}')

                # output
                if "regression" == self.task_type:
                    prediction = np.append(
                        prediction, outputs.numpy().flatten()
                    )  # a score for each input string
                elif "binary" == self.task_type:
                    prediction = np.append(
                        prediction, np.argmax(outputs.numpy(), axis=1)
                    )  # a float probability of label "1" for each input string
                else:
                    raise ValueError(f"Unknown self.task_type = {self.task_type}.")
                del outputs
                torch.cuda.empty_cache()
                verbose(f"prediction.shape={prediction.shape}")
                # verbose(f'prediction={prediction}')

            # output
            return prediction

        except RuntimeError as error:
            verbose(f"Running out of memory, retrying with a smaller batch.")
            raise RuntimeError(
                "Running out of GPU memory. Try limiting [max_length] and [batch]."
            ) from error

    # predict()


start_date_dict = {
    "0": datetime(2019, 2, 6),
    "1": datetime(2020, 3, 4),
    "4": datetime(2019, 6, 22),
    "11": datetime(2019, 3, 14),
    "14": datetime(2020, 9, 4),
}


def predict_weibo_opinion(
    topic: str,
    task_type: str = "binary",
    fp16: bool = True,
    batch: int = 700,
    year=None,
):
    topic = str(topic)
    start_date = datetime(2016, 1, 1)
    end_date = datetime(2023, 12, 31)

    if task_type == "binary":
        start_date = start_date_dict[topic]
        print(f"Start date: {start_date.strftime('%Y-%m-%d')}")

    date_range = [
        start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)
    ]

    model_path = os.path.join(output_dir_base, f"topic-{topic}", f"run-0{task_type}")

    data_path = os.path.join(WEIBO_TOPIC_KEYWORD_DIR, topic)

    predictor = OpinionPredict(task_type=task_type, model_path=model_path, fp16=fp16)

    year_path = os.path.join(data_path, "year_data")
    binary_year_path = os.path.join(data_path, "binary_year_data")

    if os.path.exists(binary_year_path) and task_type == "binary":
        # Get all parquet files in the directory
        print(f"processing based on year data")
        if year is not None:
            parquet_files = glob.glob(
                os.path.join(binary_year_path, f"{year}-*.parquet")
            )
            print(f"find {len(parquet_files)} files for year {year}")
        else:
            parquet_files = glob.glob(os.path.join(binary_year_path, "*.parquet"))
        # parquet_files = [os.path.join(data_path, "2016-10-01.parquet")]
        for file_path in parquet_files:
            start = time.time()
            df = pd.read_parquet(file_path)
            if "relevance" in df.columns:
                print(f"File {file_path} has 'relevance' column.")
                continue
            texts = df["cleaned_weibo_content"].tolist()
            predictions = predictor.predict(texts, batch=batch)
            print(f"{file_path}, predict time: {time.time() - start}")

            df["relevance"] = predictions
            df.to_parquet(file_path)
    elif os.path.exists(year_path) and task_type == "regression":
        # Get all parquet files in the directory
        if year is not None:
            parquet_files = glob.glob(os.path.join(year_path, f"{year}-*.parquet"))
            print(f"find {len(parquet_files)} files for year {year}")
        else:
            parquet_files = glob.glob(os.path.join(year_path, "*.parquet"))

        for file_path in parquet_files:
            df = pd.read_parquet(file_path)
            if "opinion" in df.columns:
                print(f"File {file_path} has 'opinion' column.")
                continue
            texts = df["cleaned_weibo_content"].tolist()
            predictions = predictor.predict(texts, batch=batch)
            df["opinion"] = predictions
            df.to_parquet(file_path)

    else:
        for date in date_range:
            start_time = time.time()

            date_str = date.strftime("%Y-%m-%d")

            data_file = os.path.join(data_path, f"{date_str}.parquet")
            if not os.path.exists(data_file):
                print(f"File {data_file} does not exist.")
                continue
            df = pd.read_parquet(data_file)

            content_column = (
                "cleaned_weibo_content"
                if "cleaned_weibo_content" in df.columns
                else "weibo_content"
            )

            if task_type == "binary":
                if "relevance" in df.columns:
                    print(f"File {data_file} has 'relevance' column.")
                    continue
                texts = df[content_column].tolist()
                predictions = predictor.predict(texts, batch=batch)
                df["relevance"] = predictions
            else:
                if "opinion" in df.columns:
                    print(f"File {data_file} has 'opinion' column.")
                    continue
                relevant_df = df[df["relevance"] == 1].copy()
                texts = relevant_df[content_column].tolist()
                predictions = predictor.predict(texts, batch=batch)
                relevant_df["opinion"] = predictions
                df.loc[relevant_df.index, "opinion"] = relevant_df["opinion"]
            df.to_parquet(data_file)

            end_time = time.time()
            print(f"Finished processing {date_str} in {end_time - start_time} seconds.")


def predict_twitter_opinion(
    topic: str, task_type: str = "binary", fp16: bool = True, batch: int = 700
):
    topic = str(topic)

    model_path = os.path.join(output_dir_base, f"topic-{topic}", f"run-0-{task_type}")

    data_path = os.path.join(TWITTER_DIR, f"{labelling_folder_map[topic]}-opinion")

    predictor = OpinionPredict(task_type=task_type, model_path=model_path, fp16=fp16)

    text_files = glob.glob(os.path.join(data_path, "*-text.pickle4"))

    for file_path in text_files:
        start = time.time()
        df = pd.read_pickle(file_path)
        if task_type == "binary":
            if "relevance" in df.columns:
                print(f"File {file_path} has 'relevance' column.")
                continue
        if task_type == "regression":
            if "opinion" in df.columns:
                print(f"File {file_path} has 'opinion' column.")
                continue
        texts = df["text"].tolist()
        predictions = predictor.predict(texts, batch=batch)
        print(f"{file_path}, predict time: {time.time() - start}")

        output_column = "opinion" if task_type == "regression" else "relevance"

        df[output_column] = predictions
        df.to_pickle(file_path)


def recover_twitter_opinion(topic: str):
    topic = str(topic)

    data_path = os.path.join(TWITTER_DIR, f"{labelling_folder_map[topic]}-opinion")

    text_files = glob.glob(os.path.join(data_path, "*-text.pickle4"))

    for file_path in text_files:
        df = pd.read_pickle(file_path)
        if "relevance" in df.columns and "opinion" in df.columns:
            opinion_df = df[["author_id", "relevance", "opinion"]].copy()
            output_path = file_path.replace("-text.pickle4", "-opinion.pickle4")
            opinion_df.to_pickle(output_path)


def recover_all():
    for topic in labelling_folder_map.keys():
        recover_twitter_opinion(topic)


if __name__ == "__main__":
    fire.Fire(
        {
            "predict_weibo": predict_weibo_opinion,
            "predict_twitter": predict_twitter_opinion,
            "recover_all": recover_all,
        }
    )
