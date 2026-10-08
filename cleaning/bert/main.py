import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import importlib
import pandas as pd
from cleaning.bert.data_preprocess import DataProcess
from cleaning.bert.train import OpinionModel
from cleaning.bert.predict import OpinionPredict
from cleaning.settings import *


def run(config):
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)
    if not os.path.exists(data_output_dir_base):
        os.makedirs(data_output_dir_base)

    dataset_dir = dataset_base  # os.path.join(dataset_base, f"topic-{config.topic}")
    output_dir = os.path.join(data_output_dir_base, f"topic-{config.topic}")
    data_process = DataProcess(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        dataset_file=f"{config.topic}_merged.parquet",
        task_type=config.task_type,
        src_type=config.src_type,
        force_update=config.force_update,
    )

    train_df, validate_df, test_df, total_df = data_process.get_dataset()

    tokenizer = config.base_model
    base_model = config.base_model

    opinion_model = OpinionModel(
        task_type=config.task_type,
        tokenize_model=tokenizer,
        model_path=base_model,
        train_df=train_df,
        validate_df=validate_df,
        test_df=test_df,
        total_df=total_df,
        src_type=config.src_type,
        max_length=config.max_length,
    )

    training_args = {
        "weight_decay": 0.03,
        "warmup_steps": 100,
        "learning_rate": config.lr,
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
    }

    if default_regression_config is not None and config.task_type == "regression":
        training_args = default_regression_config
        print("training_args", training_args)
    if default_binary_config is not None and config.task_type == "binary":
        training_args = default_binary_config
        print("training_args", training_args)

    if config.prod:
        opinion_model.prod_train(
            os.path.join(output_dir_base, f"topic-{config.topic}"),
            logging_dir=os.path.join(log_dir_base, f"topic-{config.topic}"),
            args=training_args,
        )
    else:
        for i in range(config.repeat_times):
            model_path = opinion_model.train(
                os.path.join(output_dir_base, f"topic-{config.topic}", f"run-{i}"),
                logging_dir=os.path.join(
                    log_dir_base, f"topic-{config.topic}", f"run-{i}"
                ),
                args=training_args,
                parameter_search=config.parameter_search,
            )
            rmse_dict = opinion_model.evaluate(
                model_path,
                logging_dir=os.path.join(
                    log_dir_base, f"topic-{config.topic}", f"run-{i}"
                ),
                args=training_args,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str)
    parser.add_argument("--src_type", type=str, default="weibo", choices=["weibo", "tweet"])
    parser.add_argument("--task_type", type=str, default="binary")
    parser.add_argument(
        "--base_model", type=str, default="hfl/chinese-roberta-wwm-ext-large"
    )
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--force_update", type=bool, default=False)
    parser.add_argument("--prod", type=bool, default=False)
    parser.add_argument("--repeat_times", type=int, default=5)
    parser.add_argument("--parameter_search", type=bool, default=False)

    config = parser.parse_args()
    run(config)
