"""
author: Hou Yuxin
date: 2025-02-24
"""

import numpy as np
import pandas as pd

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    mean_squared_error,
    roc_auc_score,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from transformers import Trainer, TrainingArguments

from cleaning.bert.utils import sentence_cleaner, log

import time

import os

# check GPU available
# if torch.cuda.is_available():
#     device = torch.device('cuda')
#     print(f'{torch.cuda.device_count()} GPU(s) available'
#     print(f'GPU used: {torch.cuda.get_device_name(0)}'


# else:
#     print('No GPU available, CPU used instead')
#     device = torch.device('cpu')
# torch.cuda.empty_cache()
class RMSEMetric:
    def __init__(self):
        self.batch_rmse = []

    def update(self, preds, target):
        # calculate RMSE for each batch, use numpy
        rmse = np.sqrt(np.mean((preds.cpu().numpy() - target.cpu().numpy()) ** 2))
        self.batch_rmse.append(rmse)

    def compute(self):
        # Get result across entire eval set
        result = {"rmse": np.mean(self.batch_rmse)}
        # Reset batch statistics
        self.batch_rmse = []
        return result


class AccuracyMetric:
    def __init__(self):
        self.all_probs = []
        self.all_labels = []
        self.all_preds = []

    def update(self, probs, preds, target):
        # calculate accuracy for each batch, use numpy
        self.all_probs.extend(probs.tolist())
        self.all_labels.extend(target.tolist())
        self.all_preds.extend(preds.tolist())
        # with open("accuracy.txt", "a") as f:
        #     f.write(f"{acc}\n")
        #     f.write(f"{','.join(preds)}\n")
        #     f.write(f"{','.join(target)}\n")

    def compute(self):
        overall_accuracy = accuracy_score(self.all_labels, self.all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.all_labels, self.all_preds, average="binary"
        )
        auc = roc_auc_score(self.all_labels, self.all_probs)
        result = {
            "accuracy": overall_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        }
        # Reset batch statistics
        self.all_probs = []
        self.all_labels = []
        self.all_preds = []
        return result


rmse_metric = RMSEMetric()
accuracy_metric = AccuracyMetric()


class OpinionModel(object):
    """
    Opinion model
    """

    def __init__(
        self,
        task_type: str = "regression",
        tokenize_model: str = "bert-base-cased",
        model_path: str = "bert-base-cased",
        train_df=None,
        validate_df=None,
        test_df=None,
        total_df=None,
        src_type: str = None,
        max_length: int = None,
        hp_space_optuna=None,
    ) -> None:
        """
        :param task_type: str, classifier type, 'binary' or 'regression' (default)
        :param tokenize_model: str. tokenizer name or path, e.g. 'bert-base-cased' (default), '../model-bert/gun-regression'
        :param train_model: str. model name or path, e.g. 'bert-base-cased' (default), '../model-bert/gun-regression'
        :param train_dataset, validate_dataset, test_dataset: pandas DataFrame. Set to None if no need to train.
        :param src_type: str. data source. 'tweet' or 'weibo'
        """
        self.task_type = task_type

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenize_model, do_lower_case=False
        )

        self.model_path = model_path
        self.src_type = src_type

        self.max_length = max_length

        self.train_df = train_df
        self.validate_df = validate_df
        self.test_df = test_df
        self.total_df = total_df

        self.error_analysis = {}
        self.hp_space_optuna = hp_space_optuna
        self.model_path_for_evaluation = None

    def tokenize(self, texts, labels, task_name):
        from nlp import Dataset

        # to show the tokenized length
        length_list = []

        print(f"{task_name}-labels", labels)
        count = 0
        max_length = 0
        for single_text in texts:
            single_text = sentence_cleaner(self.src_type, single_text)
            input_ids = self.tokenizer.encode(single_text, add_special_tokens=True)
            max_length = max(max_length, len(input_ids))
            length_list.append(len(input_ids))

        print(f"max_length: {max_length}")
        if self.max_length:
            max_length = min(self.max_length, max_length)
        input_ids = []
        attention_mask = []
        token_type_ids = []

        for single_text in texts:
            single_text = sentence_cleaner(self.src_type, single_text)
            encoded_dict = self.tokenizer.encode_plus(
                single_text,
                add_special_tokens=True,
                return_token_type_ids=True,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
                return_tensors=None,
            )
            # print(single_text, self.tokenizer.tokenize(single_text))
            input_ids.append(encoded_dict["input_ids"])
            attention_mask.append(encoded_dict["attention_mask"])
            token_type_ids.append(encoded_dict["token_type_ids"])

        encoded_result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": (
                [float(label) for label in labels]
                if self.task_type == "regression"
                else [int(label) for label in labels]
            ),
        }
        dataset = Dataset.from_dict(encoded_result)
        dataset = dataset.shuffle()
        return dataset

    def binary_metrics_compute(self, eval_pred):  # , compute_result):
        labels = eval_pred.label_ids
        preds = (
            eval_pred.predictions[0]
            if isinstance(eval_pred.predictions, tuple)
            else eval_pred.predictions
        )
        probs = preds[:, 1] if preds.ndim > 1 else preds
        preds = np.argmax(preds, axis=1)
        # accuracy_metric.update(probs, preds, labels)
        # if compute_result:
        #     return accuracy_metric.compute()
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )
        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, probs)
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        }

    def regression_metrics_compute(self, eval_pred, compute_result):
        preds = (
            eval_pred.predictions[0]
            if isinstance(eval_pred.predictions, tuple)
            else eval_pred.predictions
        )
        preds = np.squeeze(preds)
        rmse_metric.update(preds, eval_pred.label_ids)
        if compute_result:
            return rmse_metric.compute()

    def model_init(self):
        self.error_analysis = {}
        model_path = self.model_path
        if self.model_path_for_evaluation:
            model_path = self.model_path_for_evaluation
        print(f">>>>>>>>>initializing model:  {model_path}")
        num_labels_map = {"binary": 2, "regression": 1}
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
        return self.model

    def default_hp_space_optuna(self, trial):
        return {
            "weight_decay": trial.suggest_categorical(
                "weight_decay", [0, 0.1, 0.2, 0.3]
            ),
            "warmup_steps": trial.suggest_categorical(
                "warmup_steps", [0, 20, 30, 50, 100, 200]
            ),
            # "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "learning_rate": trial.suggest_categorical(
                "learning_rate",
                [
                    1e-5,
                    1e-4,
                    1e-3,
                ],
            ),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 5, 15, log=True),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [8, 16, 32]
            ),
        }

    def train(self, output_dir, logging_dir, args, parameter_search: bool = False):
        if self.train_df is None or self.validate_df is None:
            raise ValueError(
                "Dev Training: train dataset or validate dataset cannot be None"
            )
        # draw_hist(self.test_df.label.values, logging_dir+'aej0205-label.jpg')
        # raise ValueError

        train_x, train_y = self.train_df.text.values, self.train_df.label.values
        train_dataset = self.tokenize(train_x, train_y, "train")
        validate_dataset = self.tokenize(
            self.validate_df.text.values, self.validate_df.label.values, "validate"
        )

        output_dir = output_dir + f"-{self.task_type}/"
        logging_dir = logging_dir + f"-{self.task_type}/"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)

        arg_cls = TrainingArguments
        train_cls = Trainer

        training_args = arg_cls(
            output_dir=output_dir,
            overwrite_output_dir=True,
            do_train=True,
            batch_eval_metrics=True if self.task_type == "regression" else False,
            # load_best_model_at_end=True,
            metric_for_best_model=(
                "loss" if self.task_type == "regression" else "accuracy"
            ),
            # eval_accumulation_steps=50,
            # evaluation_strategy="no",
            evaluation_strategy="steps" if self.task_type == "regression" else "no",
            logging_strategy="steps",
            save_strategy="no",
            logging_steps=5,
            report_to=["tensorboard"],
            logging_dir=logging_dir,
            disable_tqdm=True,
        )

        for n, v in args.items():
            setattr(training_args, n, v)

        # model = self.model_init()
        self.model = self.model_init()
        self.model.train()

        trainer = train_cls(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validate_dataset,
            tokenizer=self.tokenizer,
        )
        if self.task_type == "binary":
            trainer.compute_metrics = self.binary_metrics_compute
        elif self.task_type == "ternary":
            trainer.compute_metrics = self.ternary_metrics_compute
        else:
            trainer.compute_metrics = self.regression_metrics_compute

        if parameter_search:
            import optuna

            hp_space = (
                self.hp_space_optuna
                if self.hp_space_optuna is not None
                else self.default_hp_space_optuna
            )

            trainer.model_init = self.model_init
            best_run = trainer.hyperparameter_search(
                hp_space=lambda x: hp_space(x),
                backend="optuna",
                n_trials=5,
                direction="maximize" if self.task_type == "binary" else "minimize",
            )
            print("best_run", best_run)

            for n, v in best_run.hyperparameters.items():
                setattr(trainer.args, n, v)

            self.resume_from_checkpoint = False

        trainer.train()
        trainer.save_model()

        torch.cuda.empty_cache()

        return output_dir

    def prod_train(self, output_dir, logging_dir, args):
        if self.total_df is None:
            raise ValueError("prod training: total df cannot be none")
        train_dataset = self.tokenize(
            self.total_df.text.values, self.total_df.label.values, "total"
        )

        output_dir = output_dir + f"{self.task_type}/prod/"
        logging_dir = logging_dir + f"{self.task_type}/prod/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)

        arg_cls = TrainingArguments
        train_cls = Trainer

        training_args = arg_cls(
            output_dir=output_dir,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=False,
            # load_best_model_at_end=True,
            metric_for_best_model=(
                "loss" if self.task_type == "regression" else "accuracy"
            ),
            evaluation_strategy="no",
            logging_strategy="epoch",
            save_strategy="steps",
            logging_steps=5,
            report_to=["tensorboard"],
            logging_dir=logging_dir,
            disable_tqdm=True,
        )

        for n, v in args.items():
            setattr(training_args, n, v)

        # model = self.model_init()
        self.model = self.model_init()
        self.model.train()

        trainer = train_cls(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            # disable_tqdm=True
        )
        if self.task_type == "binary":
            trainer.compute_metrics = self.binary_metrics_compute
        else:
            trainer.compute_metrics = self.regression_metrics_compute

        trainer.train()
        trainer.save_model()

        torch.cuda.empty_cache()

        return output_dir

    def evaluate(
        self, model_path, logging_dir, args={"per_device_eval_batch_size": 16}
    ):
        prediction = np.array([])
        if self.test_df is None:
            raise ValueError("Evaluate: test dataset cannot be none")

        logging_dir = logging_dir + f"{self.task_type}/"
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)

        training_args = TrainingArguments(
            output_dir=model_path,
            do_train=False,
        )

        for n, v in args.items():
            setattr(training_args, n, v)

        # model = self.model_init()
        self.model_path_for_evaluation = model_path
        self.model = self.model_init()
        self.model.eval()
        # for parameter in self.model.parameters():
        #     parameter.requires_grad = False  # freeze the finetuned model. save memory.
        # torch.cuda.empty_cache()
        texts = self.test_df.text.values
        batch = 16
        predictions = []
        for i in range(0, len(texts), batch):
            # encode input texts
            encoding = self.tokenizer(
                [
                    sentence_cleaner(self.src_type, single_text)
                    for single_text in texts[i : i + batch]
                ],
                add_special_tokens=True,
                return_token_type_ids=True,
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
                max_length=330,
            )
            if torch.cuda.is_available():
                # print(f'self.model.device={self.model.device}')  # self.model.device=cuda:0
                for key in encoding.keys():
                    encoding[key] = encoding[key].cuda()
                    # print(f'encoding[{key}].device={encoding[key].device}. encoding[{key}].shape={encoding[key].shape}') # encoding[input_ids].device=cuda:0. encoding[input_ids].shape=torch.Size([4, 128])

            # calculate the encoded input with frozen model
            outputs = self.model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                token_type_ids=encoding["token_type_ids"],
            ).logits.detach()
            # print(f'type(outputs)={type(outputs)}, outputs.device={outputs.device}') #  type(outputs)=<class 'torch.Tensor'>, outputs.device=cuda:0
            if torch.cuda.is_available():
                outputs = (
                    outputs.cpu()
                )  # copy the tensor to host memory before converting it to numpy. otherwise we will get an error "can't convert cuda:0 device type tensor to numpy"
                # print(f'type(outputs)={type(outputs)}, outputs.device={outputs.device}')  # type(outputs)=<class 'torch.Tensor'>, outputs.device=cpu
            del encoding

            # output
            if self.task_type == "regression":
                prediction = np.append(
                    prediction, outputs.numpy().flatten()
                )  # a score for each input string
            elif self.task_type in ["binary"]:
                prediction = np.append(
                    prediction, np.argmax(outputs.numpy(), axis=1)
                )  # a float probability of label '1' for each input string
            else:
                raise ValueError(f"Unknown self.task_type = {self.task_type}.")
            del outputs
            torch.cuda.empty_cache()
            print(f"prediction.shape={prediction.shape}")

        log_file = logging_dir + "log.txt"
        log(log_file, time.asctime(time.localtime(time.time())))

        if self.task_type == "regression":
            rmse = mean_squared_error(
                self.test_df.label.values, prediction, squared=False
            )
            print(rmse)
            log(log_file, f"rmse:    {rmse}")

        else:
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.test_df.label.values, prediction, average="binary"
            )
            acc = accuracy_score(self.test_df.label.values, prediction)
            auc = roc_auc_score(self.test_df.label.values, prediction)
            print(f"acc: {acc}")
            print(f"precision: {precision}")
            print(f"recall: {recall}")
            print(f"f1: {f1}")
            print(f"auc: {auc}")

            log_file = logging_dir + "log.txt"
            log(log_file, time.asctime(time.localtime(time.time())))
            log(log_file, f"accuracy:    {acc}")
            log(log_file, f"precision: {precision}")
            log(log_file, f"recall: {recall}")

        df = pd.DataFrame(
            {
                "id": self.test_df.id.values,
                "text": self.test_df.text.values,
                "prediction": prediction,
                "label": self.test_df.label.values,
            }
        )

        df.to_csv(logging_dir + f"prediction-{int(time.time())}.csv", index=False)

        # output
        return df
