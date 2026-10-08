"""
author: Hou Yuxin
date: 2022-06-16
"""

import os
import pandas as pd
import json

from sklearn.model_selection import train_test_split


def augment_class(df, target_class, target_size):
    """
    对指定类别的数据进行扩增，直到样本数量达到目标值。

    Parameters
    ----------
    df : DataFrame
        输入数据集。
    target_class : Any
        要扩增的类别值。
    target_size : int
        扩增后的目标样本数量。

    Returns
    -------
    augmented_df : DataFrame
        扩增后的数据集。
    """
    class_df = df[df["label"] == target_class]
    while len(class_df) < target_size:
        # 随机采样扩增
        class_df = pd.concat(
            [class_df, class_df.sample(n=target_size - len(class_df), replace=True)]
        )
    return class_df


class DataProcess(object):
    """
    Base class for data process.
    """

    def __init__(
        self,
        dataset_dir: str,
        output_dir: str,
        dataset_file: str,
        task_type: str,
        src_type: str = "weibo",
        force_update: bool = False,
        split_ratio: list = [0.6, 0.2, 0.2],
    ) -> None:
        """
        dataset_dir: str, path of dataset directory
        dataset_file: str, the filename of a datafile
        task_type: str, regression or binary
        force_update: bool, when it equals to False, program would use train/validate/test-{task_type}.csv as default datafile
        split_ratio: list, we split the dataset to train, validate, test dataset according to the ratio. Each item should be float and the sum of them should equal to 1
        """
        # task_map = {
        #     "ternary": "binary",
        #     "regression": "regression",
        #     "binary": "binary"
        # }
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.dataset_file = dataset_file
        self.task_type = task_type  # task_map[task_type]
        self.src_type = src_type
        self.force_update = force_update
        self.split_ratio = split_ratio

    def split_df_by_ratio(self, df):
        if sum(self.split_ratio) != 1:
            raise ValueError("Error: the sum of split ratio is larger than 1")
        if len(self.split_ratio) < 3:
            raise ValueError(
                "Error: split_ratio doesn't provide the ratio of train, validate and test dataset"
            )
        print("type", df, type(df))

        train_df = df.sample(frac=self.split_ratio[0])
        remained_df = df[~df.index.isin(train_df.index)]

        validate_df = remained_df.sample(
            frac=self.split_ratio[1] / (self.split_ratio[1] + self.split_ratio[2])
        )
        test_df = remained_df[~remained_df.index.isin(validate_df.index)]

        return train_df, validate_df, test_df

    def split_df_by_label(self, df):
        """
        Code from https://stackoverflow.com/questions/50781562/stratified-splitting-of-pandas-dataframe-into-training-validation-and-test-set
        Splits a Pandas dataframe into three subsets (train, val, and test)
        following fractional ratios provided by the user, where each subset is
        stratified by the values in a specific column (that is, each subset has
        the same relative frequency of the values in the column). It performs this
        splitting by running train_test_split() twice.

        Parameters
        ----------
        df_input : Pandas dataframe
            Input dataframe to be split.
        stratify_colname : str
            The name of the column that will be used for stratification. Usually
            this column would be for the label.
        frac_train : float
        frac_val   : float
        frac_test  : float
            The ratios with which the dataframe will be split into train, val, and
            test data. The values should be expressed as float fractions and should
            sum to 1.0.
        random_state : int, None, or RandomStateInstance
            Value to be passed to train_test_split().

        Returns
        -------
        df_train, df_val, df_test :
            Dataframes containing the three splits.
        """

        if sum(self.split_ratio) != 1:
            raise ValueError("Error: the sum of split ratio is larger than 1")
        if len(self.split_ratio) < 3:
            raise ValueError(
                "Error: split_ratio doesn't provide the ratio of train, validate and test dataset"
            )
        print("type", df, type(df))

        value_counts = df["label"].value_counts()
        for label, count in value_counts.items():
            if count < 10:
                augmented_class_df = augment_class(df, label, target_size=10)
                df = pd.concat([df[df["label"] != label], augmented_class_df])

        X = df  # Contains all columns.
        y = df[["label"]]  # Dataframe of just the column on which to stratify.

        frac_train, frac_val, frac_test = self.split_ratio

        # Split original dataframe into train and temp dataframes.
        df_train, df_temp, y_train, y_temp = train_test_split(
            X, y, stratify=y, test_size=(1.0 - frac_train), random_state=None
        )

        # Split the temp dataframe into val and test dataframes.
        relative_frac_test = frac_test / (frac_val + frac_test)

        if relative_frac_test > 0:
            df_val, df_test, y_val, y_test = train_test_split(
                df_temp,
                y_temp,
                stratify=y_temp,
                test_size=relative_frac_test,
                random_state=None,
            )
        else:
            df_val = df_temp
            df_test = pd.DataFrame(columns=df_val.columns)

        assert len(df) == len(df_train) + len(df_val) + len(df_test)

        return df_train, df_val, df_test

    def save_df(self, train_df, validate_df, test_df, total_df):
        train_df.to_csv(
            os.path.join(self.output_dir, f"train-{self.task_type}.csv"), index=False
        )
        validate_df.to_csv(
            os.path.join(self.output_dir, f"validate-{self.task_type}.csv"), index=False
        )
        test_df.to_csv(
            os.path.join(self.output_dir, f"test-{self.task_type}.csv"), index=False
        )
        total_df.to_csv(
            os.path.join(self.output_dir, f"total-{self.task_type}.csv"), index=False
        )
        print("saving finished")
        return True

    def get_cached_df(self):
        try:
            train_df = pd.read_csv(
                os.path.join(self.output_dir, f"train-{self.task_type}.csv")
            )
            validate_df = pd.read_csv(
                os.path.join(self.output_dir, f"validate-{self.task_type}.csv")
            )
            test_df = pd.read_csv(
                os.path.join(self.output_dir, f"test-{self.task_type}.csv")
            )
            total_df = pd.read_csv(
                os.path.join(self.output_dir, f"total-{self.task_type}.csv")
            )
            return train_df, validate_df, test_df, total_df
        except:
            return None, None, None, None

    def get_merged_dataset(self, split=False):
        df_list = []
        for single_file in self.dataset_files:
            single_df = self.single_file_handler(single_file)
            df_list.append(single_df)
        dataset_df = pd.concat(df_list)
        if split:
            train_df, validate_df, test_df = self.split_df_by_label(dataset_df)
            return train_df, validate_df, test_df, dataset_df
        return dataset_df

    def dataset_handler(self, dataset_file):
        if not dataset_file.endswith(".parquet"):
            raise ValueError("The dataset file should be a parquet file.")
        dataset_path = os.path.join(self.dataset_dir, dataset_file)
        dataset_df = pd.read_parquet(dataset_path, engine="fastparquet")
        dataset_df = dataset_df[dataset_df["agreement_count"] >= 2]
        if self.src_type == "weibo":
            dataset_df = dataset_df[["weibo_id", "weibo_content", "agreement_value"]]
            namemapping = {
                "weibo_id": "id",
                "weibo_content": "text",
                "agreement_value": "label",
            }

            dataset_df.rename(columns=namemapping, inplace=True)
        elif self.src_type == "tweet":
            dataset_df = dataset_df[["text", "agreement_value"]]
            # 将index重新置为id列
            dataset_df.reset_index(drop=False, inplace=True)
            namemapping = {
                "tweet_id": "id",
                "agreement_value": "label",
            }
            dataset_df.rename(columns=namemapping, inplace=True)
        if self.task_type == "binary":
            dataset_df["label"] = dataset_df["label"].apply(
                lambda x: 1 if x in [-2, -1, 0, 1, 2] else 0
            )
        else:
            dataset_df = dataset_df[dataset_df["label"] != -99]
        return dataset_df

    def get_dataset(self):
        """
        Return a 3-item tuple
        (train_dataset, validate_dataset, test_dataset)
        Each element is a pandas DataFrame
        """
        train_df, validate_df, test_df, dataset_df = (
            None,
            None,
            None,
            None,
        )
        if self.force_update is False:
            train_df, validate_df, test_df, dataset_df = self.get_cached_df()

        if train_df is None:
            dataset_df = self.dataset_handler(self.dataset_file)
            train_df, validate_df, test_df = self.split_df_by_label(dataset_df)
            dataset_df = pd.concat([train_df, validate_df, test_df])
        self.save_df(train_df, validate_df, test_df, dataset_df)

        return train_df, validate_df, test_df, dataset_df
