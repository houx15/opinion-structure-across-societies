"""Re-evaluate saved fine-tuned BERT models on their saved test splits.

For every <BERT_MODEL_DIR>/topic-<t>/run-<i>-<task>/ checkpoint, predict the
test split written at training time (<BERT_DATASET_DIR>/topic-<t>/test-<task>.csv)
with the same text cleaning and tokenisation as cleaning.bert.train
(OpinionModel.evaluate, max_length 330) and compute

* relevance (binary): accuracy, precision, recall, F1,
  auc_label = ROC AUC of the predicted labels (as OpinionModel.evaluate
  computes it; equals balanced accuracy) and auc_prob = ROC AUC of the
  predicted probability of "relevant";
* opinion (regression): RMSE.

Writes outputs/reports/<date>_bert_reevaluation/{runs,summary}.csv; summary =
mean / sd over the runs of a topic and task (the paper reports run means).
Needs a GPU environment (cleaning/bert/requirements.txt); no network access.

    python -m cleaning.bert.evaluate --src_type tweet
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import fire
import numpy as np
import pandas as pd

from cleaning.bert.utils import sentence_cleaner
from cleaning.settings import BERT_DATASET_DIR, output_dir_base
from common.paths import report_dir

MAX_LENGTH = 330


def _predict(model_dir: Path, texts, src_type: str, batch: int = 64) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, do_lower_case=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tokenizer(
                [sentence_cleaner(src_type, t) for t in texts[i: i + batch]],
                add_special_tokens=True, return_token_type_ids=True, truncation=True,
                padding="max_length", return_attention_mask=True, return_tensors="pt",
                max_length=MAX_LENGTH,
            ).to(device)
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                           token_type_ids=enc["token_type_ids"]).logits
            out.append(logits.float().cpu().numpy())
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(out)


def _metrics(task: str, labels: np.ndarray, logits: np.ndarray) -> dict:
    from sklearn.metrics import (accuracy_score, mean_squared_error,
                                 precision_recall_fscore_support, roc_auc_score)

    if task == "regression":
        return {"rmse": float(np.sqrt(mean_squared_error(labels, logits.reshape(-1))))}
    pred = logits.argmax(axis=1)
    z = logits - logits.max(axis=1, keepdims=True)
    prob = np.exp(z[:, 1]) / np.exp(z).sum(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average="binary")
    return {"accuracy": accuracy_score(labels, pred), "precision": precision, "recall": recall,
            "f1": f1, "auc_label": roc_auc_score(labels, pred), "auc_prob": roc_auc_score(labels, prob)}


def main(src_type: str = "tweet", model_dir: Optional[str] = None,
         dataset_dir: Optional[str] = None, topics: Optional[str] = None) -> Path:
    model_root = Path(model_dir or output_dir_base)
    data_root = Path(dataset_dir or BERT_DATASET_DIR)
    wanted = set(topics.split(",")) if isinstance(topics, str) else (set(topics) if topics else None)
    rows = []
    for run_dir in sorted(model_root.glob("topic-*/run-*")):
        topic = run_dir.parent.name[len("topic-"):]
        m = re.fullmatch(r"run-(\d+)-?(binary|regression)", run_dir.name)
        if not m or (wanted and topic not in wanted):
            continue
        run, task = int(m.group(1)), m.group(2)
        test_csv = data_root / f"topic-{topic}" / f"test-{task}.csv"
        if not (run_dir / "config.json").exists() or not test_csv.exists():
            print(f"[skip] {run_dir} (checkpoint or {test_csv.name} missing)")
            continue
        test = pd.read_csv(test_csv)
        logits = _predict(run_dir, test["text"].astype(str).tolist(), src_type)
        row = {"topic": topic, "task": task, "run": run, "n_test": len(test),
               **_metrics(task, test["label"].to_numpy(), logits)}
        print(row, flush=True)
        rows.append(row)

    runs = pd.DataFrame(rows)
    out = report_dir("bert_reevaluation")
    runs.to_csv(out / "runs.csv", index=False)
    metrics = [c for c in runs.columns if c not in ("topic", "task", "run", "n_test")]
    g = runs.groupby(["topic", "task"])
    summary = g.agg(runs=("run", "size"), n_test=("n_test", "first"))
    for c in metrics:
        summary[f"{c}_mean"] = g[c].mean()
        summary[f"{c}_sd"] = g[c].std()
    summary = summary.reset_index().dropna(axis=1, how="all")
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    fire.Fire(main)
