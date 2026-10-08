# Cleaning-stage tables as published in the paper

Transcribed from the paper's SI tables (LLM labelling consistency and
fine-tuned BERT performance). They are the reference values for
`cleaning/descriptive_stats.py`, which recomputes what it can from the
merged LLM labels and the BERT logs.

| file | paper table | content |
|---|---|---|
| `llm_consistency_weibo.csv` | tab:weibo-llm-performance | per topic: labelled posts, share (%) of posts with >= 2 of 3 models agreeing, counts of the agreed label (-2 ... 2, irrelevant) |
| `llm_consistency_twitter.csv` | tab:twitter-llm-performance | same for the Twitter topics labelled with LLMs |
| `bert_performance_weibo.csv` | tab:weibo-bert-performance | relevance classifier accuracy / AUC, opinion regressor RMSE |
| `bert_performance_twitter.csv` | tab:twitter-bert-performance | same for Twitter; AUC not reported for gun / abo / clc / sxo (first-wave topics, earlier labelling pipeline) |

`topic_id` is the code used in the data (Weibo topic id, Twitter topic code).
Twitter abo / clc / gun / sxo are not in the LLM table: they were labelled in
the first data collection, before the LLM labelling pipeline.

## Checked against the recomputed statistics (2026-10-08)

* `llm_consistency_weibo.csv`, `llm_consistency_twitter.csv`: reproduced
  exactly by `python -m cleaning.descriptive_stats labelling` /
  `... twitter` from the merged LLM labels (counts identical, shares equal up
  to rounding).
* `bert_performance_twitter.csv`: accuracy and RMSE are the means over the 5
  fine-tuning runs in the BERT logs and reproduce for obe, soc, dpp, hwm,
  minwage, swe, ubi. vac does not (logs: accuracy 0.794, RMSE 1.236): the vac
  models were retrained afterwards (parameter search), which overwrote the
  logs. The logs contain accuracy, precision and recall but no AUC; note that
  the AUC of vac and obe (0.7911) equals the accuracy of vac. No logs survive
  for gun / abo / clc / sxo (earlier pipeline).
* `bert_performance_weibo.csv`: the Weibo fine-tuning logs are on the PKU
  cluster; not rechecked.
