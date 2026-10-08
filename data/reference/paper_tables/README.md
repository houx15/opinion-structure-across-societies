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
