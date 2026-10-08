# Stage 1: cleaning (raw data -> cleaned opinions)

These steps ran on the clusters that hold the raw data (Twitter: Princeton
Adroit; Weibo: PKU). They are expensive and are **not** rerun to reproduce the
paper; their outputs are the precomputed inputs in `data/` (see
data/README.md). Paths and API keys come from `config/config.py`; shared
constants (topic folders, LLM endpoints, BERT hyper-parameters) are in
`cleaning/settings.py`. Run every module from the repository root.

Environments: `cleaning/requirements.txt` (CPU steps), `cleaning/bert/requirements.txt`
(GPU, pinned versions used for fine-tuning and prediction).

Opinion scale everywhere: -2 (strongly against) ... 2 (strongly for);
-99 = irrelevant / no opinion.

## Surveys

The harmonised survey files `data/survey/<src>.dta` (topic items recoded to
1-5, `country` and `year` columns; `*_media` = respondents who use social
media; `evs_resample2` = EVS resampled with replacement, `evs` = without;
`en_evs` / `eu_nen_evs` = Great Britain / rest of Europe) are prepared outside
this repository.

```bash
python -m cleaning.survey.prepare_survey_individual_data all
#   -> data/opinions/individual_opinion_<src>.parquet
```

## Twitter

**1. Where users are** (`twitter/location/`, results in `LOCATION_DIR`).
Self-reported profile locations are classified by rules first; strings the
rules cannot decide go to an LLM (OpenAI batch API, reply 0 = cannot tell,
1 = inside, 2 = outside).

```bash
# United States: only users located in the US are kept
python -m cleaning.twitter.location.non_us_user_analysis gather      # all profile locations
python -m cleaning.twitter.location.non_us_user_analysis identify    # rule-based US / non-US / unknown
python -m cleaning.twitter.location.llm_location_us                  # LLM for the unknown strings (re-run to advance the batch)
python -m cleaning.twitter.location.non_us_user_analysis merge
python -m cleaning.twitter.location.non_us_user_analysis user        # -> us_user_ids.json

# Europe, step 1: located in (geographic) Europe
python -m cleaning.twitter.location.eu_user_analysis identify
python -m cleaning.twitter.location.llm_location_eu
python -m cleaning.twitter.location.eu_user_analysis merge
python -m cleaning.twitter.location.eu_user_analysis user            # -> eu_user_ids.json
# Europe, step 2: country of the European users -> Great Britain subset
python -m cleaning.twitter.location.eu_country_user_analysis analyze
python -m cleaning.twitter.location.llm_location_eu_country
python -m cleaning.twitter.location.eu_country_user_analysis merge
python -m cleaning.twitter.location.eu_country_user_analysis gb      # -> en_user_ids.json
python -m cleaning.twitter.location.make_eu_nen_user_ids             # Europe minus GB -> eu_nen_user_ids.json
#   (originally a one-off command; this script gives the identical id set)
```

**2. Opinions** (`twitter/opinion/`, `bert/`).

```bash
# raw keyword-crawled archives -> per-topic text files; 10,000-tweet sample per topic for labelling
python -m cleaning.twitter.opinion.original_tweet_handler all_clean
python -m cleaning.twitter.opinion.original_tweet_handler sample
# LLM labels on the sample (OpenAI-compatible batch APIs; qwen = qwq-plus reasoning model)
python -m cleaning.twitter.opinion.llm_labelling label --model openai     # also: qwen, deepseek
python -m cleaning.twitter.opinion.llm_labelling merge                    # majority label; agreement_count;
#   appends the LLM consistency table <TWITTER_WORK_DIR>/llm_consistency.csv
#   -> <ORIGINAL_TWEETS_DIR>/training_data/<topic>_merged.parquet (rows with agreement_count >= 2 train BERT)
# BERT: relevance (binary) and opinion (regression), then predict every tweet
python -m cleaning.bert.main --src_type tweet --topic vac --task_type binary --base_model bert-base-cased
python -m cleaning.bert.main --src_type tweet --topic vac --task_type regression --base_model bert-base-cased
python -m cleaning.bert.predict predict_twitter --topic vac --task_type regression
# user x year opinions from the relevant tweets -> <TWITTER_OPINION_DIR>/merged-<topic>.parquet
python -m cleaning.twitter.opinion.yearly_opinion_calculator convert
python -m cleaning.twitter.opinion.yearly_opinion_calculator merge
# per-region copies (input of analysis/dimension) -> <TWITTER_OPINION_DIR>/<mode>/
python -m cleaning.twitter.opinion.scan_twitter_data --mode us            # eu, en, eu_nen
# LGBT x environment opinions for the Fig 2 case study -> data/opinions/user_opinion_<src>_lgbt_env.parquet
python -m cleaning.twitter.opinion.export_user_opinion_data all --twitter_root <TWITTER_OPINION_DIR> --weibo_root <WEIBO_OPINION_DIR>
```

## Weibo

```bash
# keyword matching in the daily dumps (Aho-Corasick over each topic's keyword list)
python -m cleaning.weibo.get_text_from_keyword --year 2020 --action extract
python -m cleaning.weibo.keyword_text_process 2020 preprocess              # -> WEIBO_TOPIC_KEYWORD_DIR/<topic>/<date>.parquet
python -m cleaning.weibo.keyword_text_process 2020 sample                 # 10,000-post sample per topic
# LLM labels (openai = gpt-4o-mini, qwen, deepseek) and their majority label
python -m cleaning.weibo.llm_labelling label 10 --model deepseek
python -m cleaning.weibo.llm_labelling merge                              # -> BERT_DATASET_DIR/<topic>_merged.parquet
#   (+ the LLM consistency table <WEIBO_WORK_DIR>/llm_consistency.csv)
# BERT (chinese-roberta-wwm-ext-large): relevance and opinion, then predict every post
python -m cleaning.bert.main --src_type weibo --topic 10 --task_type binary
python -m cleaning.bert.main --src_type weibo --topic 10 --task_type regression
python -m cleaning.bert.predict predict_weibo --topic 10 --task_type binary
python -m cleaning.bert.predict predict_weibo --topic 10 --task_type regression
python -m cleaning.weibo.prediction_merge merge 10 regression             # yearly files <-> daily files
# user x year opinions -> WEIBO_OPINION_DIR/<topic>/avg_opinion.parquet
python -m cleaning.weibo.opinion_aggregation convert_all
python -m cleaning.weibo.opinion_aggregation merge_all
```

## Descriptive statistics

```bash
python -m cleaning.descriptive_stats all --weibo_merged_dir PATH --twitter_merged_dir PATH --bert_log_dir PATH
#   -> outputs/stats/<date>_descriptive_stats.txt, outputs/reports/<date>_descriptive_stats/*.csv
```

Respondents per survey and topic, users per social-media source and year,
sample sizes behind each topic pair, the LLM consistency table (labelled posts,
valid labels per model, share with >= 2 and with all 3 models agreeing, agreed
label counts; recomputed from the merged files, plus agreement and Cohen's
kappa per model pair) and BERT evaluation metrics.
Sections whose inputs are missing are skipped.
