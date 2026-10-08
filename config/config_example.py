"""Example configuration.

Copy this file to ``config/config.py`` (git-ignored) and fill in your values:

    cp config/config_example.py config/config.py

Only the cleaning stage (LLM labelling, BERT, location filtering, raw-data
handling on the clusters) and the OpenAI / word2vec embedding steps read the
values below. The analysis and plotting stages only need the repository's
``data/`` folder (override its location with ``DATA_ROOT``).
"""

# --------------------------------------------------------------------------
# Repository data / output roots. None = <repo>/data and <repo>/outputs.
# The environment variables OSS_DATA_ROOT / OSS_OUTPUT_ROOT override these.
# --------------------------------------------------------------------------
DATA_ROOT = None
OUTPUT_ROOT = None

# --------------------------------------------------------------------------
# API keys (cleaning stage LLM labelling, OpenAI text embeddings)
# --------------------------------------------------------------------------
OPENAI_API_KEY = "sk-..."
OPENAI_ORGANIZATION = None
OPENAI_PROJECT = None

DEEPSEEK_API_KEY = ""
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

DOUBAO_API_KEY = ""
DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/"
DOUBAO_END_POINT = ""

# Qwen via Alibaba DashScope (OpenAI-compatible endpoint)
QWEN_API_KEY = ""
DASHSCOPE_API_KEY = ""
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwq-plus"

# OpenAI batch API: max requests per batch file
MAX_REQUEST_PER_BATCH = 49999

# --------------------------------------------------------------------------
# Raw-data locations on the clusters (cleaning stage only)
# --------------------------------------------------------------------------
# Twitter (Princeton Adroit in the original runs)
TWITTER_DIR = "/path/to/data-tweet"                 # per-topic tweet text pickles (+ "<topic>-opinion" folders after BERT prediction)
TWITTER_PROFILE_DIR = "/path/to/data-twitter-profile"  # user profile dumps (location field)
ORIGINAL_TWEETS_DIR = "/path/to/more_tweets"        # raw keyword-crawled tweet archives; labeling_sample/, training_data/
TWITTER_OPINION_DIR = "/path/to/opinion-correlation/twitter"  # merged-<topic>.parquet per region (twitter/<mode>/)
TWITTER_WORD_COUNT_DIR = "/path/to/opinion-correlation/noun_counter"  # per-topic/year noun+verb counts
TWITTER_WORK_DIR = "/path/to/twitter_work"          # LLM batch folders keyword_batch_<model>/, llm_consistency.csv
LOCATION_DIR = "/path/to/bot-detection"             # location classification outputs, <mode>_user_ids.json

# Weibo (PKU cluster in the original runs)
WEIBO_FRESHDATA_DIR = "/path/to/weibo/freshdata"    # zipped daily Weibo dumps
WEIBO_TOPIC_KEYWORD_DIR = "/path/to/topic_keyword_data"  # <topic_id>/<date>.parquet keyword-matched posts
WEIBO_OPINION_DIR = "/path/to/opinion_data"         # <topic_id>/avg_opinion.parquet
WEIBO_NOUN_COUNT_DIR = "/path/to/noun_count"        # per-topic/year noun counts and TF-IDF
WEIBO_WORK_DIR = "/path/to/weibo_work"              # keyword_text_data/, topic_keyword_data_sample/, LLM batch folders, logs/

# BERT fine-tuning
BERT_DATASET_DIR = "/path/to/merged_labelling"      # <topic>_merged.parquet LLM-labelled training data
BERT_MODEL_DIR = "/path/to/models"                  # fine-tuned checkpoints topic-<topic>/run-<i>-<task>
BERT_LOG_DIR = "logs/bert"
BERT_BASE_MODEL_DIR = "/path/to/base_model"         # local copies of bert-base-cased / chinese-roberta-wwm-ext-large

# Static word vectors (word2vec robustness check)
WORD2VEC_ENGLISH = "/path/to/GoogleNews-vectors-negative300.bin"
WORD2VEC_CHINESE = "/path/to/sgns.merge.word"
