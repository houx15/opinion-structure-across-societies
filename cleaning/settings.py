"""Constants shared by the cleaning scripts (``from cleaning.settings import *``).

Machine-specific paths and API keys come from config/config.py (see
config/config_example.py); everything else is fixed by the study design.
"""

from common.paths import REFERENCE_DIR
from config import cfg

# --- API keys / endpoints (LLM labelling) -------------------------------------
OPENAI_API_KEY = cfg.OPENAI_API_KEY
OPENAI_ORGANIZATION = getattr(cfg, "OPENAI_ORGANIZATION", None)
OPENAI_PROJECT = getattr(cfg, "OPENAI_PROJECT", None)
DEEPSEEK_API_KEY = cfg.DEEPSEEK_API_KEY
DEEPSEEK_BASE_URL = cfg.DEEPSEEK_BASE_URL
DOUBAO_API_KEY = cfg.DOUBAO_API_KEY
DOUBAO_BASE_URL = cfg.DOUBAO_BASE_URL
DOUBAO_END_POINT = cfg.DOUBAO_END_POINT
QWEN_API_KEY = cfg.QWEN_API_KEY
DASHSCOPE_API_KEY = cfg.DASHSCOPE_API_KEY
DASHSCOPE_BASE_URL = cfg.DASHSCOPE_BASE_URL
DEFAULT_QWEN_MODEL = cfg.DEFAULT_QWEN_MODEL
MAX_REQUEST_PER_BATCH = cfg.MAX_REQUEST_PER_BATCH

# Labelling models submitted through the OpenAI-compatible batch API vs. run by hand.
realtime_llms = []
manual_llms = ["deepseek", "doubao"]

# --- Twitter -------------------------------------------------------------------
TWITTER_DIR = cfg.TWITTER_DIR
ORIGINAL_TWEETS_DIR = cfg.ORIGINAL_TWEETS_DIR
OPINION_DIR = cfg.TWITTER_OPINION_DIR
WORD_COUNT_DIR = cfg.TWITTER_WORD_COUNT_DIR
# LLM batch folders (keyword_batch_<model>/), llm_consistency.csv
TWITTER_WORK_DIR = cfg.TWITTER_WORK_DIR

# Topic code -> folder of its tweets under TWITTER_DIR ("<folder>-opinion"
# after BERT prediction). The paper uses abo, gun, clc, sxo, vac, soc, dpp,
# minwage, ubi (US) and swe instead of gun (Europe).
folder_map = {
    "abo": "abortion-merge",
    "clc": "climate-change-merge",
    "gun": "gun-merge",
    "sxo": "sexual-orientation-merge",
    "vac": "vaccine-mandate",
    "soc": "social-media",
    "dpp": "death-penalty",
    "minwage": "minimum-wage",
    "swe": "sex-work-legalization",
    "ubi": "universal-basic-income",
}
# Topics collected in the first crawl (keyword archives already merged per topic).
original_topics = ["abo", "clc", "gun", "sxo", "chn", "dru"]

# Topics labelled from the second crawl (raw archives in ORIGINAL_TWEETS_DIR).
labelling_folder_map = {
    "vac": "vaccine-mandate",
    "obe": "obesity",
    "smok": "public-smoking",
    "hsch": "homeschooling",
    "soc": "social-media",
    "juv": "juvenile-trial",
    "iso": "country-isolationist",
    "dpp": "death-penalty",
    "drkage": "drinking-age",
    "hwm": "homework",
    "minwage": "minimum-wage",
    "swe": "sex-work-legalization",
    "schunif": "school-uniform",
    "ubi": "universal-basic-income",
}
ORIGINAL_TOPIC_LIST = list(labelling_folder_map.values())

# Topic definitions and opinion scales used in the LLM prompts.
TWITTER_OPINION_CONFIG = str(REFERENCE_DIR / "twitter_topics.csv")

# --- Twitter user locations (bot-detection project) ------------------------------
BASE_DIR = cfg.LOCATION_DIR
PROFILE_DIR = cfg.TWITTER_PROFILE_DIR

# --- Weibo -----------------------------------------------------------------------
DATA_SOURCE_DIR = cfg.WEIBO_FRESHDATA_DIR
WEIBO_TOPIC_KEYWORD_DIR = cfg.WEIBO_TOPIC_KEYWORD_DIR
WEIBO_OPINION_DIR = cfg.WEIBO_OPINION_DIR
WEIBO_WORK_DIR = cfg.WEIBO_WORK_DIR
# Weibo topic ids, names, opinion scales and keyword lists.
WEIBO_TOPICS_CSV = str(REFERENCE_DIR / "weibo_topics.csv")
ANALYSIS_YEARS = [2020, 2021, 2022, 2023]

# --- BERT fine-tuning ----------------------------------------------------------------
BERT_DATASET_DIR = cfg.BERT_DATASET_DIR
dataset_base = cfg.BERT_DATASET_DIR
data_output_dir_base = cfg.BERT_DATASET_DIR
output_dir_base = cfg.BERT_MODEL_DIR
log_dir_base = cfg.BERT_LOG_DIR

default_regression_config = {
    "weight_decay": 0.2,
    "warmup_steps": 50,
    "learning_rate": 1e-05,
    "num_train_epochs": 10,
    "per_device_train_batch_size": 16,
}
default_binary_config = {
    "weight_decay": 0,
    "warmup_steps": 20,
    "learning_rate": 1e-05,
    "num_train_epochs": 5,
    "per_device_train_batch_size": 32,
}
