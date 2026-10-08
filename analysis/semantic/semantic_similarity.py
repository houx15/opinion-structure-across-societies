import os
import glob
import pickle

import numpy as np
import pandas as pd

from collections import Counter
from itertools import combinations
from tqdm import tqdm

try:
    import fire
except ImportError:
    fire = None

from common.paths import EMBEDDING_DIR, REFERENCE_DIR, TF_IDF_DIR

CACHED_EMBEDDING = EMBEDDING_DIR / "cached_embedding.pkl"
CACHED_DICTIONARY_EMBEDDING = EMBEDDING_DIR / "cached_dictionary_embedding.pkl"


def save_to_pkl(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    # print(f"Data saved to {file_path}")


def load_from_pkl(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    print(f"Data loaded from {file_path}")
    return data


def get_weighted_keywords_centroid(embedding_list, weights):
    embedding_array = np.array(embedding_list)
    weights = np.array(weights)
    weighted_sum = np.dot(weights, embedding_array)
    centroid = weighted_sum / weights.sum()
    return centroid


def cosine_similarity_calculator(embedding1, embedding2):
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    norm_a = sum(a**2 for a in embedding1) ** 0.5
    norm_b = sum(b**2 for b in embedding2) ** 0.5
    return dot_product / (norm_a * norm_b)


class DynamicSemanticEmbedding:
    def __init__(self):
        self.cached_embeddings = load_from_pkl(CACHED_EMBEDDING)
        print(type(self.cached_embeddings))
        if self.cached_embeddings is None:
            self.cached_embeddings = {}

    def get_embedding(self, keyword):
        if keyword in self.cached_embeddings:
            return self.cached_embeddings[keyword]
        from analysis.semantic.openai_embedding import get_semantic_embedding

        embedding = get_semantic_embedding(keyword)
        self.cached_embeddings[keyword] = embedding
        save_to_pkl(self.cached_embeddings, CACHED_EMBEDDING)
        return embedding


class DynamicDictionaryEmbedding:
    def __init__(self, src_type="weibo"):
        self.cached_embeddings = load_from_pkl(CACHED_DICTIONARY_EMBEDDING)
        if self.cached_embeddings is None:
            self.cached_embeddings = {}
        if src_type == "weibo":
            from robustness.word2vec_embedding import ChineseDictionaryEmbedding

            self.dictionary_embedding = ChineseDictionaryEmbedding()
        else:
            from robustness.word2vec_embedding import EnglishDictionaryEmbedding

            self.dictionary_embedding = EnglishDictionaryEmbedding()

    def get_embedding(self, keyword):
        if keyword in self.cached_embeddings:
            return self.cached_embeddings[keyword]
        embedding = self.dictionary_embedding.get_embedding(keyword)
        self.cached_embeddings[keyword] = embedding
        save_to_pkl(self.cached_embeddings, CACHED_DICTIONARY_EMBEDDING)
        return embedding


def _normalize_topic(topic):
    if topic is None:
        return ""
    return str(topic).strip().lower().replace("_", "")


def _build_topic_alias():
    # Survey dataset topic naming aliases.
    return {
        "unemployaid": "UnemplyAid",
        "unemplyaid": "UnemplyAid",
    }


def _cosine_similarity_np(vec1, vec2):
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom == 0:
        return np.nan
    return float(np.dot(vec1, vec2) / denom)


# -- Survey topic embeddings ---------------------------------------------------
#
# Like the social-media pipeline (process_embedding), a topic centroid is the
# mean of its keyword embeddings; survey keywords are weighted equally (see
# survey_topic_keyword_counts) and row order is never used.
# Survey keywords come from survey_codebook_tfidf.py (lemmatized nouns, verbs,
# adjectives and noun-compound phrases of the codebook questions). Question
# wording is separated from topic content by three rules (select_survey_keywords),
# each comparing a keyword's embedding with the topic name ("anchor"):
#   1. cosine with its own topic's anchor >= SURVEY_ANCHOR_THRESHOLD;
#   2. closer to its own topic's anchor than to any other topic's anchor in the
#      same survey (generic wording is about equally close to every topic);
#   3. a keyword that also occurs in another topic's questions (battery
#      wording such as "justified", "card") must reach SURVEY_SHARED_THRESHOLD.
# The hand-curated lists (tf_idf/survey_<region>_topic_tfidf_curated.csv) are a
# robustness variant.

# region -> survey sources sharing its codebook topics
SURVEY_REGIONS = {
    "us": ["anes", "anes_media"],
    "europe": ["evs", "evs_media", "evs_resample2", "evs_media_resample2",
               "en_evs", "en_evs_media", "eu_nen_evs"],
    "china": ["wvs", "wvs_media"],
}

# Topic name used as the semantic anchor of each codebook topic.
SURVEY_TOPIC_ANCHORS = {
    "us": {
        "Abortion": "abortion", "Gun": "gun control", "Climate": "climate change",
        "LGBT": "LGBT rights", "VACC": "vaccination", "DeathPenalty": "death penalty",
        "Media": "news media", "MinimumWage": "minimum wage", "UBI": "universal basic income",
    },
    "europe": {
        "LGBT": "homosexuality", "Abortion": "abortion", "DeathPenalty": "death penalty",
        "Egalitarian": "welfare state", "Environment": "environmental protection",
        "SocialMedia": "social media", "Prostitution": "prostitution",
        "HealthCare": "health care", "UnemployAid": "unemployment aid", "Press": "news press",
    },
    "china": {
        "Corrup": "corruption", "GenderEqual": "gender equality",
        "Marriage": "marriage and divorce", "Childbearing": "childbearing",
        "LGBT": "homosexuality", "Environment": "environmental protection", "Econ": "economy",
        "Work": "job security", "GovCtrl": "government surveillance", "Foreign": "immigration",
    },
}

SURVEY_ANCHOR_THRESHOLD = 0.30
SURVEY_SHARED_THRESHOLD = 0.50

# Review of the automatic selection against the question text (LLM-assisted,
# confirmed by the authors). Each entry removes one keyword that is question or
# answer wording rather than topic content.
SURVEY_KEYWORD_REVIEW_DROP = {
    ("us", "Abortion", "upset"): "answer framing: 'would you be pleased or upset'",
    ("us", "MinimumWage", "lower"): "answer option: raised / kept / lowered",
    ("europe", "Environment", "important thing"): "statement wording: 'more important things'",
    ("europe", "Environment", "prevent"): "generic verb, equally close to another topic",
    ("china", "Environment", "top priority"): "statement wording: 'given priority'",
    ("china", "Econ", "consider"): "interview phrase: 'all things considered'",
}
# Topic content the automatic rule missed because the question never names it.
SURVEY_KEYWORD_REVIEW_ADD = {
    ("us", "UBI"): ["citizen", "universal basic income"],
    ("europe", "Egalitarian"): ["responsibility", "individual"],
    ("china", "Childbearing"): ["duty towards society"],
}
SURVEY_KEYWORD_SETS = ("selected", "curated")


def _survey_keyword_csv(region, keywords):
    if keywords == "selected":
        return TF_IDF_DIR / f"survey_{region}_topic_keywords_selected.csv"
    if keywords == "curated":
        return TF_IDF_DIR / f"survey_{region}_topic_tfidf_curated.csv"
    raise ValueError(f"keywords must be one of {SURVEY_KEYWORD_SETS}, got {keywords!r}")


def select_survey_keywords(
    threshold=SURVEY_ANCHOR_THRESHOLD,
    shared_threshold=SURVEY_SHARED_THRESHOLD,
):
    """Separate topic keywords from question wording for every survey region.

    Writes tf_idf/survey_<region>_topic_keywords_selected.csv with the kept
    candidates and their cosine to the own anchor (``anchor_sim``) and to the
    closest other anchor (``other_sim``). A topic left empty keeps its single
    closest keyword.
    """
    dse = DynamicSemanticEmbedding()
    for region, anchors in SURVEY_TOPIC_ANCHORS.items():
        cand = pd.read_csv(TF_IDF_DIR / f"survey_{region}_topic_tfidf_candidates.csv")
        missing = sorted(set(cand["topic"]) - set(anchors))
        if missing:
            raise KeyError(f"{region}: no anchor for {missing}")
        topics = list(anchors)
        A = np.array([dse.get_embedding(anchors[t]) for t in topics], dtype=float)
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        own_sim, other_sim = [], []
        for word, topic in zip(cand["word"], cand["topic"]):
            v = np.asarray(dse.get_embedding(word), dtype=float)
            s = A @ (v / np.linalg.norm(v))
            i = topics.index(topic)
            own_sim.append(s[i])
            other_sim.append(np.delete(s, i).max())
        cand["anchor"] = cand["topic"].map(anchors)
        cand["anchor_sim"] = own_sim
        cand["other_sim"] = other_sim
        reviewed_out = [
            (region, t, w) in SURVEY_KEYWORD_REVIEW_DROP
            for t, w in zip(cand["topic"], cand["word"])
        ]
        rule = (
            (cand["anchor_sim"] >= threshold)
            & (cand["anchor_sim"] > cand["other_sim"])
            & ((cand["df"] == 1) | (cand["anchor_sim"] >= shared_threshold))
            & ~pd.Series(reviewed_out, index=cand.index)
        )
        cand["selected_by"] = "rule"
        kept = []
        for topic, group in cand.groupby("topic", sort=False):
            keep = group[rule.loc[group.index]]
            added = [w for w in SURVEY_KEYWORD_REVIEW_ADD.get((region, topic), [])
                     if w not in set(keep["word"])]
            if added:
                rows = []
                for w in added:
                    v = np.asarray(dse.get_embedding(w), dtype=float)
                    s = A @ (v / np.linalg.norm(v))
                    i = topics.index(topic)
                    rows.append({"topic": topic, "word": w, "ngram": len(w.split()),
                                 "anchor": anchors[topic], "anchor_sim": s[i],
                                 "other_sim": np.delete(s, i).max(), "selected_by": "review"})
                keep = pd.concat([keep, pd.DataFrame(rows)], ignore_index=True)
            if keep.empty:
                keep = group.nlargest(1, "anchor_sim")
                print(f"[{region}/{topic}] no keyword passes; kept {keep['word'].tolist()}")
            kept.append(keep)
        out = pd.concat(kept).sort_values(["topic", "anchor_sim"], ascending=[True, False])
        path = _survey_keyword_csv(region, "selected")
        out.to_csv(path, index=False)
        print(f"[{region}] wrote {path}: {len(out)} keywords for {out['topic'].nunique()} topics")


def survey_topic_keyword_counts(region, keywords="selected"):
    """Survey keywords per topic: {topic_id: DataFrame(word, count)}.

    ``keywords`` is "selected" (automatic filters + review) or "curated" (by
    hand). Every survey keyword gets weight 1: in a short questionnaire item a
    keyword's count reflects how the question is phrased (battery items repeat
    the topic word), not how central the word is.
    Topic ids follow the topic-distance files (e.g. UnemployAid -> UnemplyAid).
    """
    path = _survey_keyword_csv(region, keywords)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing; run analysis.semantic.survey_codebook_tfidf and `semantic_similarity survey_select`")
    table = pd.read_csv(path)
    topic_alias = _build_topic_alias()
    out = {}
    for raw_topic, group in table.groupby("topic", sort=False):
        words = list(dict.fromkeys(group["word"].astype(str).str.strip()))
        counts = pd.DataFrame({"word": words, "count": 1.0})
        out[topic_alias.get(_normalize_topic(raw_topic), raw_topic)] = counts
    return out


# Function words skipped when a phrase vector is composed from its words.
PHRASE_STOP_WORDS = {"a", "an", "the", "of", "to", "for", "in", "on", "and", "or",
                     "towards", "toward", "with", "by", "at", "from"}


def _keyword_embedding_source(dse, word):
    """(vector, source) of one survey keyword; source is word / phrase / composed / None.

    OpenAI embeddings cover any text. For word2vec, a multi-word keyword uses
    the model's phrase token (death_penalty) when it exists, otherwise the
    mean vector of its content words, so the keyword set is the same for both
    embedding models; it still counts as one keyword.
    """
    emb = dse.get_embedding(word)
    if emb is not None:
        return emb, "word"
    if " " not in word or not isinstance(dse, DynamicDictionaryEmbedding):
        return None, None
    emb = dse.get_embedding(word.replace(" ", "_"))       # word2vec phrase token
    if emb is not None:
        return emb, "phrase"
    parts = [t for t in word.split() if t.lower() not in PHRASE_STOP_WORDS]
    vecs = [v for v in (dse.get_embedding(t) for t in parts) if v is not None]
    if not vecs:
        return None, None
    return np.mean(np.asarray(vecs, dtype=float), axis=0), "composed"


def _keyword_embedding(dse, word):
    return _keyword_embedding_source(dse, word)[0]


def survey_topic_centroids(region, dse, keywords="selected"):
    """Equal-weight keyword centroid per survey topic (prints keyword coverage)."""
    centroids = {}
    sources = Counter()
    missing = []
    for topic, kw in survey_topic_keyword_counts(region, keywords).items():
        embeddings, weights = [], []
        for word, count in zip(kw["word"], kw["count"]):
            emb, source = _keyword_embedding_source(dse, word)
            sources[source] += 1
            if emb is None:
                missing.append(f"{topic}:{word}")
                continue
            embeddings.append(emb)
            weights.append(count)
        if embeddings:
            centroids[topic] = get_weighted_keywords_centroid(embeddings, weights)
    total = sum(sources.values())
    print(f"[{region}/{keywords}] keyword vectors: {total - sources[None]}/{total} "
          f"(word {sources['word']}, phrase token {sources['phrase']}, "
          f"composed {sources['composed']}); missing: {missing or 'none'}")
    return centroids


def process_survey_embedding(
    embedding_type="gpt",
    keywords="selected",
    backup_dir=str(EMBEDDING_DIR / "backup_full_question"),
):
    """Write survey topic-pair cosines into embedding/<src>_topic_distance.csv.

    Columns: ``similarity`` (gpt) / ``dictionary_similarity`` (word2vec) from
    the selected keywords; ``*_curated`` from the hand-curated keywords. Every
    survey source of a region gets the same values. The first time a file is
    rewritten the original is copied to ``backup_dir``, and its full-question
    ``similarity`` is kept as ``similarity_question``.
    """
    column = {"gpt": "similarity", "dictionary": "dictionary_similarity"}[embedding_type]
    if keywords == "curated":
        column += "_curated"
    dse = DynamicSemanticEmbedding() if embedding_type == "gpt" else DynamicDictionaryEmbedding("twitter")
    os.makedirs(backup_dir, exist_ok=True)
    for region, sources in SURVEY_REGIONS.items():
        centroids = survey_topic_centroids(region, dse, keywords)
        for src in sources:
            path = str(EMBEDDING_DIR / f"{src}_topic_distance.csv")
            if not os.path.exists(path):
                print(f"Missing file: {path}, skip.")
                continue
            backup = os.path.join(backup_dir, os.path.basename(path))
            if not os.path.exists(backup):
                pd.read_csv(path).to_csv(backup, index=False)
            df = pd.read_csv(path)
            if "similarity_question" not in df.columns:
                df["similarity_question"] = pd.read_csv(backup)["similarity"].to_numpy()
            df[column] = [
                _cosine_similarity_np(centroids[str(t1)], centroids[str(t2)])
                if str(t1) in centroids and str(t2) in centroids else np.nan
                for t1, t2 in zip(df["topic_id1"], df["topic_id2"])
            ]
            if df[column].isna().any():
                raise ValueError(f"{path}: topics without a centroid in {column}")
            df.to_csv(path, index=False)
            print(f"[{region}] wrote {column} -> {path} ({len(df)} rows)")


weibo_topic_list = [
    "0",
    "1",
    "2",
    "4",
    "5",
    "6",
    "7",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
]
twitter_topic_list = [
    "gun",
    "abo",
    "sxo",
    "clc",
    # "chn",
    # "dru",
    "vac",
    # "obe",
    "soc",
    # "hwm",
    # "swe",
    "ubi",
    "minwage",
    "dpp",
]
# All EU twitter variants share the same list: US minus gun, plus swe.
twitter_eu_topic_list = [
    "abo",
    "sxo",
    "clc",
    "vac",
    "soc",
    "ubi",
    "minwage",
    "dpp",
    "swe",
]

SRC_TOPICS = {
    "weibo": weibo_topic_list,
    "twitter": twitter_topic_list,
    "entwitter": twitter_eu_topic_list,
    "eutwitter": twitter_eu_topic_list,
    "eu_nentwitter": twitter_eu_topic_list,
}


def process_embedding(src_type, embedding_type="gpt"):
    """
    Process  data and calculate dynamic semantic embeddings.
    """
    # Initialize the dynamic semantic embedding calculator
    dse = None
    if embedding_type == "gpt":
        dse = DynamicSemanticEmbedding()
    elif embedding_type == "dictionary":
        dse = DynamicDictionaryEmbedding(src_type)
    else:
        raise ValueError(f"Invalid embedding type: {embedding_type}")
    data = []
    if src_type not in SRC_TOPICS:
        raise ValueError(f"Unknown src_type: {src_type}. Known: {list(SRC_TOPICS)}")
    topic_list = SRC_TOPICS[src_type]
    print(f"[{src_type}/{embedding_type}] topics={topic_list}")
    word_column = "Noun" if src_type == "weibo" else "word"
    count_column = "Frequency" if src_type == "weibo" else "count"
    for year in [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, "merged"]:
        # Load the data for the specific year
        topic_centroid = {}
        for topic in tqdm(topic_list):
            # if topic in [3, 8]:
            #     continue
            # topic = str(topic)
            tf_idf_folder = TF_IDF_DIR / f"tf_idf_{src_type}"
            tf_idf_path = (
                f"{tf_idf_folder}/{topic}_{year}_tf_idf.parquet"
                if src_type == "weibo"
                else f"{tf_idf_folder}/tf-idf-{topic}-{year}.parquet"
            )
            if not os.path.exists(tf_idf_path):
                print(f"File {tf_idf_path} does not exist. Skipping...")
                continue
            tf_idf = pd.read_parquet(tf_idf_path)
            # 按照TF-IDF排序取前100
            tf_idf = tf_idf.sort_values(by="TF-IDF", ascending=False).head(10)
            # 计算关键词的embedding
            tf_idf["embedding"] = tf_idf[word_column].apply(dse.get_embedding)
            original_length = len(tf_idf)
            # 过滤掉 None 值（当词不在词典中时）
            tf_idf = tf_idf[tf_idf["embedding"].notna()]
            # 检查是否有有效的 embedding
            if len(tf_idf) == 0:
                print(
                    f"Warning: No valid embeddings for topic {topic} in year {year}. Skipping..."
                )
                continue
            print(
                f"Original length: {original_length}, New length: {len(tf_idf)}, {len(tf_idf)/original_length*100}% has embedding"
            )
            # 计算关键词的质心, Frequency作为权重
            topic_centroid[topic] = get_weighted_keywords_centroid(
                tf_idf["embedding"].tolist(), tf_idf[count_column].tolist()
            )

        # 计算主题之间的语义相似度
        topic_id_list = list(topic_centroid.keys())
        for topic_id1, topic_id2 in combinations(topic_id_list, 2):
            embedding1 = topic_centroid[topic_id1]
            embedding2 = topic_centroid[topic_id2]
            similarity = cosine_similarity_calculator(embedding1, embedding2)
            data.append(
                {
                    "topic_id1": topic_id1,
                    "topic_id2": topic_id2,
                    "year": year,
                    "similarity": similarity,
                }
            )

    # Convert the data to a DataFrame
    df = pd.DataFrame(data)
    # Save the DataFrame to a CSV file
    df.to_csv(
        EMBEDDING_DIR / f"dynamic_embedding_{src_type}_{embedding_type}.csv", index=False
    )


def process_all(embedding_type=None, src_types=None):
    """Run process_embedding across all known src_types and/or embedding modes.

    Params:
    - embedding_type: 'gpt', 'dictionary', or None (runs both).
    - src_types: comma-separated string or list. Defaults to all SRC_TOPICS keys.

    Output files land at embedding/dynamic_embedding_<src_type>_<embedding_type>.csv
    (one per combination) — no manual renaming needed.
    """
    if embedding_type is None:
        modes = ["gpt", "dictionary"]
    else:
        modes = [embedding_type]
    if src_types is None:
        chosen_srcs = list(SRC_TOPICS.keys())
    elif isinstance(src_types, str):
        chosen_srcs = [s.strip() for s in src_types.split(",") if s.strip()]
    else:
        chosen_srcs = list(src_types)

    for et in modes:
        for st in chosen_srcs:
            print(f"\n=== process_embedding(src_type={st}, embedding_type={et}) ===\n")
            try:
                process_embedding(src_type=st, embedding_type=et)
            except Exception as exc:
                print(f"!! Failed {st}/{et}: {exc!r}")


def sample_tf_idf(src_type):
    # column: topics, rows: years, output the first 10 words
    topic_dict = {}
    if src_type == "weibo":
        opinion_df = pd.read_csv(REFERENCE_DIR / "weibo_topics.csv")
        opinion_df = opinion_df[["topic_id", "topic"]].copy()
        # topic_id处理为str
        opinion_df["topic_id"] = opinion_df["topic_id"].astype(str)
        # 转为topic_id: topic的字典
        topic_dict = dict(zip(opinion_df["topic_id"], opinion_df["topic"]))
        del opinion_df
    else:
        topic_dict = {
            "gun": "Gun Control",
            "abo": "Abortion",
            "sxo": "Sexual Orientation",
            "clc": "Climate Change",
        }

    data = []
    topic_list = weibo_topic_list if src_type == "weibo" else twitter_topic_list
    word_column = "Noun" if src_type == "weibo" else "word"
    count_column = "Frequency" if src_type == "weibo" else "count"
    for year in [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, "merged"]:
        # Load the data for the specific year
        year_data = {"year": year}
        for topic in tqdm(topic_list):
            tf_idf_path = (
                TF_IDF_DIR / "tf_idf_weibo" / f"{topic}_{year}_tf_idf.parquet"
                if src_type == "weibo"
                else TF_IDF_DIR / "tf_idf_twitter" / f"tf-idf-{topic}-{year}.parquet"
            )
            if not os.path.exists(tf_idf_path):
                print(f"File {tf_idf_path} does not exist. Skipping...")
                year_data[topic_dict[topic]] = None
                continue
            tf_idf = pd.read_parquet(tf_idf_path)
            # 按照TF-IDF排序取前100
            tf_idf = tf_idf.sort_values(by="TF-IDF", ascending=False).head(10)

            year_data[topic_dict[topic]] = ";\n".join(
                tf_idf.apply(
                    lambda x: f"{x[word_column]}({x[count_column]})", axis=1
                ).tolist()
            )
        data.append(year_data)

    # Convert the data to a DataFrame
    df = pd.DataFrame(data)
    # Save the DataFrame to a CSV file
    df.to_csv(EMBEDDING_DIR / f"{src_type}_topic_keywords_sample.csv", index=False)


if __name__ == "__main__":
    if fire is None:
        raise ImportError(
            "fire is required for CLI usage. Install with `pip install fire`, "
            "or import this module and call functions directly."
        )
    fire.Fire(
        {
            "process": process_embedding,
            "process_all": process_all,
            "sample": sample_tf_idf,
            "survey_select": select_survey_keywords,
            "survey": process_survey_embedding,
        }
    )
