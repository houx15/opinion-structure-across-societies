"""Rank survey topic keywords by TF-IDF from the codebook question text.

Mirrors the social-media TF-IDF step (tfidf_twitter.py / tfidf_weibo.py):
* candidates: lemmatized nouns, verbs and adjectives (length > 1, no stop
  words) plus noun-compound phrases such as "death penalty";
* Frequency = count of the keyword in the topic's question text;
* TF = Frequency / total Frequency of the topic;
* IDF = log((T + 1) / (df + 1)), T = number of topics in the survey,
  df = topics whose questions contain the keyword;
* TF-IDF = TF * IDF.

Writes one candidate table per survey region,
data/tf_idf/survey_<region>_topic_tfidf_candidates.csv (columns: topic, word, pos,
ngram, Frequency, df, TF, IDF, TF-IDF), sorted by TF-IDF within each topic.
semantic_similarity.select_survey_keywords then keeps topic-specific
keywords automatically; a hand-curated copy
(data/tf_idf/survey_<region>_topic_tfidf_curated.csv) is a robustness variant.

Run from the repository root (input: data/reference/survey_codebook.xlsx):

    python -m analysis.semantic.survey_codebook_tfidf
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from common.paths import REFERENCE_DIR, TF_IDF_DIR


SHEETS = {"US": "us", "Europe": "europe", "China": "china"}

# Words glued together in the codebook question text (PDF extraction).
CODEBOOK_GLUE_FIXES = {
    "afederal": "a federal", "allowingtransgender": "allowing transgender",
    "amountmore": "amount more", "oftheir": "of their",
    "placeyour": "place your", "theywere": "they were",
    "servicesor": "services or", "businesshow": "business how",
    "dealof": "deal of", "ordo": "or do", "theyare": "they are",
    "withyour": "with your", "permittedto": "permitted to",
    "armedforces": "armed forces", "bepaid": "be paid",
    "currentlydoing": "currently doing", "eachof": "each of",
    "forpeople": "for people", "identifythemselves": "identify themselves",
    "increasedgovernment": "increased government",
    "lesbiansagainst": "lesbians against", "meansnever": "means never",
    "mewhether": "me whether", "moreabout": "more about",
    "moredifficult": "more difficult", "neverbe": "never be",
    "nodifference": "no difference", "otherprivate": "other private",
    "protectenvironment": "protect environment",
    "provideservices": "provide services", "recentyears": "recent years",
    "relatedservices": "related services", "relationshipdo": "relationship do",
    "requiringbackground": "requiring background",
    "requiringchildren": "requiring children", "sexcouples": "sex couples",
    "thenews": "the news", "thenumber": "the number", "therisks": "the risks",
    "thesupreme": "the supreme", "theunited": "the united",
    "thinkhomosexuality": "think homosexuality", "thiscountry": "this country",
    "youthought": "you thought",
}


def _nltk():
    import nltk

    for resource, path in [
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)
    return nltk


def clean_question(text: str) -> str:
    """Lower-case the question and split words glued during PDF extraction."""
    tokens = re.findall(r"[a-z]+", str(text).lower())
    return " ".join(CODEBOOK_GLUE_FIXES.get(t, t) for t in tokens)


def question_segments(text: str):
    """Split a question at punctuation so phrases never span two clauses."""
    for seg in re.split(r"[^A-Za-z\s]+|\n", str(text)):
        seg = clean_question(seg)
        if seg:
            yield seg


# WordNet would turn these into a different word ("media" -> "medium").
_NO_LEMMA = {"media", "news", "data"}


class _Lemmatizer:
    def __init__(self, nltk):
        self._wn = nltk.stem.WordNetLemmatizer()

    def __call__(self, word: str, tag: str) -> str:
        if word in _NO_LEMMA:
            return word
        pos = {"NN": "n", "VB": "v", "JJ": "a"}[tag[:2]]
        return self._wn.lemmatize(word, pos)


def keyword_counts(text: str, nltk, stop_words, lemmatize) -> Counter:
    """Frequency of candidate keywords in ``text``.

    Unigrams: lemmatized nouns, verbs and adjectives (length > 1, no stop
    words). Phrases: noun compounds, i.e. runs of 2-3 adjectives/nouns ending
    in a noun with no stop word inside (e.g. "death penalty", "health care
    system"), head noun lemmatized. Both kinds are counted independently.
    """
    counts: Counter = Counter()
    pos_of = {}

    def ok(word, tag):
        return len(word) > 1 and word not in stop_words and tag[:2] in ("NN", "VB", "JJ")

    for segment in question_segments(text):
        tagged = nltk.pos_tag(segment.split())
        for word, tag in tagged:
            if ok(word, tag):
                lemma = lemmatize(word, tag)
                counts[lemma] += 1
                pos_of.setdefault(lemma, tag[:2])

        for n in (2, 3):
            for i in range(len(tagged) - n + 1):
                span = tagged[i:i + n]
                if not all(ok(w, t) and t[:2] in ("NN", "JJ") for w, t in span):
                    continue
                if not span[-1][1].startswith("NN"):
                    continue
                phrase = " ".join([w for w, _ in span[:-1]] + [lemmatize(*span[-1])])
                counts[phrase] += 1
                pos_of.setdefault(phrase, "NP")
    counts.pos_of = pos_of
    return counts


def topic_tfidf_table(codebook: pd.DataFrame, nltk, stop_words) -> pd.DataFrame:
    codebook = codebook.copy()
    codebook["Topic"] = codebook["Topic"].astype(str).str.strip()
    texts = codebook.groupby("Topic", sort=False)["Question"].apply(
        lambda s: " ".join(s.fillna("").astype(str))
    )
    lemmatize = _Lemmatizer(nltk)
    counts = {topic: keyword_counts(text, nltk, stop_words, lemmatize)
              for topic, text in texts.items()}
    n_topics = len(counts)
    df_count = Counter(w for c in counts.values() for w in c)

    rows = []
    for topic, c in counts.items():
        total = sum(c.values())
        for word, freq in c.items():
            tf = freq / total
            idf = np.log((n_topics + 1) / (df_count[word] + 1))
            rows.append({"topic": topic, "word": word, "pos": c.pos_of[word],
                         "ngram": len(word.split()), "Frequency": freq, "df": df_count[word],
                         "TF": tf, "IDF": idf, "TF-IDF": tf * idf})
    out = pd.DataFrame(rows)
    order = {t: i for i, t in enumerate(counts)}
    return (out.assign(_t=out["topic"].map(order))
               .sort_values(["_t", "TF-IDF", "Frequency", "word"], ascending=[True, False, False, True])
               .drop(columns="_t").reset_index(drop=True))


def main(input_file: str, output_dir: str) -> None:
    nltk = _nltk()
    stop_words = set(nltk.corpus.stopwords.words("english"))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sheet, region in SHEETS.items():
        table = topic_tfidf_table(pd.read_excel(input_file, sheet_name=sheet), nltk, stop_words)
        path = out_dir / f"survey_{region}_topic_tfidf_candidates.csv"
        table.to_csv(path, index=False)
        print(f"Saved {path}: {table['topic'].nunique()} topics, {len(table)} words")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default=str(REFERENCE_DIR / "survey_codebook.xlsx"))
    parser.add_argument("--output-dir", default=str(TF_IDF_DIR))
    args = parser.parse_args()
    main(args.input, args.output_dir)
