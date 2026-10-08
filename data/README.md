# data/

Everything here except `reference/` is git-ignored. `scripts/import_legacy_data.py`
fills this folder from the original working directory. File formats are the
ones the original pipeline wrote; nothing was converted.

`<src>` names: surveys `anes`, `anes_media`, `wvs`, `wvs_media`, `evs`,
`evs_media`, `evs_resample2`, `evs_media_resample2`, `en_evs`, `en_evs_media`,
`eu_nen_evs`; social media `twitter` (US), `eutwitter` (Europe), `entwitter`
(Great Britain), `eu_nentwitter` (Europe without GB), `weibo`.
Main analysis: `anes`, `wvs`, `evs_resample2`, `twitter`, `weibo`, `eutwitter`.

| folder | files | produced by | precomputed? |
|---|---|---|---|
| `reference/` | `survey_codebook.xlsx` (question text per survey topic), `weibo_topics.csv`, `twitter_topics.csv` (topic definitions, opinion scales, keywords) | by hand | tracked in git |
| `survey/` | `<survey>.dta`: one row per respondent, the 9 topic items (1-5), `country`, `year` | survey recoding (outside this repo) | input |
| `opinions/` | `individual_opinion_<survey>.parquet` (respondent_id, year, 9 topics); `user_opinion_<social>_lgbt_env.parquet` (user_id, LGBT, Environment) | `cleaning.survey.prepare_survey_individual_data`, `cleaning.twitter.opinion.export_user_opinion_data` | yes (social) |
| `correlation/` | `network_analysis_<src>.csv`: year, Topic1, Topic2, (tid1, tid2), Intersection, Correlation (Pearson), P-Value (Pearson), shuffled_pearson, shuffled_pearson_std (social media only) | `analysis.correlation.*` | yes |
| `tf_idf/` | `tf_idf_<social>/tf-idf-<topic>-<year>.parquet` (word, count, TF, IDF, TF-IDF; Weibo: `<topic>_<year>_tf_idf.parquet` with Noun, Frequency); `survey_<region>_topic_tfidf_candidates.csv`, `_curated.csv` (hand-curated), `_keywords_selected.csv` | `analysis.semantic.tfidf_*`, `survey_codebook_tfidf`, `semantic_similarity survey_select` | yes (social TF-IDF, curated lists) |
| `embedding/` | `cached_embedding.pkl` (word -> 1536-d OpenAI text-embedding-3-small vector); `<survey>_topic_distance.csv` (topic_id1, topic_id2, similarity, similarity_curated, similarity_question, dictionary_similarity); `dynamic_embedding_<social>_<gpt|dictionary>.csv` (topic_id1, topic_id2, year, similarity); `backup_full_question/` (similarities from full question text); `topic_centroids/`; `word_overlap_rows.json` | `analysis.semantic.semantic_similarity`, `robustness.embedding_subspace_analysis centroids`, `plotting.prepare_word_overlap` | yes (cache, full-question similarities) |
| `dimension/csr/<stem>/` | `<year>.csr.npz`: respondents/users x 9 topics, missing = absent | `analysis.dimension.dimension_pipeline transform` | yes (social: `twitter-us`, `twitter-eu`, `twitter-en`, `twitter-eu_nen`, `weibo`) |
| `dimension/results/` | `<stem>-<year>.csr-none.json`, `-loadings.json`, `<stem>-none-summary.json` (PR, eRank, srank), `<stem>-none-bootstrap.json` | `dimension_pipeline calculate`, `spectral_bootstrap` | no (recomputed from the csr matrices) |

Dimension stems are the survey names and `twitter-<us|eu|en|eu_nen>`, `weibo`.

Raw data used by the cleaning stage stay on the clusters (paths in
`config/config.py`).
