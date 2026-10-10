# data/

## What is in the repository

The repository ships the precomputed results needed to reproduce every figure
(`python -m plotting.figures main`, `python -m robustness.run_robustness`):

- `correlation/`: pairwise topic correlations of every source, and
  `case_study_lgbt_env.json`, the aggregate stand-in for the Fig 2 case-study
  panel (signed r, covariance of the standardized pair, and a 2-D histogram of
  the standardized points, bin width 0.1; `python -m plotting.figures export_case_study`);
- `embedding/`: topic semantic similarities (`*_topic_distance.csv`,
  `dynamic_embedding_*.csv`, `backup_full_question/`), the Fig 2 word-panel
  payload `word_overlap_rows.json`, and `topic_centroids/`;
- `tf_idf/`: social-media TF-IDF tables (word counts per topic and year) and the
  survey keyword tables;
- `dimension/results/`: effective dimensionality (PR, eRank), bootstrap
  intervals and eigenvectors for every source;
- `reference/`: topic definitions, the survey codebook, and model-performance tables.

Not in the repository:

- individual-level data (`opinions/`, `dimension/csr/`) and the recoded survey
  files (`survey/`). Social-media user-level opinions are available from the
  authors on request; the surveys are distributed by their archives (ANES,
  World Values Survey, GESIS for EVS); their recoding into the nine topic scores
  (SI Appendix) was done outside this repository, and `cleaning/survey/` builds
  the opinion files from the recoded scores;
- the word-embedding cache `embedding/cached_embedding.pkl` (89 MB), which is
  rebuilt by `analysis.semantic.semantic_similarity` from the OpenAI API.

Without these, figures are drawn from the shipped results; the only steps that
need them are recomputing correlations, dimensions or similarities from scratch,
and the survey sample sizes in the statistics report. Tests that read them are
skipped. `scripts/import_legacy_data.py` fills this folder from the original
working directory. File formats are the ones the original pipeline wrote;
nothing was converted.

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
| `correlation/` | `case_study_lgbt_env.json` (aggregate Fig 2 case-study panel, see above); `network_analysis_<src>.csv`: year, Topic1, Topic2, (tid1, tid2), Intersection, Correlation (Pearson), P-Value (Pearson), shuffled_pearson, shuffled_pearson_std (social media only) | `analysis.correlation.*` | yes |
| `tf_idf/` | `tf_idf_<social>/tf-idf-<topic>-<year>.parquet` (word, count, TF, IDF, TF-IDF; Weibo: `<topic>_<year>_tf_idf.parquet` with Noun, Frequency); `survey_<region>_topic_tfidf_candidates.csv`, `_curated.csv` (hand-curated), `_keywords_selected.csv`, `_keywords_top5.csv` (main) | `analysis.semantic.tfidf_*`, `survey_codebook_tfidf`, `semantic_similarity survey_select / survey_top5` | yes (social TF-IDF, curated lists) |
| `embedding/` | `cached_embedding.pkl` (word -> 1536-d OpenAI text-embedding-3-small vector); `<survey>_topic_distance.csv` (topic_id1, topic_id2, similarity_top5 (main), similarity (selected), similarity_curated, similarity_question, dictionary_similarity_top5, dictionary_similarity); `dynamic_embedding_<social>_<gpt|dictionary>.csv` (topic_id1, topic_id2, year, similarity); `backup_full_question/` (similarities from full question text); `topic_centroids/`; `word_overlap_rows.json` | `analysis.semantic.semantic_similarity`, `robustness.embedding_subspace_analysis centroids`, `plotting.prepare_word_overlap` | yes (cache, full-question similarities) |
| `dimension/csr/<stem>/` | `<year>.csr.npz`: respondents/users x 9 topics, missing = absent | `analysis.dimension.dimension_pipeline transform` | yes (social: `twitter-us`, `twitter-eu`, `twitter-en`, `twitter-eu_nen`, `weibo`) |
| `dimension/results/cov/`, `dimension/results/corr/` | `<stem>-<year>.csr-none.json`, `-loadings.json`, `<stem>-none-summary.json` (PR, eRank, srank), `<stem>-none-bootstrap.json`; `cov/` = pairwise-available covariance (main), `corr/` = correlation matrix (robustness), same file names | `dimension_pipeline calculate[_all] [--matrix corr]`, `spectral_bootstrap [--matrix corr]` | no (recomputed from the csr matrices) |

Dimension stems are the survey names and `twitter-<us|eu|en|eu_nen>`, `weibo`.

Raw data used by the cleaning stage stay on the clusters (paths in
`config/config.py`).
