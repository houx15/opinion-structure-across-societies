# Opinion structure across societies

Code for comparing how opinions on nine policy topics hang together in
national surveys and on social media in the United States, Europe and China:

| region | survey | social media |
|---|---|---|
| US | ANES | Twitter, users located in the US |
| Europe | EVS (resampled with replacement; `evs` = without) | Twitter, users located in Europe |
| China | WVS | Weibo |

For each source we measure (1) the pairwise correlation of opinions across
topics, (2) the semantic similarity of the topics (OpenAI embeddings of topic
keywords) and how much of the correlation it explains, and (3) the effective
dimensionality of the respondent x topic opinion matrix.

## Repository layout

```
config/       config_example.py -> copy to config/config.py (git-ignored): paths, API keys
common/       paths.py: every data/output location (single source of truth)
cleaning/     stage 1  raw data -> cleaned opinions          (clusters; see cleaning/README.md)
analysis/     stage 2  cleaned opinions -> correlations, semantic similarities, dimensionality
  correlation/   pairwise correlations + shuffle baselines
  semantic/      topic keywords (TF-IDF), embeddings, topic similarities
  dimension/     csr matrices, PR / eRank / srank, bootstrap intervals
plotting/     stage 3  figures, paper statistics, graphical abstract
robustness/   stage 4  robustness variants and SI analyses
scripts/      run_from_precomputed.sh, import_legacy_data.py
data/         inputs and intermediate tables (git-ignored, see data/README.md)
outputs/      figures/, stats/, reports/ - every file starts with <YYYYMMDD>_ (git-ignored)
tests/
```

Run every module from the repository root, e.g. `python -m plotting.figures main`.

## Setup

```bash
uv sync                    # or: pip install -e .  (Python >= 3.10)
cp config/config_example.py config/config.py   # only needed for cleaning / new embeddings
```

Optional extras: `uv sync --extra openai` (embed words missing from the cache),
`uv sync --extra word2vec` (word2vec robustness check). The cleaning stage has
its own requirements (`cleaning/requirements.txt`, `cleaning/bert/requirements.txt`).

## Reproducing the results from precomputed inputs

The expensive steps (LLM labelling, BERT fine-tuning and prediction on all
keyword-matched posts, pairwise correlations with 1,000-10,000 shuffles, TF-IDF over all
posts, OpenAI embeddings) are run once; their outputs are the inputs listed in
data/README.md. With those in `data/`:

```bash
bash scripts/run_from_precomputed.sh            # a few minutes on a laptop
STEPS="plots robustness" bash scripts/run_from_precomputed.sh
```

| step | command | writes |
|---|---|---|
| semantic | `analysis.semantic.semantic_similarity survey_select / survey / process_all` | survey keyword selection, topic similarities |
| dimension | `analysis.dimension.dimension_pipeline transform / calculate_all`, `spectral_bootstrap all` | PR / eRank per source, bootstrap CIs |
| subspace | `robustness.embedding_subspace_analysis`, `robustness.centroid_rank_check --plot` | SI semantic-subspace analyses |
| plots | `plotting.prepare_word_overlap`, `plotting.figures main`, `plotting.graphical_abstract` | Fig 2, Fig 3, stats, GA panels |
| robustness | `robustness.run_robustness` | robustness figures + stats |

If you have the original working folder of the project, copy its precomputed
files into this layout with

```bash
python scripts/import_legacy_data.py /path/to/opinion_correlation [--csr_root /path/to/dimension/data]
```

## Full pipeline

1. **Cleaning** (`cleaning/`, see cleaning/README.md): survey recoding;
   Twitter user location (rule-based + LLM), keyword-matched posts, a 10,000-post
   sample per topic labelled by three LLMs, BERT relevance and opinion models
   fine-tuned on the agreed labels, user x year opinions. Output: cleaned
   opinions per user / respondent.
2. **Analysis** (`analysis/`):
   - `correlation/`: `survey_correlation all`, `twitter_correlation all`,
     `weibo_correlation` -> `data/correlation/network_analysis_<src>.csv`
     (Pearson r per topic pair and year, plus the shuffle baseline for social media).
   - `semantic/`: `tfidf_twitter`, `tfidf_weibo` (noun/verb TF-IDF per topic),
     `survey_codebook_tfidf` (codebook keywords), `semantic_similarity`
     (keyword embeddings -> topic centroids -> cosine similarity).
   - `dimension/`: `dimension_pipeline transform` (opinions -> csr matrices),
     `calculate_all`, `spectral_bootstrap all`.
3. **Plotting** (`plotting/`): `figures`, `paper_stats` (plain-text numbers
   for the paper, written with every figure), `graphical_abstract`.
4. **Robustness** (`robustness/`): `run_robustness` (survey media users only,
   EVS without replacement, Great Britain only, LOWESS span, word2vec,
   GB vs. rest of Europe, curated / full-question survey keywords),
   `run_word2vec.sh`, `embedding_subspace_analysis`, `centroid_rank_check`,
   `missingness_sensitivity` (to do).

## Outputs

`outputs/figures/<YYYYMMDD>_<task>_results_<fig>.pdf`,
`outputs/stats/<YYYYMMDD>_<task>_results_stats.txt` (every number quoted in the
paper, with its source files and calculation), `outputs/reports/<YYYYMMDD>_<analysis>/`
(tables behind SI analyses). Tasks: `main` (main text), `supp_*` (supplement),
`robust_*` (robustness). See outputs/README.md.

## Data availability

Survey microdata are available from ANES, EVS and WVS under their terms of
use. Raw tweets and Weibo posts cannot be redistributed; `data/reference/`
holds the topic definitions, keywords and the survey codebook used here.
