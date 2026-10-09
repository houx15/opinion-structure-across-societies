# outputs/

Every file starts with the run date, `<YYYYMMDD>_`. Git-ignored.

| folder | files |
|---|---|
| `figures/` | `<date>_<task>_results_fig2.pdf` / `_fig3.pdf`; `<date>_supp_centroid_rank_results_fig.pdf`; `<date>_graphical_abstract_panels/` |
| `stats/` | `<date>_<task>_results_stats.txt`: every number quoted in the paper for that task, each with its source files and calculation; `<date>_supp_centroid_rank_results_stats.txt`; `<date>_descriptive_stats.txt` |
| `reports/` | `<date>_<analysis>/`: csv tables behind SI analyses (`centroid_rank_check`, `embedding_subspace`, `dimension_loadings`, `descriptive_stats`) |

Tasks (plotting/figures.py `TASKS`):

| task | figures | content |
|---|---|---|
| `main` | Fig 2, Fig 3 | main text |
| `supp_lowess` | Fig 2 | LOWESS (span 0.4) instead of linear trends: nonlinear alternative |
| `supp_centroid_rank` | SI figure | rank / conditioning of the topic centroids |
| `robust_1_media` | Fig 2, 3 | survey respondents who use social media |
| `robust_2_noputback` | Fig 2, 3 | EVS without resampling |
| `robust_3_england` | Fig 2, 3 | Great Britain only |
| `robust_5_word2vec` | Fig 2 | word2vec topic similarities |
| `robust_6_eu_nen` | Fig 2, 3 | GB and rest of Europe separately |
| `robust_7_survey_curated` | Fig 2 | hand-curated survey keywords |
| `robust_8_survey_question` | Fig 2 | survey topics embedded from the full question text |
