"""Robustness checks that swap one input of the main figures.

Each check re-draws Fig 2 (and Fig 3 where the respondents change) with the
plotting code of the main analysis; only the data sources differ. All inputs
are produced by the analysis stage (see the "inputs" column) and
the task definitions live in plotting/figures.py (TASKS).

| task                       | what changes                                           | inputs                                           |
|----------------------------|--------------------------------------------------------|--------------------------------------------------|
| robust_1_media             | survey respondents who use social media                | anes_media, wvs_media, en_evs_media              |
| robust_5_word2vec          | word2vec instead of OpenAI embeddings                  | dictionary similarities (run_word2vec.sh)        |
| robust_7_survey_curated    | hand-curated survey keywords                           | similarity_curated column                        |
| robust_8_survey_question   | survey topic = embedding of the full question text     | similarity_question column                       |
| robust_9_correlation       | Fig 3 from the correlation instead of covariance matrix | data/dimension/results/corr/ (calculate_all / spectral_bootstrap --matrix corr) |
| robust_10_annual           | Figs 2, 3 from social-media opinions of 2021 instead of pooled over years | year == 2021 rows of the correlation files; <stem>-2021.csr-none.json (+ spectral_bootstrap run weibo --year 2021) |
| robust_11_survey_selected  | survey keywords before the top-5 cap (all selected keywords) | similarity column                            |
| robust_12_wvs_us           | U.S. survey from WVS wave 7; every source on eight topics | wvs_us8 ... entwitter_8 (wvs_us.py, topics8.py) |

Other robustness analyses in this folder:
    embedding_subspace_analysis.py   ngram-level semantic subspace diagnostics
    centroid_rank_check.py           SI rank / conditioning of the 9 topic centroids
    missingness_sensitivity.py       PR / eRank vs. sample size, missingness, true dimensionality (simulation)

Usage (from the repository root):
    python -m robustness.run_robustness              # every task above
    python -m robustness.run_robustness robust_5_word2vec
"""

from typing import Optional

import fire

from plotting import figures


def main(task: Optional[str] = None, n_bootstrap: int = 2000) -> None:
    names = [task] if task else figures.task_names("robustness")
    for name in names:
        print(f"[robustness] {name}")
        figures.run_task(name, n_bootstrap=n_bootstrap)


if __name__ == "__main__":
    fire.Fire(main)
