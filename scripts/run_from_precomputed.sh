#!/usr/bin/env bash
# Rerun every analysis, figure and paper number from the precomputed inputs in
# data/ (cleaned opinions, pairwise correlations with shuffle tests, TF-IDF
# keyword tables, cached embeddings, social-media csr matrices).
# No raw data, GPU or API access is needed.
#
#   bash scripts/run_from_precomputed.sh          # everything
#   STEPS="semantic plots" bash scripts/run_from_precomputed.sh
#
# Steps: semantic dimension subspace plots robustness
# Outputs: data/ (intermediate tables) and outputs/{figures,stats,reports}/<YYYYMMDD>_*
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
STEPS=${STEPS:-"semantic dimension subspace plots robustness"}
has() { [[ " $STEPS " == *" $1 "* ]]; }
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }

if has semantic; then
  step "semantic: survey keywords (topic-name anchors + review) -> data/tf_idf/survey_*_keywords_selected.csv"
  $PY -m analysis.semantic.semantic_similarity survey_select
  step "semantic: survey topic similarities (selected / curated keywords) -> data/embedding/<survey>_topic_distance.csv"
  $PY -m analysis.semantic.semantic_similarity survey --embedding_type gpt --keywords selected
  $PY -m analysis.semantic.semantic_similarity survey --embedding_type gpt --keywords curated
  step "semantic: social-media topic similarities -> data/embedding/dynamic_embedding_<src>_gpt.csv"
  $PY -m analysis.semantic.semantic_similarity process_all --embedding_type gpt
fi

if has dimension; then
  step "dimension: survey csr matrices from data/survey/*.dta"
  $PY -m analysis.dimension.dimension_pipeline transform
  step "dimension: PR / eRank / srank for every csr folder"
  $PY -m analysis.dimension.dimension_pipeline calculate_all
  step "dimension: bootstrap intervals"
  $PY -m analysis.dimension.spectral_bootstrap all
fi

if has subspace; then
  step "subspace: topic centroids, ngram SVD, expressibility, SI centroid rank check"
  $PY -m robustness.embedding_subspace_analysis centroids
  $PY -m robustness.embedding_subspace_analysis svd
  $PY -m robustness.embedding_subspace_analysis expressibility
  $PY -m robustness.centroid_rank_check --plot
fi

if has plots; then
  step "plots: word-overlap panel data, main figures + stats, graphical abstract"
  $PY -m plotting.prepare_word_overlap compute
  $PY -m plotting.figures main
  $PY -m plotting.graphical_abstract
fi

if has robustness; then
  step "robustness: Fig 2 / Fig 3 variants + stats"
  $PY -m robustness.run_robustness
fi

step "done"
