#!/usr/bin/env bash
# word2vec robustness check: topic similarities from static word vectors
# (GoogleNews word2vec for English, SGNS merge for Chinese) instead of
# OpenAI embeddings, then Fig 2 with those similarities.
#
# Needs the two word2vec files (config: WORD2VEC_ENGLISH / WORD2VEC_CHINESE)
# and gensim:  uv sync --extra word2vec
# Run from the repository root (on a cluster: sbatch robustness/run_word2vec.slurm).
set -euo pipefail

# social media: top-10 TF-IDF words per topic (same as the main analysis)
python -m analysis.semantic.semantic_similarity process_all --embedding_type dictionary
# surveys: the selected codebook keywords
python -m analysis.semantic.semantic_similarity survey --embedding_type dictionary

python -m plotting.figures task robust_5_word2vec
