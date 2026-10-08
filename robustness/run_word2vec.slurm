#!/bin/bash
# Slurm wrapper for robustness/run_word2vec.sh (as run on Princeton Adroit:
# 8 min, ~12 GB peak memory for the two word2vec models).
# Compute nodes have no network: install the environment (gensim, statsmodels,
# matplotlib, ...) on the login node first, and set DATA_ROOT / OUTPUT_ROOT /
# WORD2VEC_* in config/config.py.
#
#   sbatch robustness/run_word2vec.slurm          # from the repository root
#
#SBATCH --job-name=oss-word2vec
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-word2vec-%j.out

set -euo pipefail
# Python environment with the project dependencies + gensim, e.g.
#   export PATH=$HOME/.conda/envs/<env>/bin:$PATH
export MPLBACKEND=Agg
cd "${SLURM_SUBMIT_DIR:-.}"
python --version
bash robustness/run_word2vec.sh
