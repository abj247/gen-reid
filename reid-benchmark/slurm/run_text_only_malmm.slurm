#!/bin/bash
# Full text-only run for MA-LMM (vicuna7b) on the debiased 3667-question set.
#SBATCH -J malmm_text_only
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem-per-cpu=8G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C gmem48
#SBATCH -t 0-06:00:00
#SBATCH -o logs/malmm_text_%j.out
#SBATCH -e logs/malmm_text_%j.err

set -eo pipefail
mkdir -p logs
source /share/apps/anaconda3-2019.03-1/etc/profile.d/conda.sh
conda activate ma-lmm
cd /home/ab260989/gen-reid
export PYTHONPATH=/home/ab260989/third_party/MA-LMM:$PYTHONPATH
export PYTHONIOENCODING=utf-8

MODELS=${MODELS:-"ma-lmm-vicuna7b"}
BENCHMARK=${BENCHMARK:-/home/ab260989/gen-reid/combined_all_hard_v3.json}
OUTPUT_DIR=${OUTPUT_DIR:-/home/ab260989/gen-reid/results_text_only_v2}

stdbuf -oL python eval_malmm_text_only.py \
    --benchmark "$BENCHMARK" \
    --output_dir "$OUTPUT_DIR" \
    --models $MODELS
