#!/bin/bash
# Full text-only run for LongVU on the debiased 3667-question set (longvu env).
#SBATCH -J longvu_text_only
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem-per-cpu=8G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C gmem48
#SBATCH -t 0-12:00:00
#SBATCH -o logs/longvu_text_%j.out
#SBATCH -e logs/longvu_text_%j.err

set -eo pipefail
mkdir -p logs
source /share/apps/anaconda3-2019.03-1/etc/profile.d/conda.sh
conda activate longvu
cd /home/ab260989/gen-reid
export PYTHONPATH=/home/ab260989/third_party/LongVU:$PYTHONPATH
export PYTHONIOENCODING=utf-8

MODELS=${MODELS:-"longvu-qwen2-7b"}
BENCHMARK=${BENCHMARK:-/home/ab260989/gen-reid/combined_all_hard_v3.json}
OUTPUT_DIR=${OUTPUT_DIR:-/home/ab260989/gen-reid/results_text_only_v2}

stdbuf -oL python eval_longvu_text_only.py \
    --benchmark "$BENCHMARK" \
    --output_dir "$OUTPUT_DIR" \
    --models $MODELS
