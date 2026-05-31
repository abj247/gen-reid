#!/bin/bash
# Full text-only run for VideoChat-Flash 2B + 7B on the debiased 3667-question set.
# Runs in the videochat-flash conda env (transformers 4.40.1 + flash_attn stub).
#SBATCH -J vcflash_text_only
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem-per-cpu=8G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C gmem48
#SBATCH -t 1-00:00:00
#SBATCH -o logs/vcflash_text_%j.out
#SBATCH -e logs/vcflash_text_%j.err

set -eo pipefail
mkdir -p logs
source /share/apps/anaconda3-2019.03-1/etc/profile.d/conda.sh
conda activate videochat-flash
cd /home/ab260989/gen-reid

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

MODELS=${MODELS:-"videochat-flash-2b videochat-flash-7b"}
BENCHMARK=${BENCHMARK:-/home/ab260989/gen-reid/combined_all_hard_v3.json}
OUTPUT_DIR=${OUTPUT_DIR:-/home/ab260989/gen-reid/results_text_only_v2}

echo "=== $(date) | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) ==="
stdbuf -oL python evaluate_vlm_text_only_v2.py \
    --benchmark "$BENCHMARK" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda \
    --models $MODELS
echo "=== done $(date), exit $? ==="
