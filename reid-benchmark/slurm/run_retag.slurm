#!/bin/bash
# LLM re-tagging of Re-ID challenge categories (Gemma-3-12B-it classifier).
# Submit from the reid-benchmark/ directory:  sbatch slurm/run_retag.slurm
#SBATCH -J reid_retag
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem-per-cpu=8G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C gmem48
#SBATCH -t 0-02:00:00
#SBATCH -o logs/retag_%j.out
#SBATCH -e logs/retag_%j.err

set -eo pipefail
mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate reid

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8

MODEL=${MODEL:-google/gemma-3-12b-it}
BATCH=${BATCH:-8}
BENCH=${BENCH:-data/benchmark_debiased.json}

stdbuf -oL python src/llm_retag_challenges.py \
    --bench "$BENCH" \
    --model "$MODEL" \
    --batch_size "$BATCH" \
    --out_tags results/reid_retags.json \
    --out_bench data/benchmark_debiased_retagged.json
