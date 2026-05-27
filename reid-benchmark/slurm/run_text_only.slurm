#!/bin/bash
# Text-only (no-vision) bias evaluation across the model matrix.
# Submit from the reid-benchmark/ directory:  sbatch slurm/run_text_only.slurm
#SBATCH -J reid_text_only
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem-per-cpu=8G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C gmem48
#SBATCH -t 2-00:00:00
#SBATCH -o logs/text_only_%j.out
#SBATCH -e logs/text_only_%j.err

set -eo pipefail
mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate reid

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8

BENCHMARK=${BENCHMARK:-data/benchmark_text_only.json}
OUTPUT_DIR=${OUTPUT_DIR:-results/text_only}
MODELS=${MODELS:-}
LIMIT=${LIMIT:-}
FORCE=${FORCE:-}

EXTRA=""
[ -n "$MODELS" ] && EXTRA="$EXTRA --models $MODELS"
[ -n "$LIMIT" ]  && EXTRA="$EXTRA --limit $LIMIT"
[ "$FORCE" = "1" ] && EXTRA="$EXTRA --force"

stdbuf -oL python src/eval_text_only.py \
    --benchmark "$BENCHMARK" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda \
    $EXTRA
