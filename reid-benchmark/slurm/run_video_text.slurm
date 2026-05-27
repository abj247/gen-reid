#!/bin/bash
# Video+text evaluation across the model matrix on the debiased benchmark.
# Submit from the reid-benchmark/ directory:  sbatch slurm/run_video_text.slurm
#SBATCH -J reid_video_text
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem-per-cpu=8G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C gmem48
#SBATCH -t 2-00:00:00
#SBATCH -o logs/video_text_%j.out
#SBATCH -e logs/video_text_%j.err

set -eo pipefail
mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate reid

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8

BENCHMARK=${BENCHMARK:-data/benchmark_debiased_real_ids.json}
VIDEO_DIR=${VIDEO_DIR:-/home/c3-0/datasets/moviechat1k-test}
OUTPUT_DIR=${OUTPUT_DIR:-results/video_text}
MODELS=${MODELS:-}
NUM_FRAMES=${NUM_FRAMES:-8}
LIMIT_VIDEOS=${LIMIT_VIDEOS:-}
FORCE=${FORCE:-}

EXTRA=""
[ -n "$MODELS" ]       && EXTRA="$EXTRA --models $MODELS"
[ -n "$LIMIT_VIDEOS" ] && EXTRA="$EXTRA --limit_videos $LIMIT_VIDEOS"
[ "$FORCE" = "1" ]     && EXTRA="$EXTRA --force"

stdbuf -oL python src/eval_video_text.py \
    --benchmark "$BENCHMARK" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_frames "$NUM_FRAMES" \
    $EXTRA
