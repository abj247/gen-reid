#!/bin/bash
# Video+text evaluation for VideoChat-Flash 2B + 7B on the debiased 3667-question
# set. Runs in the videochat-flash conda env.
#SBATCH -J vcflash_video
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem-per-cpu=8G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C gmem48
#SBATCH -t 1-12:00:00
#SBATCH -o logs/vcflash_video_%j.out
#SBATCH -e logs/vcflash_video_%j.err

set -eo pipefail
mkdir -p logs
source /share/apps/anaconda3-2019.03-1/etc/profile.d/conda.sh
conda activate videochat-flash
cd /home/ab260989/gen-reid

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8

MODELS=${MODELS:-"videochat-flash-2b videochat-flash-7b"}
BENCHMARK=${BENCHMARK:-/home/ab260989/gen-reid/combined_all_hard_v3_real_ids.json}
VIDEO_DIR=${VIDEO_DIR:-/home/c3-0/datasets/moviechat1k-test}
OUTPUT_DIR=${OUTPUT_DIR:-/home/ab260989/gen-reid/results_video_v2}

stdbuf -oL python evaluate_vlm_video_v2.py \
    --benchmark "$BENCHMARK" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --models $MODELS \
    --num_frames 8
