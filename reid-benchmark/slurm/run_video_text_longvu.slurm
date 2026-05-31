#!/bin/bash
#SBATCH -J longvu_video
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem-per-cpu=8G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C gmem48
#SBATCH -t 1-12:00:00
#SBATCH -o logs/longvu_video_%j.out
#SBATCH -e logs/longvu_video_%j.err

set -eo pipefail
mkdir -p logs
source /share/apps/anaconda3-2019.03-1/etc/profile.d/conda.sh
conda activate longvu
cd /home/ab260989/gen-reid
export PYTHONPATH=/home/ab260989/third_party/LongVU:$PYTHONPATH
export PYTHONIOENCODING=utf-8

stdbuf -oL python eval_longvu_video_text.py \
    --benchmark /home/ab260989/gen-reid/combined_all_hard_v3_real_ids.json \
    --video_dir /home/c3-0/datasets/moviechat1k-test \
    --output_dir /home/ab260989/gen-reid/results_video_v2 \
    --models longvu-qwen2-7b
