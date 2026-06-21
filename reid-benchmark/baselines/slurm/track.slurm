#!/bin/bash
#SBATCH -J trk
#SBATCH -n 1 -c 8 --mem-per-cpu=8G -p gpu --gres=gpu:1 -C gmem48 -t 06:00:00
#SBATCH -o logs/track_%x_%A_%a.out -e logs/track_%x_%A_%a.err
cd /home/ab260989/gen-reid
export HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
/home/ab260989/.conda/envs/track/bin/python track_all.py --tracker "$1" --shard ${SLURM_ARRAY_TASK_ID:-0} --nshards "$2"
