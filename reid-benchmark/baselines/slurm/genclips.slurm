#!/bin/bash
#SBATCH -J clips
#SBATCH -n 1 -c 8 --mem-per-cpu=8G -p gpu --gres=gpu:1 -C gmem48 -t 06:00:00
#SBATCH -o logs/clips_%x_%j.out -e logs/clips_%x_%j.err
cd /home/ab260989/gen-reid
export HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
/home/ab260989/.conda/envs/reid/bin/python gen_conditioned_clips.py --tracks_dir results/tracks_"$1" --out conditioned/"$1"
