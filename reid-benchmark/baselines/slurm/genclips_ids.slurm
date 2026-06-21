#!/bin/bash
#SBATCH -J clips_ids
#SBATCH -n 1 -c 8 --mem-per-cpu=8G -p gpu --gres=gpu:1 -C gmem48 -t 06:00:00
#SBATCH -o logs/clipsids_%x_%j.out -e logs/clipsids_%x_%j.err
# args: $1=reid  $2=tracker   (pipeline name = <reid>__<tracker>)
cd /home/ab260989/gen-reid
export HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
/home/ab260989/.conda/envs/reid/bin/python gen_conditioned_clips.py \
  --tracks_dir results/tracks_"$2" --ids_dir results/ids_"$1"__"$2" --out conditioned/"$1"__"$2"
