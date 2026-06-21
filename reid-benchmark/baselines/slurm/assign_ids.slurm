#!/bin/bash
#SBATCH -J assign
#SBATCH -n 1 -c 8 --mem-per-cpu=8G -p gpu --gres=gpu:1 -C gmem48 -t 06:00:00
#SBATCH -o logs/assign_%x_%A_%a.out -e logs/assign_%x_%A_%a.err
# args: $1=reid  $2=tracker  $3=nshards  $4=python  $5=thresh
cd /home/ab260989/gen-reid
export HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
PY="${4:-/home/ab260989/.conda/envs/track/bin/python}"
TH="${5:-0.30}"
"$PY" assign_ids.py --reid "$1" --tracks_dir results/tracks_"$2" \
  --out results/ids_"$1"__"$2" --shard ${SLURM_ARRAY_TASK_ID:-0} --nshards "$3" --thresh "$TH"
