#!/bin/bash
#SBATCH -J ev
#SBATCH -n 1 -c 8 --mem-per-cpu=8G -p gpu --gres=gpu:1 -C gmem48 -t 12:00:00
#SBATCH -o logs/eval_%x_%j.out -e logs/eval_%x_%j.err
cd /home/ab260989/gen-reid
export HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
"$3" eval_conditioned.py --model "$1" --clips conditioned/"$2" --pipeline "$2"
