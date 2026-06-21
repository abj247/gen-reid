#!/bin/bash
#SBATCH -J evlongvu
#SBATCH -n 1 -c 8 --mem-per-cpu=8G -p gpu --gres=gpu:1 -C gmem48 -t 12:00:00
#SBATCH -o logs/eval_longvu_%x_%j.out -e logs/eval_longvu_%x_%j.err
source /share/apps/anaconda3-2019.03-1/etc/profile.d/conda.sh; conda activate longvu
cd /home/ab260989/gen-reid
export PYTHONPATH=/home/ab260989/third_party/LongVU:$PYTHONPATH HF_HUB_OFFLINE=1
python eval_longvu_conditioned.py --manifest conditioned/"$1"/manifest.json --output_dir results_baseline/"$1"
