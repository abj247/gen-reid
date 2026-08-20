#!/bin/bash
# CLIP-keyframe arms + their two CONTROLS, via eval_conditioned.py.
#
#   MODEL=internvl3-14b CLIPS=conditioned_keyframes/random   PIPE=kf_random   sbatch run_kf_eval.slurm
#   MODEL=ovis2.5-9b    CLIPS=conditioned_keyframes/uniform8 PIPE=kf_uniform8 sbatch run_kf_eval.slurm
#
# ENV: videochat-flash-{2b,7b} need the `videochat-flash` interpreter, everything else uses `reid`
# (eval_conditioned.py's own docstring says so). Set ENVN=videochat-flash for those.
#
# WHY THESE TWO CONTROLS EXIST. The kf arms were only ever compared against the published uniform-8
# row, and that comparison cannot separate three explanations:
#   (1) query-conditioned frames are better        -> isolated by kf_random
#   (2) the mp4v/4fps render pipeline differs      -> isolated by kf_uniform8
#   (3) genuine method gain
# This project has already watched a question-conditioned selector LOSE to its own random control
# (qcond 28.89 vs qrand 29.31), so (1) is a live risk, not a formality.
#
#SBATCH -J kfeval -n 1 -c 8 --mem-per-cpu=8G -p gpu --gres=gpu:1 -C gmem48 -t 1-00:00:00
#SBATCH -o logs/kf_%x_%j.out -e logs/kf_%x_%j.err
set -eo pipefail; mkdir -p logs
source /share/apps/anaconda3-2019.03-1/etc/profile.d/conda.sh; conda activate "${ENVN:-reid}"
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# CVD guard: some nodes (c2-2) export an invalid CUDA_VISIBLE_DEVICES
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9,]*$ ]] || export CUDA_VISIBLE_DEVICES=0
REPO="${PERSISTQA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO"

MODEL="${MODEL:?set MODEL}"
CLIPS="${CLIPS:?set CLIPS}"
PIPE="${PIPE:?set PIPE}"

echo "=== $SLURM_JOB_ID on $SLURMD_NODENAME | $MODEL | $CLIPS -> results_baseline/$PIPE ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
test -f "$CLIPS/manifest.json" || { echo "no manifest at $CLIPS" >&2; exit 2; }
echo "[clips] $(python -c "import json;print(len(json.load(open('$CLIPS/manifest.json'))))") in manifest"

stdbuf -oL python eval_conditioned.py --model "$MODEL" --clips "$CLIPS" --pipeline "$PIPE" ${LIMIT:+--limit $LIMIT}
echo "=== exit $? at $(date) ==="
