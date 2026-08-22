#!/bin/bash
# Chunk memory bank: encode 8x32 frames of memory per video ONCE, then answer each question from
# the chunks whose memory CLIP ranks most similar to it.
#
#   ARMS=mb_uniform32,mb_top2,mb_rand2,mb_top1,mb_rand1,mb_oracle SUBSET=core sbatch run_membank.slurm
#
# Requires analysis3/membank/index/<vid>.npz -- build it first with build_clip_index.py (CPU is
# fine, ~70 min for the corpus).
#
# -c 8 (=64 GB host RAM): the bank is 256 frames x 256 tokens x 5120 dims held on GPU, and the
# decode side stages 32 frames of raw video at source resolution per batch.
#
#SBATCH -J membank -n 1 -c 8 --mem-per-cpu=8G -p gpu --gres=gpu:1 -C gmem48 -t 1-00:00:00
#SBATCH -o logs/mb_%x_%j.out -e logs/mb_%x_%j.err
set -eo pipefail; mkdir -p logs
source /share/apps/anaconda3-2019.03-1/etc/profile.d/conda.sh; conda activate reid
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO:$PYTHONPATH"
# CVD guard: some nodes (c2-2) export an invalid CUDA_VISIBLE_DEVICES
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9,]*$ ]] || export CUDA_VISIBLE_DEVICES=0
# SLURM copies the batch script into /var/spool before running it, so BASH_SOURCE
# points at the spooled copy and not at the repository. SLURM_SUBMIT_DIR is the
# directory sbatch was invoked from, which is the reliable anchor here.
REPO="${PERSISTQA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -d "$REPO/solutions" ]; then
  echo "cannot locate the repository root (tried PERSISTQA_ROOT, SLURM_SUBMIT_DIR, PWD)" >&2
  exit 2
fi
cd "$REPO"

ARMS="${ARMS:-mb_uniform32,mb_top2,mb_rand2,mb_top1,mb_rand1,mb_oracle}"
SUBSET="${SUBSET:-core}"
POOL="${POOL:-max}"
QMODE="${QMODE:-referent}"

echo "=== $SLURM_JOB_ID on $SLURMD_NODENAME | ARMS=$ARMS SUBSET=$SUBSET POOL=$POOL QMODE=$QMODE ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "[index] $(ls analysis3/membank/index/*.npz 2>/dev/null | wc -l) videos indexed"

stdbuf -oL python analysis3/membank/run_membank.py \
    --arms "$ARMS" --subset "$SUBSET" --pool "$POOL" --query_mode "$QMODE" \
    ${SUBSET_FILE:+--subset_file $SUBSET_FILE} ${LIMIT:+--limit $LIMIT}
echo "=== exit $? at $(date) ==="
