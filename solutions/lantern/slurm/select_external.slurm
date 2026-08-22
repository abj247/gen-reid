#!/bin/bash
# Frame selection for an external benchmark (MLVU or LongVideoBench).
#
#   BENCH=benchmark/data/external/mlvu.json MODE=question OUT=external_clips/mlvu_lantern \
#     sbatch solutions/lantern/slurm/select_external.slurm
#
# Two things differ from the PersistQA run and both matter.
#
# External benchmarks have no temporal anchor, so Lantern runs in `question` mode there
# rather than `referent`. That is a genuine limitation of taking the method off PersistQA
# and is reported, not papered over.
#
# MLVU videos run to two hours, against three minutes on PersistQA, so this is decode
# bound rather than GPU bound. The wall clock is generous for that reason.
#
#SBATCH -J extsel -n 1 -c 8 --mem-per-cpu=8G -p gpu --gres=gpu:1 -C gmem48 -t 1-00:00:00
#SBATCH -o logs/extsel_%x_%j.out -e logs/extsel_%x_%j.err
set -eo pipefail; mkdir -p logs
source /share/apps/anaconda3-2019.03-1/etc/profile.d/conda.sh; conda activate reid
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9,]*$ ]] || export CUDA_VISIBLE_DEVICES=0
# SLURM copies the batch script into /var/spool before running it, so BASH_SOURCE
# points at the spooled copy and not at the repository. SLURM_SUBMIT_DIR is the
# directory sbatch was invoked from, which is the reliable anchor here.
REPO="${PERSISTQA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -d "$REPO/solutions" ]; then
  echo "cannot locate the repository root (tried PERSISTQA_ROOT, SLURM_SUBMIT_DIR, PWD)" >&2
  exit 2
fi
cd "$REPO"; export PYTHONPATH="$REPO:$PYTHONPATH"

BENCH="${BENCH:?set BENCH}"; MODE="${MODE:?set MODE}"; OUT="${OUT:?set OUT}"
echo "=== $SLURM_JOB_ID on $SLURMD_NODENAME | $(basename $BENCH) | mode=$MODE -> $OUT ==="
stdbuf -oL python solutions/lantern/select_frames.py --bench "$BENCH" --query_mode "$MODE" --out "$OUT"
echo "=== exit $? at $(date) ==="
