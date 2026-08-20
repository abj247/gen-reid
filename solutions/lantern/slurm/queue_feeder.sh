#!/bin/bash
# Feed chunk-mode (and missing control) evals onto the cluster one at a time, never exceeding the
# 3-GPU budget.
#
# THE BUG THIS AVOIDS. The previous watcher counted `squeue -t RUNNING,PENDING` and so counted the
# four JobHeldUser pg_* jobs as mine. Its "am I under the cap" test was therefore never true and it
# waited ~24h while the GPUs sat idle. Here the count excludes BOTH the user's `bash` shell and any
# job whose reason is JobHeldUser, and the logic was verified against live squeue output before use.
REPO="${PERSISTQA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO"

mine() { squeue -u "$USER" -h -o "%j|%t|%r" | awk -F'|' '$1!="bash" && $3!~/JobHeldUser/' | wc -l; }

# model | clips dir | pipeline tag | conda env
JOBS=(
  "internvl3-14b|conditioned_keyframes/chunk|kf_chunk|reid"
  "qwen2.5-vl-7b|conditioned_keyframes/chunk|kf_chunk|reid"
  "ovis2.5-9b|conditioned_keyframes/chunk|kf_chunk|reid"
  "videochat-flash-7b|conditioned_keyframes/chunk|kf_chunk|videochat-flash"
  "videochat-flash-7b|conditioned_keyframes/random|kf_random|videochat-flash"
  "videochat-flash-7b|conditioned_keyframes/uniform8|kf_uniform8|videochat-flash"
  "internvl3-8b|conditioned_keyframes/chunk|kf_chunk|reid"
  "internvl3-8b|conditioned_keyframes/referent|kf_referent|reid"
  "internvl3-8b|conditioned_keyframes/random|kf_random|reid"
  "internvl3-8b|conditioned_keyframes/uniform8|kf_uniform8|reid"
  "videochat-flash-2b|conditioned_keyframes/chunk|kf_chunk|videochat-flash"
  "videochat-flash-2b|conditioned_keyframes/referent|kf_referent|videochat-flash"
  "videochat-flash-2b|conditioned_keyframes/random|kf_random|videochat-flash"
  "videochat-flash-2b|conditioned_keyframes/uniform8|kf_uniform8|videochat-flash"
  "gemma3-12b|conditioned_keyframes/chunk|kf_chunk|reid"
  "gemma3-12b|conditioned_keyframes/referent|kf_referent|reid"
  "gemma3-12b|conditioned_keyframes/random|kf_random|reid"
  "gemma3-12b|conditioned_keyframes/uniform8|kf_uniform8|reid"
)

echo "[feeder] $(date '+%F %T') waiting for the QSVideo runs to clear"
while squeue -u "$USER" -h -o "%j" | grep -q '^qsv_'; do sleep 120; done
echo "[feeder] $(date '+%F %T') QSVideo done; $((${#JOBS[@]})) jobs to place"

for spec in "${JOBS[@]}"; do
  IFS='|' read -r M C P E <<< "$spec"
  # skip anything already scored
  if [ -s "results_baseline/$P/$M/predictions.jsonl" ]; then
    n=$(wc -l < "results_baseline/$P/$M/predictions.jsonl")
    if [ "$n" -ge 3233 ]; then echo "[feeder] skip $M/$P (already $n rows)"; continue; fi
  fi
  while [ "$(mine)" -ge 3 ]; do sleep 120; done
  echo "[feeder] $(date '+%F %T') submit $M / $P (env=$E)"
  MODEL="$M" CLIPS="$C" PIPE="$P" ENVN="$E" \
    sbatch -x c2-2,c4-4,c4-5 -J "kfq_${M:0:8}_${P#kf_}" run_kf_eval.slurm
  sleep 20
done
echo "[feeder] $(date '+%F %T') all jobs placed"
