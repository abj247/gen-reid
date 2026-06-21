#!/bin/bash
# Schedule the FULL baseline matrix: 4 trackers x (Stage1 track -> Stage2a clips -> Stage2b 17-model eval)
# with SLURM dependencies. Resumable throughout. Run AFTER per-model smokes pass.
set -u
cd /home/ab260989/gen-reid
TRACKERS="botsort bytetrack deepocsort strongsort"
NSHARDS=16
REID_PY=/home/ab260989/.conda/envs/reid/bin/python
VCF_PY=/home/ab260989/.conda/envs/videochat-flash/bin/python
REID_MODELS="internvl3-2b internvl3-8b internvl3-14b qwen2.5-vl-3b qwen2.5-vl-7b qwen3-vl-real-2b qwen3-vl-real-4b qwen3-vl-real-8b ovis2.5-2b ovis2.5-9b gemma3-4b gemma3-12b video-llava"
VCF_MODELS="videochat-flash-2b videochat-flash-7b"
T_DIR=/home/ab260989/gen-reid/slurm_tpl
LOG=/home/ab260989/gen-reid/schedule_log.txt; : > "$LOG"

for T in $TRACKERS; do
  TID=$(sbatch --parsable --array=0-$((NSHARDS-1)) -J trk_$T "$T_DIR/track.slurm" "$T" "$NSHARDS")
  echo "[track]  $T -> array $TID ($NSHARDS shards)" | tee -a "$LOG"
  CID=$(sbatch --parsable --dependency=afterany:$TID -J clips_$T "$T_DIR/genclips.slurm" "$T")
  echo "[clips]  $T -> $CID (after $TID)" | tee -a "$LOG"
  for M in $REID_MODELS; do
    J=$(sbatch --parsable --dependency=afterok:$CID -J ev_${T}_${M} "$T_DIR/eval_cond.slurm" "$M" "$T" "$REID_PY")
    echo "[eval]   $T / $M -> $J" | tee -a "$LOG"
  done
  for M in $VCF_MODELS; do
    J=$(sbatch --parsable --dependency=afterok:$CID -J ev_${T}_${M} "$T_DIR/eval_cond.slurm" "$M" "$T" "$VCF_PY")
    echo "[eval]   $T / $M -> $J" | tee -a "$LOG"
  done
  J=$(sbatch --parsable --dependency=afterok:$CID -J ev_${T}_longvu "$T_DIR/eval_longvu.slurm" "$T"); echo "[eval]   $T / longvu-qwen2-7b -> $J" | tee -a "$LOG"
  J=$(sbatch --parsable --dependency=afterok:$CID -J ev_${T}_malmm  "$T_DIR/eval_malmm.slurm"  "$T"); echo "[eval]   $T / ma-lmm-vicuna7b -> $J" | tee -a "$LOG"
done
N=$(grep -c "^\[" "$LOG")
echo "=== scheduled: 4 tracking arrays + 4 clip-gen + $(($(echo $REID_MODELS $VCF_MODELS|wc -w)*4 + 8)) eval jobs (per-pipeline deps) ===" | tee -a "$LOG"
echo "watch with: squeue -u ab260989 | head" | tee -a "$LOG"
