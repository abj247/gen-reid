#!/bin/bash
# Schedule the ReID-MODEL identity-assignment baselines: for each (reid model x tracker):
#   Stage-A assign_ids (array) -> Stage-B genclips(--ids_dir) -> 17-model eval.
# Pipeline name = <reid>__<tracker>  (clips in conditioned/<pipe>, preds in results_baseline/<pipe>).
# Resumable throughout. Run AFTER the OSNet smoke passes.
set -u
cd /home/ab260989/gen-reid
REIDS="${REIDS:-osnet}"                 # osnet | clipreid | transreid | solider
TRACKERS="${TRACKERS:-botsort bytetrack deepocsort strongsort}"
NSHARDS=16
REID_PY=/home/ab260989/.conda/envs/reid/bin/python
VCF_PY=/home/ab260989/.conda/envs/videochat-flash/bin/python
TRACK_PY=/home/ab260989/.conda/envs/track/bin/python

# per-ReID Stage-A env + cluster threshold (SOLIDER features are anisotropic -> smaller threshold)
assign_py() { case "$1" in osnet) echo "$TRACK_PY";; *) echo "$REID_PY";; esac; }
assign_th() { case "$1" in solider) echo 0.04;; *) echo 0.30;; esac; }
REID_MODELS="internvl3-2b internvl3-8b internvl3-14b qwen2.5-vl-3b qwen2.5-vl-7b qwen3-vl-real-2b qwen3-vl-real-4b qwen3-vl-real-8b ovis2.5-2b ovis2.5-9b gemma3-4b gemma3-12b video-llava"
VCF_MODELS="videochat-flash-2b videochat-flash-7b"
T_DIR=/home/ab260989/gen-reid/slurm_tpl
LOG=/home/ab260989/gen-reid/schedule_reid_log.txt; : > "$LOG"

for R in $REIDS; do
  for T in $TRACKERS; do
    PIPE="${R}__${T}"
    AID=$(sbatch --parsable --array=0-$((NSHARDS-1)) -J as_${PIPE} "$T_DIR/assign_ids.slurm" "$R" "$T" "$NSHARDS" "$(assign_py $R)" "$(assign_th $R)")
    echo "[assign] $PIPE -> array $AID (py=$(assign_py $R) th=$(assign_th $R))" | tee -a "$LOG"
    CID=$(sbatch --parsable --dependency=afterany:$AID -J ci_${PIPE} "$T_DIR/genclips_ids.slurm" "$R" "$T")
    echo "[clips]  $PIPE -> $CID (after $AID)" | tee -a "$LOG"
    for M in $REID_MODELS; do
      J=$(sbatch --parsable --dependency=afterok:$CID -J ev_${PIPE}_${M} "$T_DIR/eval_cond.slurm" "$M" "$PIPE" "$REID_PY")
      echo "[eval]   $PIPE / $M -> $J" | tee -a "$LOG"
    done
    for M in $VCF_MODELS; do
      J=$(sbatch --parsable --dependency=afterok:$CID -J ev_${PIPE}_${M} "$T_DIR/eval_cond.slurm" "$M" "$PIPE" "$VCF_PY")
      echo "[eval]   $PIPE / $M -> $J" | tee -a "$LOG"
    done
    J=$(sbatch --parsable --dependency=afterok:$CID -J ev_${PIPE}_longvu "$T_DIR/eval_longvu.slurm" "$PIPE"); echo "[eval]   $PIPE / longvu -> $J" | tee -a "$LOG"
    J=$(sbatch --parsable --dependency=afterok:$CID -J ev_${PIPE}_malmm  "$T_DIR/eval_malmm.slurm"  "$PIPE"); echo "[eval]   $PIPE / malmm  -> $J" | tee -a "$LOG"
  done
done
NP=$(echo $REIDS|wc -w); NT=$(echo $TRACKERS|wc -w); NM=$(($(echo $REID_MODELS $VCF_MODELS|wc -w)+2))
echo "=== scheduled: $((NP*NT)) assign arrays + $((NP*NT)) genclips + $((NP*NT*NM)) eval jobs ===" | tee -a "$LOG"
echo "watch: squeue -u ab260989 | head" | tee -a "$LOG"
