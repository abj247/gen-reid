#!/bin/bash
# ---------------------------------------------------------------------------
# run_all_baselines.sh — run the FULL re-ID baseline matrix in one go:
#     5 ID-methods  x  4 trackers  x  17 VLMs   (= 20 combinations)
#
#   ID-methods:  clip-link  (tracker-only, CLIP feature linking)
#                osnet | clipreid | transreid | solider  (tracker + dedicated ReID)
#
# Runs combination-by-combination, sequentially, on the current GPU. Every
# stage is resumable, so re-running skips finished work. For parallel cluster
# execution use the SLURM schedulers instead (see README "Running on SLURM").
#
#   usage:   ./run_all_baselines.sh                  # everything
#            METHODS="osnet solider" ./run_all_baselines.sh   # subset of methods
#            TRACKERS="botsort" ./run_all_baselines.sh        # subset of trackers
# ---------------------------------------------------------------------------
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/common.sh"

# allow METHODS / TRACKERS overrides from the environment
read -r -a METHODS  <<< "${METHODS:-${METHODS[*]}}"
read -r -a TRACKERS <<< "${TRACKERS:-${TRACKERS[*]}}"
TOTAL=$(( ${#METHODS[@]} * ${#TRACKERS[@]} ))

RUN_T0=$SECONDS
banner "RE-ID BASELINE MATRIX  ·  ${#METHODS[@]} methods × ${#TRACKERS[@]} trackers = $TOTAL combinations × $N_MODELS VLMs"
info "methods : ${METHODS[*]}"
info "trackers: ${TRACKERS[*]}"
info "repo=$REPO   gpu=$CUDA_VISIBLE_DEVICES"
info "started $(date)"

n=0; bad=0
for M in "${METHODS[@]}"; do
  for T in "${TRACKERS[@]}"; do
    n=$((n+1))
    echo; echo "${C_B}${C_BGG}  ━━━━━━━━━━━━━━━━━━━━━━━━  COMBINATION $n / $TOTAL  ━━━━━━━━━━━━━━━━━━━━━━━━  ${C_RESET}"
    echo "${C_B}  method=${C_CYN}$M${C_RESET}${C_B}   tracker=${C_CYN}$T${C_RESET}${C_B}   pipeline=${C_CYN}$(pipe_name "$M" "$T")${C_RESET}"
    if "$HERE/run_combo.sh" "$M" "$T"; then :; else bad=$((bad+1)); warn "combo $n ($M/$T) returned non-zero"; fi
    elapsed=$((SECONDS-RUN_T0))
    info "progress: $n/$TOTAL combinations done  ·  elapsed ${elapsed}s (~$((elapsed/60))m)"
  done
done

echo
banner "ALL DONE  ·  $((n-bad))/$TOTAL combinations OK  ·  total $(( (SECONDS-RUN_T0)/60 )) min"
info "aggregate the table + figures with:"
echo "      ${C_DIM}$REID_PY $REPO/aggregate_baselines_final.py${C_RESET}"
echo "      ${C_DIM}$REID_PY $REPO/plot_tracker_breakdown.py${C_RESET}"
[ "$bad" -eq 0 ] || { warn "$bad combination(s) had failures — check baselines/logs/"; exit 1; }
