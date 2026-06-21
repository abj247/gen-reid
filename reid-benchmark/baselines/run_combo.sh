#!/bin/bash
# ---------------------------------------------------------------------------
# run_combo.sh — run ONE baseline combination end-to-end: a single
#   (ID-method x tracker) pair, evaluated by all 17 VLMs.
#
#   usage:  ./run_combo.sh <method> <tracker>
#     method  : clip-link | osnet | clipreid | transreid | solider
#     tracker : botsort | bytetrack | deepocsort | strongsort
#
# Pipeline:  track  ->  [assign_ids]  ->  gen_conditioned_clips  ->  eval x17
# Every stage is RESUMABLE (already-finished outputs are detected and skipped).
# Per-stage logs go to baselines/logs/<pipeline>/.
# ---------------------------------------------------------------------------
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/common.sh"

M="${1:-}"; T="${2:-}"
[ -z "$M" ] || [ -z "$T" ] && { err "usage: run_combo.sh <method> <tracker>"; exit 2; }
PIPE="$(pipe_name "$M" "$T")"
LOGD="$HERE/logs/$PIPE"
COMBO_T0=$SECONDS

banner "COMBINATION:  method=$M   tracker=$T   ->  pipeline '$PIPE'"
info "repo=$REPO   gpu=$CUDA_VISIBLE_DEVICES   models=$N_MODELS"

# ---------------------------------------------------------------- 1. TRACK ---
section "STAGE 1/4 · tracking (shared per tracker)"
TRK_DIR="$REPO/results/tracks_$T"
have=$(ls "$TRK_DIR"/*.json 2>/dev/null | wc -l)
if [ "$have" -ge 400 ]; then
  skip "tracking — $have tracked videos already present in results/tracks_$T"
else
  step "tracking all videos with $T (YOLOv8x + BoxMOT)"
  run_logged "track[$T]" "$LOGD/01_track_$T.log" \
    "$TRACK_PY" track_all.py --tracker "$T" --shard 0 --nshards 1 || exit 1
fi

# ------------------------------------------------------ 2. ASSIGN GLOBAL IDs --
section "STAGE 2/4 · global identity assignment"
IDS_ARG=()
if [ "$M" = clip-link ]; then
  skip "ID assignment — clip-link does CLIP-feature clustering inside gen_conditioned_clips"
else
  APY="$(reid_assign_py "$M")"; TH="$(reid_thresh "$M")"
  IDS_DIR="results/ids_${M}__${T}"
  ndone=$(ls "$REPO/$IDS_DIR"/*.json 2>/dev/null | wc -l)
  if [ "$ndone" -ge 400 ]; then
    skip "ID assignment — $ndone videos already in $IDS_DIR"
  else
    step "extracting $M ReID features -> clustering (thresh=$TH, py=$(basename "$(dirname "$(dirname "$APY")")"))"
    run_logged "assign_ids[$M/$T]" "$LOGD/02_assign_${M}.log" \
      "$APY" assign_ids.py --reid "$M" --tracks_dir "results/tracks_$T" \
        --out "$IDS_DIR" --shard 0 --nshards 1 --thresh "$TH" || exit 1
  fi
  IDS_ARG=(--ids_dir "$IDS_DIR")
fi

# ---------------------------------------------------- 3. CONDITIONED CLIPS ----
section "STAGE 3/4 · rendering identity-conditioned clips"
CLIPDIR="conditioned/$PIPE"
if [ -f "$REPO/$CLIPDIR/manifest.json" ]; then
  nclip=$($REID_PY -c "import json;print(len(json.load(open('$REPO/$CLIPDIR/manifest.json'))))" 2>/dev/null || echo 0)
  skip "genclips — manifest present ($nclip clips) in $CLIPDIR"
else
  step "grounding referent + rendering marked clips -> $CLIPDIR"
  run_logged "genclips[$PIPE]" "$LOGD/03_genclips.log" \
    "$REID_PY" gen_conditioned_clips.py --tracks_dir "results/tracks_$T" "${IDS_ARG[@]}" --out "$CLIPDIR" || exit 1
fi

# ----------------------------------------------------------- 4. EVAL x17 ----
section "STAGE 4/4 · evaluating $N_MODELS VLMs on '$PIPE'"
i=0; failed=0
for spec in "${MODELS_ENV[@]}"; do
  model="${spec%%:*}"; envtag="${spec##*:}"; i=$((i+1))
  pred="$REPO/results_baseline/$PIPE/$model/predictions.jsonl"
  nlines=$( [ -f "$pred" ] && wc -l <"$pred" || echo 0 )
  printf "${C_B}${C_MAG}[model %2d/%d]${C_RESET} %-20s ${C_DIM}(%s env)${C_RESET}\n" "$i" "$N_MODELS" "$model" "$envtag"
  if [ "$nlines" -ge 3233 ]; then skip "$model already complete ($nlines preds)"; continue; fi
  log="$LOGD/04_eval_${model}.log"
  case "$envtag" in
    reid)  run_logged "$model" "$log" "$REID_PY" eval_conditioned.py --model "$model" --clips "$CLIPDIR" --pipeline "$PIPE" || failed=$((failed+1));;
    vcf)   run_logged "$model" "$log" "$VCF_PY"  eval_conditioned.py --model "$model" --clips "$CLIPDIR" --pipeline "$PIPE" || failed=$((failed+1));;
    longvu) run_logged "$model" "$log" env PYTHONPATH="$LONGVU_REPO" HF_HUB_OFFLINE=1 \
              "$LONGVU_PY" eval_longvu_conditioned.py --manifest "$CLIPDIR/manifest.json" --output_dir "results_baseline/$PIPE" || failed=$((failed+1));;
    malmm) run_logged "$model" "$log" env PYTHONPATH="$MALMM_REPO" HF_HUB_OFFLINE=1 \
              "$MALMM_PY" eval_malmm_conditioned.py --manifest "$CLIPDIR/manifest.json" --output_dir "results_baseline/$PIPE" || failed=$((failed+1));;
  esac
done

echo
if [ "$failed" -eq 0 ]; then
  ok "COMBINATION '$PIPE' COMPLETE — $N_MODELS/$N_MODELS models  ${C_DIM}[total $((SECONDS-COMBO_T0))s]${C_RESET}"
else
  warn "COMBINATION '$PIPE' finished with $failed model failure(s) — inspect $LOGD"
fi
exit 0
