#!/bin/bash
# ---------------------------------------------------------------------------
# common.sh — shared config, colors and logging helpers for the re-ID baselines.
# Sourced by run_all_baselines.sh, run_combo.sh and the per-combo scripts.
# ---------------------------------------------------------------------------

# --- repo + data root (all pipeline scripts + benchmark json live here) -----
export GEN_REID_ROOT="${GEN_REID_ROOT:-/home/ab260989/gen-reid}"
REPO="$GEN_REID_ROOT"

# --- conda env pythons ------------------------------------------------------
REID_PY="${REID_PY:-/home/ab260989/.conda/envs/reid/bin/python}"
VCF_PY="${VCF_PY:-/home/ab260989/.conda/envs/videochat-flash/bin/python}"
LONGVU_PY="${LONGVU_PY:-/home/ab260989/.conda/envs/longvu/bin/python}"
MALMM_PY="${MALMM_PY:-/home/ab260989/.conda/envs/ma-lmm/bin/python}"
TRACK_PY="${TRACK_PY:-/home/ab260989/.conda/envs/track/bin/python}"
LONGVU_REPO="${LONGVU_REPO:-/home/ab260989/third_party/LongVU}"
MALMM_REPO="${MALMM_REPO:-/home/ab260989/third_party/MA-LMM}"

# --- GPU / runtime ----------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# --- the matrix -------------------------------------------------------------
TRACKERS=(botsort bytetrack deepocsort strongsort)
# methods: clip-link = tracker-only (CLIP feature linking); the rest = tracker + dedicated ReID
METHODS=(clip-link osnet clipreid transreid solider)
# 17 answering VLMs, tagged by the env that runs them
MODELS_ENV=(
  internvl3-2b:reid  internvl3-8b:reid  internvl3-14b:reid
  qwen2.5-vl-3b:reid qwen2.5-vl-7b:reid
  qwen3-vl-real-2b:reid qwen3-vl-real-4b:reid qwen3-vl-real-8b:reid
  ovis2.5-2b:reid ovis2.5-9b:reid
  gemma3-4b:reid gemma3-12b:reid
  video-llava:reid
  videochat-flash-2b:vcf videochat-flash-7b:vcf
  longvu-qwen2-7b:longvu
  ma-lmm-vicuna7b:malmm
)
N_MODELS=${#MODELS_ENV[@]}

# per-ReID cluster threshold (SOLIDER features are anisotropic -> smaller threshold)
reid_thresh() { case "$1" in solider) echo 0.04;; *) echo 0.30;; esac; }
# Stage-A env (OSNet lives in the boxmot `track` env; the rest in `reid`)
reid_assign_py() { case "$1" in osnet) echo "$TRACK_PY";; *) echo "$REID_PY";; esac; }
# pipeline / output name for a (method, tracker)
pipe_name() { local m="$1" t="$2"; if [ "$m" = clip-link ]; then echo "$t"; else echo "${m}__${t}"; fi; }

# --- colors -----------------------------------------------------------------
if [ -t 1 ] || [ "${FORCE_COLOR:-0}" = 1 ]; then
  C_RESET=$'\e[0m'; C_B=$'\e[1m'; C_DIM=$'\e[2m'
  C_RED=$'\e[31m'; C_GRN=$'\e[32m'; C_YEL=$'\e[33m'; C_BLU=$'\e[34m'; C_MAG=$'\e[35m'; C_CYN=$'\e[36m'
  C_BGB=$'\e[44m'; C_BGG=$'\e[42m'
else
  C_RESET=; C_B=; C_DIM=; C_RED=; C_GRN=; C_YEL=; C_BLU=; C_MAG=; C_CYN=; C_BGB=; C_BGG=
fi

_ts() { date +"%H:%M:%S"; }
banner()  { echo; echo "${C_B}${C_BGB}                                                                            ${C_RESET}";
            printf "${C_B}${C_BGB}  %-72s${C_RESET}\n" "$1";
            echo "${C_B}${C_BGB}                                                                            ${C_RESET}"; }
section() { echo; echo "${C_B}${C_CYN}━━━ $1 ${C_RESET}${C_DIM}($(_ts))${C_RESET}"; }
step()    { echo "${C_B}${C_BLU}▶ $1${C_RESET}"; }
info()    { echo "  ${C_DIM}· $1${C_RESET}"; }
ok()      { echo "  ${C_GRN}✓ $1${C_RESET}"; }
skip()    { echo "  ${C_YEL}↷ skip: $1${C_RESET}"; }
warn()    { echo "  ${C_YEL}⚠ $1${C_RESET}"; }
err()     { echo "  ${C_RED}✗ $1${C_RESET}" >&2; }

# run a command, tee to a log, time it, report status
run_logged() {
  local label="$1" log="$2"; shift 2
  mkdir -p "$(dirname "$log")"
  local t0=$SECONDS
  info "log -> ${log/#$REPO\//}"
  if ( cd "$REPO" && "$@" ) >"$log" 2>&1; then
    ok "$label  ${C_DIM}[$((SECONDS-t0))s]${C_RESET}"
    return 0
  else
    err "$label FAILED  (see $log)"; tail -n 8 "$log" | sed 's/^/      /'
    return 1
  fi
}
