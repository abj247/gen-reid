# Re-ID Baselines: Track → Re-Identify → Identity-Conditioned VLM QA

This folder contains the **tracking + re-identification baselines** for the video
re-ID benchmark. They test one question: *if we do the identity bookkeeping for the
VLM — find the question's referent, follow it across the whole video, and hand the
model a clean clip of just that identity — does video-QA accuracy improve?*

The answer, across **5 identity methods × 4 trackers × 17 VLMs**, is **no**: every
identity-conditioning pipeline scores **at or below** feeding the model the raw video.

---

## 1. Objective — why tracker, why ReID, why tracker + ReID

A VLM answering a long-video question only sees ~8–16 sampled frames of the whole
scene. For a question about one specific person ("what is the man in the red jacket
doing later?") it must (a) locate that person, (b) keep their identity consistent
across shots/time, and (c) read off the answer. These baselines try to remove
burdens (a) and (b) so the model can focus on (c).

| Stage | What it does | Why we try it |
|-------|--------------|---------------|
| **Tracker only** (`clip-link`) | A tracker (BoT-SORT / ByteTrack / DeepOCSORT / StrongSORT) finds people/animals and links them **within a shot** into tracklets. Tracklets are then stitched **across shots** into global identities using **CLIP** appearance features. | Removes "which person?" ambiguity using only a generic, off-the-shelf feature. Baseline lower bound for identity linking. |
| **Tracker + ReID** (`osnet`, `clipreid`, `transreid`, `solider`) | Replaces the generic CLIP linking with a **purpose-built person-ReID model** for the cross-shot global-ID assignment. | ReID models are trained specifically to say "same identity across different shots/poses/lighting", so they should produce cleaner identities → a cleaner single-identity clip → better answers. The benchmark is ~75 % person-referent, so person-ReID is well matched. |

For **every** method the downstream is identical: ground the question's referent to one
global identity, render a short MP4 showing **only that identity** (boxed/marked), and
let each VLM answer on that conditioned clip. The comparison point is **`raw`** — the
same VLM on the untouched full video.

**Pipeline (per method × tracker):**
```
video ──▶ track_all.py ──▶ [assign_ids.py] ──▶ gen_conditioned_clips.py ──▶ eval_*.py (×17 VLMs)
          (detect+track)    (ReID→global IDs)    (ground + render clip)      (answer the MCQ)
                            (skipped for clip-link; CLIP clustering done in genclips)
```

---

## 2. Folder layout

```
baselines/
├── run_all_baselines.sh      # MASTER: runs the whole matrix (5 methods × 4 trackers), sequential, resumable
├── run_combo.sh              # WORKER: one (method, tracker) end-to-end, all 17 VLMs
├── combos/                   # 20 individual one-combination scripts (run_<method>_<tracker>.sh)
├── lib/common.sh             # shared config: env pythons, model list, colors, logging
├── src/                      # reference snapshot of the pipeline scripts (live copies at $GEN_REID_ROOT)
├── slurm/                    # SLURM templates + schedulers for the parallel cluster path
├── figures/                  # result figures (PNG+PDF) + baseline_matrix_final.csv
├── logs/                     # per-stage run logs (gitignored), written as <pipeline>/<stage>.log
└── README.md
```

---

## 3. How to run

All scripts read/write data under `$GEN_REID_ROOT` (default `/home/ab260989/gen-reid`),
where the benchmark JSON, videos mapping, and the shared model factory
(`evaluate_vlm_bm.py`, `probe_video_judge_v2.py`) live. Override with
`export GEN_REID_ROOT=/path/to/gen-reid`. Every stage is **resumable** — finished
outputs are detected and skipped, so re-running is cheap.

### a) Everything, one command
```bash
cd reid-benchmark/baselines
./run_all_baselines.sh
```
Colorful progress prints which combination (`COMBINATION 7 / 20`) and which model
(`[model 14/17] videochat-flash-2b (vcf env)`) is running, with per-stage timings.
Subsets:
```bash
METHODS="osnet solider" ./run_all_baselines.sh        # only some methods
TRACKERS="botsort"      ./run_all_baselines.sh        # only one tracker
```

### b) One combination at a time
```bash
./run_combo.sh transreid bytetrack          # parametric
./combos/run_transreid_bytetrack.sh         # equivalent individual script
```

### c) Aggregate the table + figures
```bash
"$REID_PY" "$GEN_REID_ROOT/aggregate_baselines_final.py"   # master table + 4 figures
"$REID_PY" "$GEN_REID_ROOT/plot_tracker_breakdown.py"      # method×tracker table fig + full per-model heatmap
```

### d) Running on SLURM (recommended for a full from-scratch run)
The sequential master runs one combination at a time on one GPU (a full cold run is
GPU-days). To parallelise, use the schedulers in `slurm/` which submit a dependency
DAG (track → assign → genclips → 17 evals) per pipeline:
```bash
bash slurm/schedule_baselines.sh                       # tracker-only (clip-link) × 4 trackers
REIDS="osnet clipreid transreid solider" bash slurm/schedule_reid_baselines.sh   # tracker + ReID grid
```

### Environments (conda)
| env | used for |
|-----|----------|
| `track` | tracking (BoxMOT) + OSNet ReID feature extraction |
| `reid`  | CLIP-ReID/TransReID/SOLIDER ReID, clip generation, 13 VLMs, plotting |
| `videochat-flash` | videochat-flash-2b / 7b |
| `longvu` | longvu-qwen2-7b |
| `ma-lmm` | ma-lmm-vicuna7b |

ReID checkpoints (`weights/`) and per-model details: OSNet `osnet_x0_25_msmt17`
(BoxMOT), CLIP-ReID/TransReID = ViT-B (Market1501, 256×128), SOLIDER = Swin-Base
(Market1501, 384×128). **SOLIDER's features are anisotropic (cosine distances ~6×
compressed), so its clustering threshold is 0.04 vs 0.30 for the others** —
calibrated to give comparable clustering granularity.

---

## 4. Results

Common evaluation set: **3128 questions** (intersection across all pipelines), chance = **12.5 %**.

### Accuracy by ID method × tracker (mean over 17 VLMs)

| ID method  | BoT-SORT | ByteTrack | DeepOCSORT | StrongSORT | **avg** | spread |
|------------|----------|-----------|------------|------------|---------|--------|
| CLIP-link  | 19.00    | 18.94     | 18.70      | 18.90      | **18.89** | 0.31 |
| OSNet      | 18.60    | 18.43     | 18.70      | 18.56      | **18.57** | 0.27 |
| CLIP-ReID  | 18.82    | 18.47     | 18.23      | 18.65      | **18.54** | 0.59 |
| TransReID  | 18.61    | 18.28     | 18.53      | 18.46      | **18.47** | 0.33 |
| SOLIDER    | 18.73    | 18.45     | 18.28      | 18.51      | **18.49** | 0.45 |

`raw` (VLM on full video) = **19.67 %**. Max tracker spread within any method = **0.59 pts**.

### Mean Δ vs raw (over 17 VLMs)
| method | CLIP-link | OSNet | CLIP-ReID | TransReID | SOLIDER |
|--------|-----------|-------|-----------|-----------|---------|
| Δ vs raw | −0.78 | −1.09 | −1.13 | −1.20 | −1.17 |

### Per-model accuracy (mean over the 4 trackers)
| Model | raw | CLIP-link | OSNet | CLIP-ReID | TransReID | SOLIDER |
|---|---|---|---|---|---|---|
| videochat-flash-7b | **26.4** | 21.4 | 21.0 | 20.3 | 20.4 | 20.3 |
| internvl3-14b | **25.7** | 24.9 | 24.0 | 23.9 | 23.8 | 23.6 |
| videochat-flash-2b | 25.4 | 21.6 | 20.3 | 20.1 | 20.4 | 20.0 |
| internvl3-8b | 23.4 | 22.8 | 22.0 | 22.0 | 21.9 | 21.9 |
| ovis2.5-9b | 23.0 | 22.4 | 21.1 | 21.1 | 20.9 | 21.0 |
| ovis2.5-2b | 22.1 | 21.6 | 20.7 | 20.6 | 20.2 | 20.2 |
| gemma3-12b | 21.3 | 21.0 | 20.4 | 20.4 | 20.5 | 20.3 |
| internvl3-2b | 20.9 | 20.4 | 19.8 | 20.0 | 19.8 | 19.9 |
| gemma3-4b | 18.3 | 17.5 | 17.6 | 17.9 | 17.9 | 17.7 |
| qwen2.5-vl-7b | 17.7 | 18.3 | 17.4 | 17.6 | 17.5 | 17.4 |
| longvu-qwen2-7b | 17.6 | 17.9 | 17.8 | 17.5 | 17.3 | 17.3 |
| qwen3-vl-real-8b | 17.3 | 17.2 | 18.4 | 17.9 | 17.8 | 18.2 |
| qwen3-vl-real-4b | 17.2 | 16.7 | 17.2 | 17.3 | 17.0 | 17.9 |
| qwen3-vl-real-2b | 17.0 | 16.0 | 16.4 | 16.7 | 16.5 | 16.5 |
| qwen2.5-vl-3b | 16.3 | 16.4 | 16.4 | 16.5 | 16.5 | 16.6 |
| video-llava | 13.3 | 13.2 | 13.7 | 13.5 | 13.5 | 13.7 |
| ma-lmm-vicuna7b | 11.3 | 11.6 | 11.8 | 12.0 | 12.0 | 12.0 |
| **MEAN** | **19.7** | **18.9** | **18.6** | **18.5** | **18.5** | **18.5** |

### Figures (`figures/`)
- `figures_final_method_means` — mean accuracy per method vs raw/chance.
- `figures_final_method_by_tracker` — each method's bar + its 4 individual tracker markers.
- `figures_final_tableA_method_x_tracker` — the method×tracker table as a shaded figure.
- `figures_final_tableB_full_model_x_tracker` — full 17 VLMs × (raw + 5 methods × 4 trackers) heatmap.
- `figures_final_delta_heatmap` — per-model Δ-vs-raw heatmap.
- `figures_final_per_model` — grouped per-VLM bars.
- `figures_final_tracker_invariance` — trackers on x-axis, methods grouped.
- `baseline_matrix_final.csv` — all numbers.

---

## 5. Takeaways

1. **No identity-conditioning method beats raw video** (mean 19.7 % → 18.5–18.9 %).
2. **Better ReID ≠ better QA**: dedicated person-ReID (OSNet/CLIP-ReID/TransReID/SOLIDER,
   Δ ≈ −1.1 to −1.2) is *worse* than even generic CLIP linking (−0.78). Identity-assignment
   quality is **not** the bottleneck.
3. **The strongest VLMs lose the most** (videochat-flash −5 to −6 pts): cropping to one boxed
   identity discards scene context capable models otherwise exploit. The only small "gains"
   are on near-chance models — noise.
4. **Tracker choice is irrelevant** — all four trackers land within ≤0.6 pts of each other.

Net: explicit track-then-ReID identity conditioning does not rescue long-video re-ID QA. The
bottleneck is fine-grained answer read-out, not identity tracking — which motivates a method
that improves the read-out rather than the bookkeeping.
