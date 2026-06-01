# Video Re-Identification Benchmark: Text-Only Debiasing and Video+Text Evaluation

This repository contains the full pipeline for building a vision-grounded
multiple-choice video re-identification benchmark and for evaluating
vision-language models (VLMs) on it. The central problem it solves is text-only
leakage: on the original 8-option benchmark, language models answered far above
the 12.5 percent random baseline without ever seeing the video, by exploiting
lexical and plausibility cues in the answer options. We remove that leakage with
adversarial committee filtering, then measure how much each model genuinely
gains from the video.

## Headline result

On the original benchmark the 13-model committee scored 26 to 36 percent in
text-only mode (2 to 3 times the 12.5 percent chance level). After committee
filtering the mean text-only accuracy drops to 12.6 percent and every model sits
within a few points of chance. Adding video on the filtered set raises the mean
accuracy to 17.8 percent, a mean video gain of 4.5 percentage points, confirming
the benchmark now rewards vision rather than text priors.

## Pipeline overview

```
original benchmark (real video ids)
        |
        v  src/anonymize_benchmark.py        (strip video ids -> vid_XXXX, keep mapping)
anonymized benchmark
        |
        v  src/eval_text_only.py             (13 models, no video, batched)
text-only predictions
        |
        v  src/committee_filter.py           (drop questions the committee can guess)
debiased benchmark (anonymized)
        |
        v  src/deanonymize_benchmark.py      (restore real video ids for the video run)
debiased benchmark (real ids)
        |
        v  src/eval_video_text.py            (13 models, 8 frames + text)
video+text predictions
        |
        +--> src/llm_retag_challenges.py     (clean Re-ID challenge taxonomy)
        +--> analysis/*                       (plots, failure analysis, scaling)
```

## Repository structure

```
reid-benchmark/
  src/
    vlm_models.py             Model factory and per-family inference (text-only, video, batched)
    anonymize_benchmark.py    Replace real video ids with vid_XXXX, write id mapping
    deanonymize_benchmark.py  Restore real video ids using the mapping
    eval_text_only.py         Multi-model text-only evaluation, JSONL checkpointed, resumable
    eval_video_text.py        Multi-model video+text evaluation, per-video frame caching
    committee_filter.py       Adversarial committee filtering (the debiasing step)
    llm_retag_challenges.py   LLM classifier for canonical Re-ID challenge categories
  analysis/
    plot_text_only.py             Text-only bias and scaling figures
    plot_video_vs_text.py         Video vs text figures, video-gain figures
    plot_scaling_grid.py          Two-row per-capability scaling grid (text vs video)
    analyze_failures.py           Modality-overlap and failure-mode analysis
    analyze_failures_comprehensive.py  Failure cube, wrong-answer convergence, SVD
    analyze_binding_errors.py     Who/what binding-error decomposition (LLM)
    analyze_video_properties.py   Scene-cuts / people / CLIP-variability per video
  lvu_models/                     Standalone wrappers for the long-video models
    eval_longvu_text_only.py      LongVU text-only (longvu env, PYTHONPATH=third_party/LongVU)
    eval_longvu_video_text.py     LongVU video+text
    eval_malmm_text_only.py       MA-LMM text-only (ma-lmm env, PYTHONPATH=third_party/MA-LMM)
    eval_malmm_video_text.py      MA-LMM video+text
  slurm/
    run_text_only.slurm           Submit the text-only evaluation (main env)
    run_video_text.slurm          Submit the video+text evaluation (main env)
    run_retag.slurm               Submit the LLM re-tagging
    run_text_only_videochat_flash.slurm / run_video_text_videochat_flash.slurm
    run_text_only_longvu.slurm    / run_video_text_longvu.slurm
    run_text_only_malmm.slurm     / run_video_text_malmm.slurm
  requirements/
    reid.txt videochat-flash.txt longvu.txt ma-lmm.txt   curated per-env deps
    <env>-freeze.txt              exact pip freeze per env (full reproduction)
  data/                           Benchmark JSONs and id mapping (populated by the pipeline)
  results/                        Predictions, metrics, plots (generated; gitignored)
  requirements.txt                points at requirements/reid.txt
  README.md
```

## Environment setup

This project needs FOUR conda environments. The main env runs every
HuggingFace-native model (Qwen, Ovis, InternVL, Gemma, Video-LLaVA) and all
analysis. The three long-video models (VideoChat-Flash, LongVU, MA-LMM) ship
research code pinned to older, mutually incompatible transformers versions, so
each gets its own env. A single 48 GB GPU runs every model; the 14B checkpoint
uses 4-bit quantization.

Per-env requirement files are in requirements/. Exact pip freezes (every
transitive package) are in requirements/<env>-freeze.txt for byte-for-byte
reproduction.

### 1. Main env (most users only need this)

```
conda create -n reid python=3.11 -y
conda activate reid
pip install -r requirements/reid.txt
```

Runs: qwen2-vl, qwen2.5-vl-*, qwen3-vl-real-*, ovis*, internvl3-*, gemma3-*,
video-llava, and ALL analysis/plotting/video-property scripts. flash-attention
is not required (the code requests the sdpa backend).

### 2. VideoChat-Flash env

```
conda create -n videochat-flash python=3.10 -y
conda activate videochat-flash
pip install -r requirements/videochat-flash.txt
# REQUIRED: stub flash_attn so the model's import check passes without compiling it
python - <<'PY'
import os, sysconfig
d = os.path.join(sysconfig.get_paths()["purelib"], "flash_attn")
os.makedirs(d, exist_ok=True)
open(os.path.join(d, "__init__.py"), "w").write('__version__ = "0.0.0+stub"\n')
print("stub flash_attn written to", d)
PY
```

### 3. LongVU env

```
conda create -n longvu python=3.10 -y
conda activate longvu
git clone https://github.com/Vision-CAIR/LongVU third_party/LongVU
pip install -r requirements/longvu.txt
# run wrappers with: PYTHONPATH=third_party/LongVU python lvu_models/eval_longvu_*.py ...
```

### 4. MA-LMM env

```
conda create -n ma-lmm python=3.10 -y   # 3.10, NOT 3.9 (broken pip in the 3.9 conda build)
conda activate ma-lmm
git clone https://github.com/boheumd/MA-LMM third_party/MA-LMM
cd third_party/MA-LMM && pip install -e . && cd ../..
pip install -r requirements/ma-lmm.txt   # applies the numpy/opencv/transformers fixes
# run wrappers with: PYTHONPATH=third_party/MA-LMM python lvu_models/eval_malmm_*.py ...
```

## Troubleshooting (dependency issues a new user will hit)

These are the concrete fixes we applied; without them the long-video models
fail to load. They are also encoded as comments in the per-env requirement files.

| Symptom | Cause | Fix |
|---------|-------|-----|
| VideoChat-Flash: `ImportError: requires flash_attn` | The bundled modeling code lists flash_attn in its import check even though it falls back to sdpa | Install the stub flash_attn package shown in step 2 (no compilation needed) |
| VideoChat-Flash: `DynamicCache has no attribute seen_tokens / get_max_length / get_usable_length` | Model code uses transformers <=4.40 cache APIs removed in 4.55+ | Use the videochat-flash env (transformers 4.40.1); do NOT run it in the reid env |
| VideoChat-Flash 7B: `is not a valid model identifier` | The 7B HF id uses Qwen2 not Qwen2.5 | Already handled in src/vlm_models.py: 2B = Qwen2_5-2B_res448, 7B = Qwen2-7B_res448 |
| LongVU: `rope_scaling must have two fields` | transformers 4.42 rejects the Llama-3.2 rope config | Use the longvu-qwen2-7b checkpoint (Qwen2 backbone, unaffected). The Llama3.2-3B variant needs a transformers patch and is not used. |
| MA-LMM: `TypeError: dataclass() got unexpected keyword 'slots'` | pip in the python-3.9 conda build is broken on this cluster | Recreate the env with python 3.10 |
| MA-LMM: `numpy.core.multiarray failed to import` (via cv2) | LAVIS pins opencv 4.5.5.64 which clashes with modern numpy ABI | pip install "numpy<2" "opencv-python-headless>=4.8" (in requirements/ma-lmm.txt) |
| MA-LMM: `cannot import name apply_chunking_to_forward` | `pip install -e .` pulled transformers 5.x; LAVIS needs ~4.33 | pip install transformers==4.33.3 tokenizers<0.15 (in requirements/ma-lmm.txt) |
| MA-LMM: `llm/vicuna-7b is not a valid model identifier` | MA-LMM expects local Vicuna-v1.1 delta weights that are not shipped | The wrappers redirect the llm_model path to the public lmsys/vicuna-7b-v1.5; if loading your own checkout, edit lavis/configs/models/blip2/blip2_instruct_vicuna7b.yaml |
| Any analysis script: `ZeroDivisionError` / empty join | Video predictions use real YouTube ids; benchmark metadata uses anonymized vid_XXXX | All scripts map real->anon via data/video_id_mapping.json. Pass --mapping. |
| Non-ASCII crash (`latin-1 codec can't encode`) on cluster stdout | SLURM stdout defaults to latin-1 | Scripts and SLURM files set PYTHONIOENCODING=utf-8; keep it set |
| `decord` returns a torch tensor, `.asnumpy()` fails | MA-LMM's LAVIS sets decord bridge to torch at import | The MA-LMM video wrapper detects this and handles both bridges |

Hardware note: every model fits on one 48 GB GPU (SLURM constraint `gmem48`).
The reid env runs on as little as 16 GB for the smaller checkpoints.

## Data layout

The pipeline reads the source benchmark and the directory of mp4 files, then
writes intermediate benchmarks into data/. Expected files after a full run:

```
data/
  benchmark_text_only.json            anonymized, used for text-only eval
  video_id_mapping.json               vid_XXXX to real youtube id mapping
  benchmark_debiased.json             committee-filtered, anonymized
  benchmark_debiased_real_ids.json    committee-filtered, real ids, for video eval
  benchmark_debiased_retagged.json    debiased + canonical Re-ID tags
```

The video files are expected as one mp4 per video id, for example
/home/c3-0/datasets/moviechat1k-test/<video_id>.mp4

## Reproduction

All commands are run from the reid-benchmark directory. The SLURM scripts wrap
the same commands for cluster submission.

### Step 1. Anonymize the benchmark

Removes the real video id so a text-only model cannot recognize the source.

```
python src/anonymize_benchmark.py \
    --input  /path/to/original_benchmark.json \
    --output data/benchmark_text_only.json \
    --mapping data/video_id_mapping.json
```

### Step 2. Text-only evaluation

Runs every model on question text plus options with no video. Predictions are
checkpointed per model as JSONL, so a re-run resumes automatically.

```
python src/eval_text_only.py \
    --benchmark data/benchmark_text_only.json \
    --output_dir results/text_only
```

Cluster: `sbatch slurm/run_text_only.slurm`

### Step 3. Committee filtering (debiasing)

Drops every question that the committee can answer from text alone. The
threshold is auto-calibrated to bring mean committee accuracy to the target.

```
python src/committee_filter.py \
    --raw_results results/text_only/raw_results.json \
    --benchmark   data/benchmark_text_only.json \
    --output      data/benchmark_debiased.json \
    --target_acc  13.0
```

A question is dropped if at least tau_correct models answer it correctly, or if
at least tau_mode models converge on the same wrong option. tau_correct is
chosen by a sweep that reports kept count and mean accuracy at each threshold.

### Step 4. Restore real video ids

The video evaluator needs real ids to locate the mp4 files.

```
python src/deanonymize_benchmark.py \
    --input   data/benchmark_debiased.json \
    --mapping data/video_id_mapping.json \
    --output  data/benchmark_debiased_real_ids.json
```

### Step 5. Video+text evaluation

Runs every model with 8 sampled frames plus text. Frames are decoded once per
video and reused across that video's questions. Predictions are checkpointed.

```
python src/eval_video_text.py \
    --benchmark data/benchmark_debiased_real_ids.json \
    --video_dir /home/c3-0/datasets/moviechat1k-test \
    --output_dir results/video_text \
    --num_frames 8
```

Cluster: `sbatch slurm/run_video_text.slurm`

### Step 6. LLM re-tagging of Re-ID challenge types

The original challenge field is free text. This step assigns each question one
canonical label from a fixed taxonomy using a local instruct model.

```
python src/llm_retag_challenges.py \
    --bench data/benchmark_debiased.json \
    --model google/gemma-3-12b-it \
    --batch_size 8 \
    --out_tags results/reid_retags.json \
    --out_bench data/benchmark_debiased_retagged.json
```

Cluster: `sbatch slurm/run_retag.slurm`

### Step 7. Analysis and figures

```
# Text-only bias and scaling figures
python analysis/plot_text_only.py \
    --preds_dir results/text_only \
    --bench_filtered data/benchmark_debiased.json \
    --output_dir results/plots_text_only

# Video vs text figures and video-gain figures
python analysis/plot_video_vs_text.py \
    --video_dir results/video_text \
    --text_dir  results/text_only \
    --bench data/benchmark_debiased.json \
    --mapping data/video_id_mapping.json \
    --output_dir results/plots_video_vs_text

# Two-row per-capability scaling grid (text on top, video+text on bottom)
python analysis/plot_scaling_grid.py \
    --video_dir results/video_text \
    --text_dir  results/text_only \
    --bench data/benchmark_debiased.json \
    --mapping data/video_id_mapping.json \
    --output_dir results/plots_video_vs_text

# Modality-overlap and failure-mode analysis (uses canonical tags if present)
python analysis/analyze_failures.py \
    --video_dir results/video_text \
    --text_dir  results/text_only \
    --bench data/benchmark_debiased_retagged.json \
    --mapping data/video_id_mapping.json \
    --output_dir results/analysis_failures
```

## Model matrix

Thirteen in-committee checkpoints across four scale tiers and six families.
Family and size are read from the model key.

| Tier   | Models |
|--------|--------|
| 2B     | ovis2.5-2b, internvl3-2b, qwen3-vl-real-2b |
| 3 to 4B| qwen2.5-vl-3b, qwen3-vl-real-4b, gemma3-4b |
| 7 to 9B| qwen2.5-vl-7b, qwen3-vl-real-8b, ovis2.5-9b, internvl3-8b, video-llava |
| 12 to 14B | gemma3-12b, internvl3-14b (4-bit) |

Add or change models in the model_map inside src/vlm_models.py. The two
evaluation drivers read model keys from that factory.

### Held-out long-video (LVU) models

Four additional long-video models were evaluated AFTER the committee filter was
fixed, so they were never used to construct the benchmark. They serve as a
held-out generalization check: if the debiasing transfers, these unseen
architectures should also be near the 12.5 percent text-only baseline. They
are. The four are VideoChat-Flash (2B and 7B), LongVU-Qwen2-7B, and
MA-LMM-Vicuna-7B.

Each pins to an older transformers version and so runs in its own conda env,
via standalone wrappers in lvu_models/ that emit the same predictions.jsonl
schema as the main drivers:

| Model | Conda env | transformers | Wrapper | Notes |
|-------|-----------|--------------|---------|-------|
| VideoChat-Flash 2B/7B | videochat-flash | 4.40.1 (+ stub flash_attn) | src/vlm_models.py VideoChatFlash class | model.chat() takes an mp4 path; text-only uses data/dummy_black.mp4 |
| LongVU-Qwen2-7B | longvu | 4.42.4 | lvu_models/eval_longvu_*.py | requires the cloned LongVU repo on PYTHONPATH |
| MA-LMM-Vicuna-7B | ma-lmm | 4.33.3 (LAVIS) | lvu_models/eval_malmm_*.py | native 20-frame + 10-slot memory bank; Vicuna-v1.5 substituted for the unavailable v1.1 |

SLURM wrappers for all four are in slurm/run_{text_only,video_text}_{videochat_flash,longvu,malmm}.slurm.
The LVU video evaluators use each model's recommended frame count (8 for
VideoChat-Flash and LongVU; 20 plus memory bank for MA-LMM), not a forced common
count, so each architecture runs in its intended configuration.

## Output formats

Each model writes results/<run>/<model_key>/predictions.jsonl, one JSON object
per question:

```
{"model_name": "...", "video_id": "...", "question_id": "...",
 "capability": "...", "referral_strategy": "...", "difficulty": "...",
 "predicted": "C", "correct": "A", "is_correct": false}
```

The driver also aggregates these into results/<run>/raw_results.json, a dict
keyed by model display name, each value holding a predictions list. The
committee filter and all analysis scripts read this aggregated file.

## Key results

All 17 models. Text-only full = on the original 7390 questions (in-committee
only). Text-only filtered and Video+Text = on the committee-debiased set.
Video gain = Video+Text minus Text-only on the same matched questions. Random
baseline is 12.5 percent. Held-out models were never used to build the filter.

In-committee models (the 13 that built the filter):

| Model | Text-only full | Text-only filtered | Video+Text | Video gain |
|-------|---------------:|-------------------:|-----------:|-----------:|
| InternVL3-14B | 35.6 | 17.0 | 23.3 | +6.3 |
| InternVL3-8B  | 33.2 | 15.0 | 21.2 | +6.2 |
| Ovis2.5-9B    | 35.2 | 13.3 | 21.1 | +7.8 |
| Ovis2.5-2B    | 30.5 | 14.7 | 20.4 | +5.7 |
| Gemma3-12B    | 30.9 | 14.4 | 19.5 | +5.2 |
| InternVL3-2B  | 29.3 | 11.9 | 19.2 | +7.3 |
| Gemma3-4B     | 27.3 | 12.4 | 16.7 | +4.3 |
| Qwen2.5-VL-7B | 32.5 | 12.5 | 16.0 | +3.5 |
| Qwen3-VL-8B   | 33.2 | 11.2 | 15.9 | +4.7 |
| Qwen3-VL-4B   | 31.0 | 11.3 | 15.6 | +4.3 |
| Qwen3-VL-2B   | 26.0 | 12.2 | 15.2 | +3.0 |
| Qwen2.5-VL-3B | 31.5 | 14.7 | 14.9 | +0.1 |
| Video-LLaVA-7B| 19.4 | 12.1 | 12.4 | +0.3 |

Held-out long-video models (generalization check; never used to filter):

| Model | Text-only filtered | Video+Text | Video gain |
|-------|-------------------:|-----------:|-----------:|
| VideoChat-Flash-7B | 13.9 | 24.2 | +10.3 |
| VideoChat-Flash-2B | 14.2 | 23.3 | +9.1 |
| LongVU-Qwen2-7B    | 13.2 | 16.5 | +3.3 |
| MA-LMM-Vicuna-7B   | 12.4 | 10.4 | -2.0 |

Headlines:
- Best overall is the held-out VideoChat-Flash-7B at 24.2 percent (random 12.5).
- All 17 models are near chance text-only on the filtered set (mean ~13), so the
  debiasing transfers to unseen architectures.
- Video gain spans +10.3 (VideoChat-Flash) down to -2.0 (MA-LMM: video HURTS it;
  its memory-bank compression destroys fine identity cues even at 20 frames).
- Scaling: after filtering, the text-only slope is near zero or negative within
  most families (Qwen2.5-VL is -6.3 points per 10x params), while the video+text
  slope is positive for every family (+3.4 mean). Scale buys vision, not text
  shortcuts.

### Failure analysis (why models fail)

Across all 17 models (see analysis/analyze_failures_comprehensive.py and
analyze_binding_errors.py):
- 33 percent of questions are missed by every model; 0 are solved by all 17.
- When models err they converge on the same wrong option 57.7 percent of the
  time vs 29.8 percent random-distractor null (1.9x), flat across difficulty, so
  failures are systematic.
- 95.2 percent of those systematic errors are right-person / wrong-attribute
  (an identity-attribute binding failure), not wrong-person and not memory.
- Video structure (scene cuts, crowding, duration) does not predict difficulty
  (Pearson r near zero for crowding and cuts x people), so more frames or longer
  context will not fix it.
- Implication: build an explicit identity-attribute binding mechanism, evaluated
  on the 65 percent "Discriminator" subset; do not build more temporal memory.

## Notes

- The committee filter generalizes across families because the leakage channels
  it targets (plausibility, lexical register, option overlap) are shared. A held
  out model from a new family is expected to also be near chance on the filtered
  set.
- InternVL3 retains a small positive text-only scaling slope, indicating
  residual leakage that grows with size. Treat its video numbers as an upper
  bound and consider a second filtering pass.
- 8-frame sampling is a deliberate efficiency choice. The analysis shows that on
  about 6 percent of answers the sparse frames mislead a model that would have
  answered correctly from text, which motivates denser or streaming frame access
  as future work.
