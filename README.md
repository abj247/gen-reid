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
    plot_text_only.py         Text-only bias and scaling figures
    plot_video_vs_text.py     Video vs text figures, video-gain figures
    plot_scaling_grid.py      Two-row per-capability scaling grid (text vs video)
    analyze_failures.py       Modality-overlap and failure-mode analysis
  slurm/
    run_text_only.slurm       Submit the text-only evaluation
    run_video_text.slurm      Submit the video+text evaluation
    run_retag.slurm           Submit the LLM re-tagging
  data/                       Benchmark JSONs and id mapping (populated by the pipeline)
  results/                    Predictions, metrics, plots (generated; gitignored)
  requirements.txt
  README.md
```

## Environment

Tested with Python 3.10 or newer, PyTorch 2.9 with CUDA, and Transformers 4.57.
A single 48 GB GPU runs every model in the matrix; the 14B checkpoint uses 4-bit
quantization.

```
conda create -n reid python=3.11 -y
conda activate reid
pip install -r requirements.txt
```

flash-attention is not required and is not used (the code requests the sdpa
attention backend, which runs on all supported GPUs).

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

Thirteen checkpoints across four scale tiers and six families. Family and size
are read from the model key.

| Tier   | Models |
|--------|--------|
| 2B     | ovis2.5-2b, internvl3-2b, qwen3-vl-real-2b |
| 3 to 4B| qwen2.5-vl-3b, qwen3-vl-real-4b, gemma3-4b |
| 7 to 9B| qwen2.5-vl-7b, qwen3-vl-real-8b, ovis2.5-9b, internvl3-8b, video-llava |
| 12 to 14B | gemma3-12b, internvl3-14b (4-bit) |

Add or change models in the model_map inside src/vlm_models.py. The two
evaluation drivers read model keys from that factory.

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

Text-only accuracy before and after committee filtering, and the video+text
result on the filtered set. Random baseline is 12.5 percent.

| Model | Text-only full | Text-only filtered | Video+Text filtered | Video gain |
|-------|---------------:|-------------------:|--------------------:|-----------:|
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
| Mean          | 30.5 | 13.3 | 17.8 | +4.5 |

Scaling: after filtering, the text-only slope is near zero or negative within
most families (for example Qwen2.5-VL is minus 6.3 points per 10x parameters),
while the video+text slope is positive for every family (mean plus 3.4 points
per 10x parameters). Scale buys vision, not text shortcuts.

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
