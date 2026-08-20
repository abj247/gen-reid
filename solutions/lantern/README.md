# Lantern

Frame-level query-conditioned selection. Training-free, adds no parameters, and works with any
backbone that accepts a video file, because its output is a rendered clip rather than a change to
the model.

## What it does

For each question:

1. Sample a dense pool of candidate frames uniformly across the full video. Default 64.
2. Embed every candidate with a frozen CLIP image encoder.
3. Embed the query text with the same encoder and score every candidate by cosine similarity.
4. Keep the top K candidates, restored to temporal order. Default 8.
5. Render those frames to a short clip and record the mapping in a manifest.

The candidate pool is encoded once per video and reused across that video's questions, so the cost
is one CLIP pass per video plus one text embedding per question.

Default settings, all in `select_frames.py`:

```
N_CANDIDATES = 64                     candidate pool per video
TOPK         = 8                      frames kept, matched to the uniform-8 budget
CLIP_MODEL   = ViT-B-32-quickgelu, openai weights
CLIP_FPS     = 4                      playback rate of the rendered clip
```

## Why the budget matches by construction

With 64 candidates and 8 kept, the pool is `linspace(0, N-1, 64)` and 63 divided by 7 is 9, so
`linspace(0, N-1, 8)` is an exact subset of the pool at positions 0, 9, 18 and so on. The uniform
control therefore renders precisely the frames the uniform-8 baseline sees, not an approximation of
them, and the method and its control differ only in which 8 of the same 64 are chosen.

## Query modes

The mode determines what text the candidates are scored against.

- `question` the question text alone.
- `question_options` the question text and all eight options.
- `referent` the question text and the referring phrase from the question metadata. This is the
  identity-aware variant and the one reported as the method.
- `chunk` scores contiguous segments instead of individual frames. This is the model-agnostic
  variant of Cairn and is documented in `../cairn/README.md`. It lives in this file because it
  reuses the same scoring pass; running it here keeps the two methods on one identical pool.

Two further modes are controls and consult neither CLIP nor the query. The encoder is not even
loaded for them, which makes it impossible for a control to use query information by accident.

- `random` K frames drawn from the same pool with a per-question seed derived from the question
  key. Reproducible across reruns, independent of iteration order, and different for two questions
  on the same video.
- `uniform` the plain uniform-K frames, rendered identically.

## Running it

Selection is a CPU-friendly job. It uses a GPU when one is visible and falls back to CPU
otherwise, which takes roughly an hour for the corpus.

Render the method arm and both controls:

```bash
python -m solutions.lantern.select_frames --query_mode referent --out conditioned_keyframes/referent
python -m solutions.lantern.select_frames --query_mode random   --out conditioned_keyframes/random
python -m solutions.lantern.select_frames --query_mode uniform  --out conditioned_keyframes/uniform8
```

Each writes `<out>/<video_id>__<question_id>.mp4` and `<out>/manifest.json`.

Then evaluate each arm with the same backbone:

```bash
for arm in referent random uniform8; do
  python -m benchmark.evaluation.run_manifest --model internvl3-14b \
      --clips conditioned_keyframes/$arm --pipeline lantern_$arm
done
```

On a SLURM cluster, `slurm/evaluate.slurm` runs one arm and `slurm/queue_feeder.sh` submits a
backbone and arm matrix while respecting a concurrency cap. Both resolve the repository root from
their own location, so neither contains a machine-specific path.

## Reading the result

The headline is the method arm against the `random` control, on the paired intersection, by exact
McNemar. The gap between the `uniform` arm and the published uniform row from
`benchmark.evaluation.run_uniform` is the rendering offset and belongs to the harness rather than
to the method. `persistqa.stats.compare` computes both and phrases a non-significant result as a
bound.

## Known limits

The gain shrinks as the frame budget grows. The claim this method supports is better frame choice
under a tight budget, not better performance at any budget, so state the budget with every number.

Backbones that perform their own internal frame handling are sensitive to the rendering path far
more than others. That is a property of clip delivery, not of selection; see
`benchmark/evaluation/README.md` before concluding the method failed on such a backbone.
