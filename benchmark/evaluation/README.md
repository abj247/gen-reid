# Evaluation harness

Two entry points. Both write the same prediction schema, so anything downstream reads them
identically and a raw-versus-method comparison is a join on the question key.

## run_uniform.py

Samples frames uniformly across the full video and answers. This produces the baseline row for
every model and the frame-budget ladders.

```bash
python -m benchmark.evaluation.run_uniform --model internvl3-14b --num_frames 8
python -m benchmark.evaluation.run_uniform --model internvl3-14b --num_frames 32
```

Arguments that matter:

- `--model` a key in `benchmark/models_registry.py`.
- `--num_frames` the frame budget. State it whenever a number from this run is quoted.
- `--limit` stop after N questions. Use this to smoke test a new backbone.

## run_manifest.py

Answers on a set of frames chosen by a method. The method writes a manifest mapping each question
key to a rendered clip, and this runner feeds those clips through the model's standard video path,
so no model-specific caching code is involved and every backbone is driven identically.

```bash
python -m benchmark.evaluation.run_manifest --model internvl3-14b \
    --clips conditioned_keyframes/referent --pipeline lantern
```

The `--clips` directory must contain `manifest.json` of the form
`{"<video_id>|<question_id>": "<path to clip>"}`.

`--pipeline` names the output directory and should identify the arm, including whether it is a
method or a control, since the two are compared against each other later.

## The rendering path is not free

`run_manifest.py` evaluates rendered clips, while `run_uniform.py` evaluates the source video.
These are different decoders, containers and resolution histories, and the difference is not
constant across models. Measured cost of the rendering path, holding the frames identical:

- InternVL3-14B: about 0.8 accuracy points.
- Qwen2.5-VL-7B: about 0.2 accuracy points.
- VideoChat-Flash-7B: about 5.8 accuracy points.

A method evaluated through `run_manifest.py` must therefore be compared against uniform frames
that were also rendered and evaluated through `run_manifest.py`, never against the published
uniform row from `run_uniform.py`. Both solutions ship exactly that control. Ignoring this turns a
pipeline penalty into an apparent method failure, which is what the large VideoChat-Flash figure
above would otherwise look like.

## Output schema

One JSON object per line:

```
{"key": "vid_0001|q14", "model": "internvl3-14b", "pipeline": "lantern",
 "video_id": "vid_0001", "question_id": "q14",
 "capability": "location", "reid": "cross_scene_reid",
 "predicted": "B", "correct": "A", "is_correct": false}
```

Runs are resumable. On restart the runner reads the existing file, skips completed keys and
appends. Deleting the file forces a full rerun.

## Environments

Most backbones run in the environment defined by the top-level `requirements.txt`. VideoChat-Flash
and several long-video models need their own. See `docs/ENVIRONMENTS.md`.
