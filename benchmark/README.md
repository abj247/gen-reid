# The PersistQA benchmark

This directory holds the benchmark itself: the question set, the evaluation harness that produces
a prediction file for any supported backbone, the construction pipeline that built and debiased
the questions, and the analysis that turns prediction files into the reported tables.

## Contents

```
data/               Question set, identifier mapping, evidence windows. See data/README.md.
models_registry.py  Model factory. One entry per supported backbone.
video_io.py         Video resolution, frame sampling and rendering helpers.
evaluation/         Entry points that produce prediction files. See evaluation/README.md.
construction/       Question generation, adversarial debiasing, metadata tagging.
analysis/           Leaderboard and per-slice accuracy from prediction files.
assets/             Small fixtures, including the warmup clip used at model load.
```

## The task

Every question carries eight options, so chance is 12.5 percent. A model receives the question,
the eight options and a set of frames, and must emit a single option letter. Output is parsed by
one shared routine and scored by exact match. A response that cannot be parsed counts as wrong,
which keeps a model from benefiting by refusing to answer. Every model is evaluated zero-shot and
no model is trained on benchmark videos or questions.

## Evaluation protocol

Three properties are enforced by the harness and must be preserved by any new method added here.

**Budget matching.** A comparison is only meaningful when both arms receive the same number of
frames or the same number of visual tokens. Uniform sampling alone buys several accuracy points
when the budget grows, which is larger than any method effect measured in this repository, so an
unmatched comparison will manufacture a positive result. Every prediction file records the budget
it was produced at.

**A common question set.** Thirty source videos cannot be decoded by the evaluation stack. They
and their questions are excluded for every model rather than scored as wrong for some and skipped
for others, because the latter understates locally run models relative to API models. The excluded
identifiers are released with the data.

**Paired testing.** Two arms are compared on the intersection of the questions both answered,
using an exact McNemar test on the discordant pairs. A non-significant result is reported as a
bound, not as an absence. `persistqa.stats` implements this and is the only place a comparison
should be computed.

## Producing a prediction file

```bash
python -m benchmark.evaluation.run_uniform --model internvl3-14b --num_frames 8
```

This writes one JSON object per line to the results directory, with the question key, the model,
the predicted letter, the gold letter and the metadata tags. Runs are resumable: rerunning the
same command skips questions already present in the output file.

To evaluate on a set of frames chosen by a method rather than sampled uniformly, use the manifest
runner, which reads a mapping from question key to a rendered clip:

```bash
python -m benchmark.evaluation.run_manifest --model internvl3-14b \
    --clips <clip-directory> --pipeline <name>
```

`evaluation/README.md` describes both runners, the supported backbones and the environment each
one needs.

## Adding a backbone

Add a loader to `models_registry.py` exposing `load_model()` and
`inference(video_path, question, options)`. The harness handles sampling, prompting, parsing and
scoring, so a new backbone needs no changes anywhere else. Confirm the addition on a small subset
with `--limit` before launching a full run.

## Analysis

```bash
python -m benchmark.analysis.leaderboard
```

This reads every prediction file and produces the leaderboard and the per-slice breakdowns. It
operates only on prediction files, so it never needs a GPU and reruns in seconds.
