# PersistQA

PersistQA is a benchmark for identity-grounded question answering in long videos, together with
two training-free methods that improve accuracy on it.

Each benchmark question names a person by visual description at one moment and asks about a
fine-grained attribute of that same person at a different moment. Answering therefore requires
carrying an identity across an intervening shot boundary and then perceiving a detail at the far
end. Existing long-video benchmarks reward coarse temporal gist and do not isolate this ability.

This repository contains everything needed to reproduce the benchmark, the evaluation of every
model reported in the paper, both solutions, and every figure.

## Repository layout

```
persistqa/          Shared library. Path resolution and paired significance testing.
benchmark/          The benchmark: data, evaluation harness, construction, analysis.
solutions/          Two methods, each in its own directory, plus the code they share.
  lantern/          Frame-level query-conditioned selection.
  cairn/            Segment-level retrieval over a persistent visual memory.
  shared/           Code used by both, including the mechanism analysis that compares them.
paper/              LaTeX source and generated figures.
scripts/            Cluster and maintenance utilities.
docs/               Environment setup and third-party model notes.
archive/            Superseded experiments retained on disk. Not part of the released code.
```

Documentation is layered. This file states what exists and how to install it. Each directory
carries a README that explains its own contents, and the explanation becomes more detailed the
deeper you go: `solutions/README.md` explains why there are two methods, and
`solutions/lantern/README.md` explains how that one method works and how to run it.

## The two methods

Both apply the same retrieval signal at a different granularity, and share a candidate pool, a
query, a frame budget and a pair of controls. The names are deliberately about wayfinding rather
than about question answering.

**Lantern** selects individual frames. It scores a pool of candidate frames against the question
and keeps the highest scoring ones wherever they fall, illuminating a handful of moments per
question.

**Cairn** selects contiguous segments and stores them. It builds a durable index over the video
once, then returns to the segments the question points at. A cairn is a marker you build once and
navigate back to, which is the property that separates it from Lantern.

Because the two differ only in the size of the retrieval unit, comparing them measures
granularity rather than comparing two unrelated systems.

## Installation

Python 3.10 or newer is required.

```bash
git clone <repository-url> persistqa
cd persistqa
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Installing in editable mode puts `persistqa`, `benchmark` and `solutions` on the import path, which
is what the entry points below assume.

Several backbones evaluated in the paper ship mutually incompatible dependencies and cannot share
one environment. See `docs/ENVIRONMENTS.md` for which model needs which environment, and
`docs/EXTERNAL_MODELS.md` for the third-party checkouts that live under `external/`.

## Configuration

Two environment variables control where things live. Neither is required if you run from a clone
and keep videos at the default location.

```bash
export PERSISTQA_ROOT=/path/to/persistqa        # inferred from the package location if unset
export PERSISTQA_VIDEO_DIR=/path/to/videos      # directory of source .mp4 files
```

The benchmark videos are not redistributed. `benchmark/data/README.md` explains how to obtain them
from their source corpora and how identifiers map onto files.

## Quickstart

Evaluate one backbone on the benchmark with the default uniform frame budget:

```bash
python -m benchmark.evaluation.run_uniform --model internvl3-14b --num_frames 8
```

Run Lantern end to end, which selects frames and then evaluates on the selection:

```bash
python -m solutions.lantern.select_frames --query_mode referent --out conditioned_keyframes/referent
python -m benchmark.evaluation.run_manifest --model internvl3-14b \
    --clips conditioned_keyframes/referent --pipeline lantern
```

Run Cairn, which builds the index once and then answers from it:

```bash
python -m solutions.cairn.build_index --out solutions/cairn/index
python -m solutions.cairn.run_memory_bank --arms cairn_top2,cairn_rand2,cairn_uniform32
```

Every method must be compared against its controls rather than against a published row. The
controls, and the reason each exists, are described in `solutions/README.md`.

## Reproducing the paper

`paper/iclr2026_submission/` holds the LaTeX source. The command sequence that regenerates every
number and figure in it is given in `solutions/shared/analysis/README.md`.

## Citation

See `CITATION.cff`.

## License

Code is released under the MIT License, in `LICENSE`. The benchmark videos are not redistributed
and remain subject to the licences of CinePile, MovieChat-1k and LVU.
