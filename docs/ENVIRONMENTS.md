# Environments

Most backbones run in the environment defined by the top-level `requirements.txt`. Several ship
dependencies that conflict with it and need their own. Attempting to run everything in one
environment will fail at model load rather than silently degrade, so the failure is easy to
recognise.

## Default environment

Covers the benchmark harness, both solutions, all analysis and figures, and the majority of
backbones.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Backbones needing a separate environment

Create one environment per group below and install that group's own requirements from its checkout
under `external/`. Point the evaluation harness at that environment's interpreter when running
those models; the SLURM templates take the environment name as a variable for this reason.

- **VideoChat-Flash** (2B and 7B). Needs its own transformers pin and its own vision tower.
- **LongVU**, **Video-XL**, **Video-XL-Pro**, **MA-LMM**. Each pins an older transformers release
  and its own model code.
- **BIMBA**, **TimeViper**. State-space backbones with custom kernels; both need their upstream
  checkout on the import path.

## Offline operation

Runs on a cluster without outbound network should export the following so that model loading uses
only the local cache:

```bash
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
```

## GPU memory

The largest backbones are loaded in 4-bit where necessary. When a model is quantized this is a
property of that run and must be disclosed next to its number, because it is not comparable with a
half-precision row of the same model.
