# Maintenance notes

## Legacy files at the repository root

A small number of files remain at the repository root that are duplicated inside the new layout:

```
eval_conditioned.py            duplicate of benchmark/evaluation/run_manifest.py
evaluate_vlm_bm.py             duplicate of benchmark/models_registry.py
probe_video_judge_v2.py        duplicate of benchmark/video_io.py
gen_keyframe_clips.py          duplicate of solutions/lantern/select_frames.py
run_kf_eval.slurm              duplicate of solutions/lantern/slurm/evaluate.slurm
kf_queue_feeder.sh             duplicate of solutions/lantern/slurm/queue_feeder.sh
combined_all_hard_v3_retagged.json   duplicate of benchmark/data/persistqa.json
video_id_mapping.json          duplicate of benchmark/data/video_id_mapping.json
```

They exist because an evaluation campaign was in flight when the repository was reorganised, and
the queued jobs reference these paths. They are not the canonical copies and they receive no
further changes.

Once no job references them, remove them:

```bash
git rm -f eval_conditioned.py evaluate_vlm_bm.py probe_video_judge_v2.py \
          gen_keyframe_clips.py run_kf_eval.slurm kf_queue_feeder.sh \
          combined_all_hard_v3_retagged.json video_id_mapping.json
```

Confirm the queue is empty first.

## The archive directory

`archive/` holds superseded experiments, earlier iterations of the question set, and result trees
from lines of work that did not reach the paper. It is excluded from version control and is not
part of the released code. It is retained on disk because several negative results reported in the
paper were measured there, and removing it would make those claims unverifiable.

Nothing under `archive/` is imported by anything under `benchmark/`, `solutions/` or `persistqa/`.
That property is worth checking before any cleanup:

```bash
grep -rn "archive/" benchmark solutions persistqa --include=*.py
```
