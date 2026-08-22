# External benchmarks

MLVU and LongVideoBench-long, converted to the same question schema the PersistQA harness
reads, so both solutions run on them without a separate code path.

```
mlvu.json                   1,242 videos, 2,174 questions
longvideobench_long.json      395 videos,   976 questions
convert_external.py         regenerates the two files from their upstream releases
bench_filters.py            shared subset and exclusion helpers
```

## Three differences from PersistQA that change how results must be read

**Four options, not eight.** Chance is 25 percent here against 12.5 percent on PersistQA, so
accuracies are not comparable across the two benchmarks and must never be placed in one column.

**No referent phrase.** These benchmarks do not name a person to track, so Lantern's `referent`
query mode has nothing to condition on and runs as `question` instead. The mode that carries the
headline result on PersistQA therefore does not exist here, and that is a limitation of taking the
method off its home benchmark rather than a tuning choice.

**Video ids can contain a path separator.** MLVU ids look like `1_plotQA/1`. Any code writing a file
named after an id must sanitise it: `cv2.VideoWriter` does not raise when the parent directory is
missing, it returns a writer that silently discards every frame, and the run finishes with a full
manifest and no clips on disk.

## Running a solution on them

```bash
python solutions/lantern/select_frames.py \
    --bench benchmark/data/external/mlvu.json --query_mode question \
    --out external_clips/mlvu_lantern

python -m benchmark.evaluation.run_manifest \
    --model internvl3-14b --clips external_clips/mlvu_lantern --pipeline mlvu_lantern
```

Use `--query_mode chunk` for Cairn's portable variant, and `random` and `uniform` for the two
controls. Cairn's stored-embedding variant is InternVL-only and is not used here.

These numbers are not comparable to published leaderboards for either benchmark: the prompt and the
answer parser are ours, not theirs.
