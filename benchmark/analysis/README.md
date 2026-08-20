# Benchmark analysis

Turns prediction files into the reported tables. Reads only prediction files, so it needs no GPU
and no video, and reruns in seconds.

```bash
python -m benchmark.analysis.leaderboard
```

Produces the overall leaderboard and the per-slice breakdowns across the capability axis and the
identity-challenge axis.

## Two things this code has to get right

**Duplicate keys.** The pair of video identifier and question identifier is not unique across the
released file: a small number of source videos appear as more than one entry, and some of the
duplicated pairs carry a different correct letter. Any aggregation must therefore state its
deduplication rule and apply it consistently, because two defensible rules give different
denominators and therefore different accuracies. The rule used here is first occurrence, and the
number of rows it drops is reported.

**Identifier spaces.** Prediction files from the evaluation harness are keyed on anonymous
identifiers, while the evidence windows are keyed on real ones. Joining without mapping yields an
empty intersection, silently. See `../data/README.md`.

## Per-slice numbers

Per-slice accuracy is averaged over models rather than reported for one model, so a slice describes
the difficulty the benchmark poses to the field. Slices with few questions carry wide intervals and
should be read as directional only. Bootstrap intervals resample over videos rather than over
questions, because questions drawn from the same video are not independent.
