# Cairn

Segment-level retrieval over a persistent visual memory. Training-free. Two variants share one
idea and differ in what is stored.

## The idea

A cairn is built once and navigated back to. The video is divided into contiguous segments, each
segment is scored against the question, and the budget is spent inside the segments that score
highest. The index is built once per video; answering a question is a lookup rather than a fresh
pass over the video.

Segments are scored by the maximum similarity of their frames, not the mean. The evidence for a
question typically occupies a small fraction of a segment, so a segment earns its place because
one frame matches strongly. Averaging over a long segment dilutes exactly the signal being looked
for.

## Two variants

**Stored-embedding variant.** `run_memory_bank.py` encodes frames once into the backbone's own
visual embedding space and keeps them. Answering splices stored embeddings into the prompt, so no
video is decoded again. This is the variant that realises the encode-once property, and it is
currently implemented for InternVL only, because it manipulates that model's visual token layout
directly.

**Rendered variant.** Selecting segments and rendering frames from them produces a clip, which any
backbone can read. This is the portable variant and it is what makes the method comparable across
every backbone in the paper. It is implemented as the `chunk` query mode in
`../lantern/select_frames.py`, sharing the candidate pool and the scoring pass with Lantern so
that the two methods are compared on identical inputs. Run it from there and evaluate it through
the manifest runner.

Defaults for the stored variant, in `run_memory_bank.py`:

```
N_CHUNKS       = 8       contiguous segments per video
BANK_PER_CHUNK = 32      frames encoded per segment, so 256 frames of memory per video
N_FRAMES       = 32      frames every arm feeds at answer time
TOK            = 256     visual tokens per frame, so 8192 visual tokens per arm
```

Every arm feeds the same 8192 visual tokens, so the arms are budget matched by construction.

## Running the stored variant

Build the index first. This is a CLIP pass over the corpus and runs on CPU in about an hour.

```bash
python -m solutions.cairn.build_index --out solutions/cairn/index
```

The index stores a per-frame embedding rather than a per-segment average, so the read side can try
either pooling rule without re-encoding anything. Cost is roughly 130 KB per video.

Then run the arms:

```bash
python -m solutions.cairn.run_memory_bank \
    --arms mb_top2,mb_rand2,mb_top1,mb_rand1,mb_uniform32,mb_oracle
```

Arm names carry an `mb_` prefix, for memory bank. That is the form written into result files and
the form the analysis code joins on. The equivalent `cairn_` names are accepted on the command
line and resolve to the same arms.

- `mb_top2` the method. The two highest scoring segments.
- `mb_top1` all frames on the single highest scoring segment. An ablation on how many segments to
  keep, not a second method.
- `mb_rand2`, `mb_rand1` the controls. The same number of segments chosen at random. The
  comparison against these is the headline, because it is what isolates the retrieval signal.
- `mb_uniform32` frames spread uniformly across the whole video at the same token budget. The
  strong baseline.
- `mb_oracle` the segment that actually contains the evidence. This is a **ceiling**, computed from
  answer-informed annotations. It bounds how much better retrieval could get. It is never a method
  and must never be quoted as an achievable accuracy.

## Running the rendered variant

```bash
python -m solutions.lantern.select_frames --query_mode chunk --out conditioned_keyframes/chunk
python -m benchmark.evaluation.run_manifest --model internvl3-14b \
    --clips conditioned_keyframes/chunk --pipeline cairn_rendered
```

Use the `random` and `uniform` arms from `../lantern/README.md` as its controls. They are the same
controls, drawn from the same pool, which is what makes Lantern and Cairn directly comparable.

## Reading the result

Compare the method against its random control first. Then compare it against the budget-matched
uniform arm, which is a harder test and the one a reviewer will ask about. Report the second even
when it is not significant, as a bound rather than as an absence.

Retrieval quality is directly measurable here and should be reported alongside accuracy: the rate
at which a retrieved segment contains the evidence window, against the same rate for the random
control. That number says whether the retrieval signal works, independently of whether the
backbone can then use what it retrieved.

## Known limits

Keeping one segment is worse than keeping two. At the hit rates observed here the second segment
is doing real work, so concentrating the whole budget on a single guess is not the better trade.

The stored-embedding variant runs on a smaller question subset than the rendered variant, so its
minimum detectable effect is looser. State the sample size with every number from it.
