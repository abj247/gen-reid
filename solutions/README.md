# Solutions

Two training-free methods, one per directory, plus the code they share.

```
lantern/    Frame-level query-conditioned selection.
cairn/      Segment-level retrieval over a persistent visual memory.
shared/     Code used by both, and the analysis that compares them.
```

There are exactly two methods. `shared/` is support code, not a third method.

## Why two

The diagnosis in the paper is that both the sampling failure and the memory failure have one
cause: the visual token budget is allocated before the question is read. The remedy is to allocate
it afterwards. That leaves one thing open, which is the unit of allocation, and these two methods
are the experiment on that question rather than two independent attempts at the same goal.

Both score a pool of candidate frames against the question with the same frozen CLIP encoder, use
the same query text, draw from the same candidate pool, and spend the same budget. They differ
only in the size of the unit they retrieve.

**Lantern** keeps the highest-scoring individual frames, wherever they fall in the video.

**Cairn** splits the video into contiguous segments, keeps the segments whose best frame scores
highest, and samples inside them. At a larger budget the segments are encoded once into a stored
visual memory that later questions read from without decoding video again.

Because everything except the retrieval unit is held fixed, the difference between them measures
granularity. That is the reason both are in the repository, and the reason they share `shared/`
rather than each carrying a private copy of the scoring code.

## Controls

Neither method may be compared against a published baseline row. Each ships two controls, and both
are required before any gain can be claimed.

**random.** Draw the same number of frames from the same candidate pool, using a per-question
seed, with no reference to the query. This isolates query conditioning from the budget. It is the
decisive control: a query-conditioned selector in an earlier iteration of this project lost to its
own random control, which is only visible if the control is run.

**uniform.** Render plain uniform frames through the identical pipeline and evaluate them the same
way. This isolates the method from the rendering path, whose cost is model-specific and large for
some backbones. See `benchmark/evaluation/README.md`.

Report a method against both. The gain over `random` is the value of query conditioning; the gap
between `uniform` here and the published uniform row is the rendering offset, which belongs to the
harness rather than to the method.

## Running them

Each method has its own README with the full argument list, the pipeline it expects and the
commands in order. Start there:

- `lantern/README.md`
- `cairn/README.md`

The analysis that produces the comparison between them, including the figures in the paper, is in
`shared/analysis/README.md`.
