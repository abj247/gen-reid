# Two Paths, One Signal: Frame-Level Selection vs Chunk-Level Retrieval

*Serves as both the Notion page and the paper's method + analysis section. Data frozen 2026-08-19.*

**Reporting discipline used throughout.** Every comparison is **paired on the intersection of the
keys both arms answered**, and the intersection size `n` is stated on every number. Every test is the
**exact McNemar** test (two-sided `binomtest` on the discordant pairs `b` vs `c`) — never chi-square,
never an unpaired test on overlapping strata. Every non-significant result is written as
**"no difference larger than *X* points at 80% power"**, where *X* is the minimum detectable effect
at 80% power / two-sided alpha = 0.05 for that pairing's observed discordance. Every row states its
**frame or token budget**. Every quantity computed against the human-annotated evidence windows
(hit rate, depth, dose curve, mediation, chunk-hit, `mb_oracle`) is **answer-informed**: it is a
**diagnostic that explains an already-measured gain, never a method and never an achievable
accuracy**. It is labelled as such at every point of use.

> **Status.** The selection geometry and all mechanism analysis are **final**: `selections_tol.jsonl`
> covers 3,233 questions x 4 modes over 405 of 449 videos, 2,962 of them carrying an evidence window.
> The accuracy arms are **final at n = 3,233 on all four backbones** (InternVL3-14B, Qwen2.5-VL-7B,
> Ovis2.5-9B, VideoChat-Flash-7B): VideoChat-Flash's 8-frame arms have since finished and hold the
> same 3,233 keys as the other three, so section 9 now quotes completed numbers. What remains open on
> that backbone is bookkeeping, not measurement, and only these three rows still carry
> `[PROVISIONAL]`: the collector table (`RESULTS_MASTER.md`, stale at n = 2,341), the native-pipeline
> comparison (real-keyed, n = 3,160), and its mediation percentage in section 4.2, which was computed
> on the partial arms and has not been recomputed.

---

## 0. The four questions, answered head-on

### 0.1 WHAT the two paths are

Both paths answer the same question — *given a fixed visual budget, which part of the video should
the VLM see?* — and both answer it with the same CLIP retrieval signal. They differ in the unit they
retrieve.

* **Path 1, frame-level selection (`referent` / "keyframe").** Sample a 64-frame candidate pool
  uniformly over the video. Score every candidate frame against the query with CLIP. Keep the
  **top-8 frames by similarity**, re-sort them into temporal order, hand those 8 frames to the VLM.
  The 8 frames need not be contiguous and need not be spread: the score decides.
* **Path 2, chunk-level retrieval (`chunk` / memory bank).** Split the video into **8 contiguous
  chunks**. Score each chunk by the **max** CLIP similarity over its frames (max, not mean: the
  evidence occupies a median 9.9 s inside a median 20.7 s chunk, so a chunk earns its place because
  *one* frame matches, not because the average matches). Keep the **top-2 chunks** and sample frames
  evenly *inside* the winners. Path 2 ships in two implementations: a frame-rendered twin that runs
  on every backbone at the same 8-frame budget, and a true **memory bank** that encodes the video
  once into LLM-embedding space and splices stored tokens at question time.

### 0.2 HOW they differ

**Both paths use the SAME CLIP retrieval signal (ViT-B-32-quickgelu, `openai` weights), the SAME
query text (question + temporal anchor, byte-identical across the two paths), the SAME 64-frame
candidate pool, the SAME 8-frame budget and the SAME two controls. They differ in exactly ONE thing:
the granularity of the retrieval unit — a frame, or a contiguous eighth of the video.** That single
difference is what makes the head-to-head a **measurement of granularity** rather than a comparison
of two unrelated systems.

The consequence of the granularity choice, stated as geometry rather than as a slogan: Path 1 can
place its 8 frames anywhere and lets the score decide how concentrated the selection is; Path 2 must
**pre-commit** to a fixed one-eighth granularity and to contiguity **before** it knows how much
concentration this question needs.

### 0.3 WHY we use these two paths

Three reasons, in decreasing strength.

1. **To measure the granularity axis with everything else held fixed.** One retrieval signal, two
   points on one axis, shared controls. Any difference in outcome is attributable to granularity and
   to nothing else. This is the only comparison in the study where that is true.
2. **Because they answer different deployment questions.** Path 1 answers *"one question, tight
   budget: which 8 frames?"*. Path 2 answers *"a video I will be asked about repeatedly: what do I
   store?"*. The second question is real, and the memory bank is the only arm here that never
   re-decodes the video.
3. **Because the chunk ranker demonstrably works, so *any* chunk-level under-performance is
   attributable to the unit and not to the ranker.** Top-2 chunk retrieval lands on the annotated
   evidence chunk **48.62%** of the time vs **25.56%** for its random control (**+23.06 points, exact
   McNemar p = 3.88e-21, n = 759**; ORACLE DIAGNOSTIC). That isolates whatever loss there is to the
   *unit*, which is exactly the axis being measured. Note that the antecedent is not assumed here:
   `keyframe - chunk` is a bounded null on two of the four backbones (+0.43, p = 0.492 and +0.80,
   p = 0.233; section 3) and on all three backbones on the n = 2,962 mechanism subset (section 5.5),
   so "chunk-level under-performs" is itself only partly established.

### 0.4 The MOTIVATION

Long-video VLMs are budget-bound: 8 to 32 frames stand in for tens of thousands. The field's default
is uniform sampling, which is question-blind. The obvious fix is retrieval, and the moment you decide
to retrieve you face a choice nobody in the literature isolates: **retrieve frames, or retrieve
segments?** Every memory-bank / RAG-over-video system silently picks segments (they are what you can
store and re-read cheaply); every frame-selector silently picks frames. Because the two families
differ in backbone, budget, index and benchmark, the granularity question has never been asked with
everything else held fixed. This document asks it: same signal, same query, same pool, same budget,
same controls, one variable.

The finding that came out is not the one we set out to test. **Both of our original hypotheses are
dead** (section 8): hitting the evidence is only 37.3% of the story, and there is no breadth/depth
trade-off at all. What survives is a cleaner mechanism — **scale matching**, section 5.4 — and it is
the reason the two paths belong in the same paper rather than in two.

---

## 1. The common substrate

Shared query construction: `query = question_text + " " + temporal_anchor`. Mode `chunk` reuses the
`referent` query **verbatim**; the query text is byte-identical across paths, verified in code at
`gen_keyframe_clips.py:83-86`. Shared candidate pool: `linspace(0, N-1, 64)` produced by a single
`select_pool_indices()` used by all four modes (`gen_keyframe_clips.py:90`), so the arms cannot drift
apart in how many frames they keep or in what order.

Shared controls — this is what makes the comparison legal:

| Control | What it is | What it isolates |
|---|---|---|
| `random` | 8 frames drawn at random from the **same** 64-frame pool, RNG seeded per question key | **Query conditioning.** Removes CLIP and the query; keeps pool, budget and pipeline. |
| `uniform` | plain uniform-8, rendered through the **identical** mp4 pipeline | **The pipeline.** Removes the decoder/container/resolution confound. With `n_candidates = 64, topk = 8`, `63/7 == 9`, so uniform-8 is an **exact subset** of the pool at positions `[0, 9, ..., 63]` — literally the frames the baseline sees, not an approximation. |

Both controls are seeded per question, not per video, so two questions on the same video draw
differently. Both paths are scored against both controls.

The two controls agree with each other on every backbone, which is what makes the
query-conditioning reading credible (n = 3,233, paired, 8 frames each):
InternVL3-14B `random - uniform` +0.15 (p = 0.834; **no difference larger than 1.65 points at 80%
power**); Qwen2.5-VL-7B -0.46 (p = 0.474; **no difference larger than 1.70 points**);
Ovis2.5-9B +0.12 (p = 0.892; **no difference larger than 1.91 points**).

**Benchmark and attrition.** Declared: 3,667 MCQ, 8 options, chance = 12.5%, 449 videos
(148 CinePile + 170 MovieChat-1k + 131 LVU). Evaluated in every 8-frame arm: **n = 3,233** questions
over 405 videos. The missing **434 questions (11.8%) are not yet explained** and must be accounted
for before submission (limitation 8, section 10). All four backbones hold the same 3,233 keys, so
their contrasts run on one question set.

**Scale of the material** (n = 2,962 questions with a window, measured from the selection dump):
median video 165.5 s (p25 142.8, p75 213.5); one chunk = one eighth = **median 20.7 s**; the
evidence window occupies a **median 9.9 s = 3.4% of the video** after the tolerance correction of
section 2. Uniform-8 therefore places one frame every ~20.7 s against a ~9.9 s target. These three
numbers are the whole mechanism of section 5.4 and are worth remembering now.

---

## 2. Measuring against the evidence windows: why the windows were widened

This section is methodology, and it must be in the paper, because every mechanism number depends on
it. It is not a fudge, and the raw-window results are reported alongside as a robustness column
throughout.

**The problem.** The oracle windows in `analysis3/evidence_windows.json` were localised on a **dense
frame grid**: an annotation names one or more `dense_frames`, which map to `video_frames`. The grid's
median step is **145.7 video frames (~6 s)**, so the annotation *cannot resolve the evidence to
better than about +/- 73 frames, ever*. Worse, **1,491 of the 2,962 questions name a SINGLE dense
frame**, so `t0 == t1` and the raw window is **one frame wide**. Asking whether 8 selected frames
land on one exact frame out of several thousand measures nothing, and the measurement says so:

| raw window type | n | referent | chunk | random | uniform | reading |
|---|---|---|---|---|---|---|
| **point** (t0 == t1, 1 frame wide) | 1,491 | 1.81 | 1.88 | 0.40 | **2.68** | noise — the **control wins** |
| **span** (t1 > t0, median 10.34 s) | 1,471 | **55.40** | 48.81 | 44.87 | 51.26 | signal |

(cells are hit rate %, all arms 8 frames). The pooled raw hit rate is a 50/50 mixture of a real
measurement and a near-zero one. That is not a property of the selection methods; it is an artefact
of treating a grid-resolution point estimate as an exact instant, and it dilutes every quantity that
is mediated through `hit`.

**The fix.** Widen **every** window by the grid half-step. The tolerance is derived **per video**
from that video's own dense-to-video frame mapping wherever two distinct dense frames exist, and
falls back to the corpus median step otherwise (per-video tolerance: median 64 frames, mean 82,
range 22-387; corpus median half-step 73 frames ~ 2.7 s). Span windows are widened too, because
their endpoints come from the same dense frames and carry the same uncertainty; applying the
tolerance uniformly is the consistent choice, and applying it only to the degenerate half would bake
in two different precisions. **The tolerance is a property of the annotation and was never tuned
against accuracy.** Code: `analysis3/selanal/widen_windows.py`. After widening, the median window is
9.9 s = 3.4% of the video, i.e. still a small target.

**What changes, stated openly.** Widening moves the mechanism numbers a lot, so both columns are
reported everywhere:

| quantity (InternVL3-14B where backbone-specific) | PRIMARY (tolerance-corrected) | ROBUSTNESS (raw windows) |
|---|---|---|
| hit rate: referent / chunk / random / uniform | 57.73 / 51.76 / 41.39 / 42.51 | 28.43 / 25.19 / 22.48 / 26.81 |
| referent - random hit rate | +16.34, p = 1.51e-38, n = 2,962 | +5.94, p = 1.40e-11, n = 2,962 |
| referent - uniform hit rate | +15.23, p = 1.29e-35, n = 2,962 | +1.62, p = 0.0582 — **no difference larger than 2.35 points at 80% power** |
| referent - chunk hit rate | +5.98, p = 4.73e-10, n = 2,962 | +3.24, p = 5.75e-06, n = 2,962 |
| **chunk - uniform hit rate** | **+9.25, p = 3.68e-13, n = 2,962** | **-1.62, p = 0.0798 — SIGN FLIP; no difference larger than 2.54 points at 80% power** |
| chunk - random hit rate | +10.36, p = 7.11e-16, n = 2,962 | +2.70, p = 0.00292, n = 2,962 |
| E[depth given hit]: referent / chunk | 2.002 / 1.862 | 2.129 / 1.999 |
| percent of the referent-vs-random gain mediated by hitting | 37.3% [21.4, 70.6] | 9.5% [3.2, 21.0] |
| acc-given-hit, referent - random, paired both-hit | +2.62, p = 0.0495, n = 762 | +3.13, p = 0.0854, n = 415 — **no difference larger than 4.61 points at 80% power** |
| in-window frames beyond the first: **depth >= 2 vs depth == 1** (mode-balanced; a binary contrast, NOT a per-frame slope) | **+4.57 pts**, CI [2.72, 6.50], p = 1e-4, MDE 2.69, n = 5,667 pairs / 894 questions | +2.98 pts, CI [0.47, 5.62], p = 0.018, MDE 3.70 |
| per-extra-frame slope, FE `(depth-1)` coefficient | +2.13 pts/frame, se 0.42, p = 4.40e-07, MDE 1.18 | +1.29 pts/frame, se 0.53, p = 0.0147, MDE 1.48 |
| saturation verdict | does NOT saturate (as a step from 1 to >= 2 frames) | **UNDERPOWERED** -- +2.98 pts is smaller than its own MDE 3.70, so no verdict is reported (the raw figure prints exactly this) |

Four of these change status between the columns (referent - uniform hit rate, **chunk - uniform hit
rate**, the both-hit acc test, and the size of the mediated fraction). **Whoever quotes a mechanism
number must quote the pair.** The two columns agree on sign for every `referent` contrast, on the
*ordering of the four modes on depth*. The *saturation verdict does NOT carry over*: it is
established on the primary windows and underpowered on the raw ones. **They disagree in sign on
`chunk - uniform`: +9.25 (p = 3.68e-13) on the tolerance-corrected windows against -1.62 (p = 0.0798)
on the raw windows, where uniform-8 out-hits chunk outright (26.81 vs 25.19).** The concentration
inverted-U of section 5.3 is therefore a property of the tolerance-corrected windows and **does not
survive intact on raw windows**; the raw-window figure pair shows this without smoothing, with chunk
sitting below uniform-8 in panel (a). `chunk - random` keeps its sign but loses most of its size
(+10.36 primary vs +2.70, p = 0.00292, raw). This is the one place where the widening does not merely
change magnitudes, and it is carried into limitations 5 and 6.

---

## 3. Accuracy: the measured result, before any mechanism

All arms **8 frames**, paired, exact McNemar, **n = 3,233**, chance = 12.5%. `base` is uniform-8
through the identical mp4 pipeline.

| Backbone | base | `random` | chunk (Path 2) | keyframe (Path 1) | chunk - base | keyframe - base | keyframe - `random` | keyframe - chunk |
|---|---|---|---|---|---|---|---|---|
| InternVL3-14B | 24.62 | 24.47 | 26.94 | **27.37** | **+2.32** p = 2.36e-04 | **+2.75** p = 1.93e-05 | **+2.91** p = 9.03e-06 | +0.43 p = 0.492 |
| Qwen2.5-VL-7B | 17.14 | 17.60 | 17.94 | **19.27** | +0.80 p = 0.215 | **+2.13** p = 1.06e-03 | **+1.67** p = 0.01015 | **+1.33** p = 0.0274 |
| Ovis2.5-9B | 22.86 | 22.73 | 24.56 | **25.36** | **+1.70** p = 0.0173 | **+2.51** p = 4.35e-04 | **+2.63** p = 3.21e-04 | +0.80 p = 0.233 |
| VideoChat-Flash-7B | 20.07 | 20.14 | 20.88 | **23.01** | +0.84 p = 0.231 | **+2.94** p = 2.44e-05 | **+2.88** p = 6.28e-05 | **+2.10** p = 0.00108 |

MDEs for the same cells, in accuracy points at 80% power: chunk - base 1.75 / 1.75 / 1.97 / 1.88;
keyframe - base 1.79 / 1.80 / 1.97 / 1.93; keyframe - `random` 1.82 / 1.79 / 2.03 / 2.00;
keyframe - chunk 1.64 / 1.65 / 1.82 / 1.78.

The VideoChat-Flash row is new: those arms finished after the first draft of this document and now
hold the same 3,233 keys as the other three backbones (section 9, which also records a one-discordant-
pair difference between the run owner's numbers, quoted here, and an independent recomputation from
the prediction files; neither reading changes a verdict).

Reading the nulls correctly:

* Qwen, chunk - base: **no difference larger than 1.75 points at 80% power** (p = 0.215, n = 3,233).
* InternVL, keyframe - chunk: **no difference larger than 1.64 points at 80% power** (p = 0.492).
* Ovis, keyframe - chunk: **no difference larger than 1.82 points at 80% power** (p = 0.233).
* VideoChat-Flash, chunk - base: **no difference larger than 1.88 points at 80% power** (p = 0.231).

**The headline contrast is `keyframe - random`, not `keyframe - base`**: it is the only one that
isolates query conditioning from pool and pipeline. It is positive and significant on **all four**
backbones, at **p = 9.03e-06 / 0.01015 / 3.21e-04 / 6.28e-05**. The blanket phrase *"all p <= 0.01"*
is **false** — Qwen is p = 0.01015. If one bound is wanted, use **"all p <= 0.011"**, and say that
Qwen's contrast *just* survives.

**Head-to-head, honestly.** At 8 frames Path 1's point estimate is ahead of Path 2 on all four
backbones and the margin is significant on **two of four** (Qwen +1.33, p = 0.0274;
VideoChat-Flash +2.10, p = 0.00108). On the other two the correct statement is a bounded null
(no difference larger than 1.64 and 1.82 points at 80% power). So: *"at an 8-frame budget,
frame-level is at least as good as chunk-level on all four backbones, and strictly better on two."*
Not "frame-level wins".

### 3.1 Ablation: does the temporal anchor earn its place?

`kf_question` is the identical pipeline with a **question-only** CLIP query; `kf_referent` appends
the temporal anchor. Same 8-frame budget, same pool, paired, n = 3,233:

| Backbone | `kf_question` | `kf_referent` | anchor - no anchor | p | MDE |
|---|---|---|---|---|---|
| InternVL3-14B | 26.42 | 27.37 | **+0.95** | 0.0185 | 1.11 |
| Qwen2.5-VL-7B | 19.12 | 19.27 | +0.15 | 0.779 | 1.23 |
| Ovis2.5-9B | 25.36 | 25.36 | 0.00 | 1.000 | 1.22 |

**Statement.** The anchor pays on **one backbone of three**. On Ovis the two arms are numerically
identical (**no difference larger than 1.22 points at 80% power**), on Qwen they are within noise
(**no difference larger than 1.23 points**). **Almost all of the Path-1 gain is carried by the
question alone** — `kf_question` still beats `random` by +1.95 (p = 0.00272), +1.52 (p = 0.0181) and
+2.63 (p = 3.37e-04) on the three backbones, n = 3,233. The anchor is a small InternVL-specific
increment, not a component the method depends on.

### 3.2 Budget decay: the gain is a small-budget gain

Two statements, one reproducible from repo artefacts today and one not.

*Reproducible now* (InternVL3-14B, paired, exact McNemar; `kf_q_t16` / `kf_q_t32` are the
question-only selector at 16 / 32 frames and are **NOT budget-matched** to the 8-frame arms):

| contrast | budgets | n | delta | p | MDE |
|---|---|---|---|---|---|
| `kf_q_t16` vs `kf_referent` | 16 f vs 8 f | 3,233 | +0.46 | 0.429 | 1.53 |
| `kf_q_t32` vs `kf_referent` | 32 f vs 8 f | 3,059 | **+2.32** | 8.79e-05 | 1.64 |
| `kf_q_t32` vs `kf_q_t16` | 32 f vs 16 f | 3,059 | **+1.83** | 2.70e-04 | 1.39 |

Doubling the budget from 8 to 16 buys nothing detectable (**no difference larger than 1.53 points at
80% power**), but 32 frames beats the 8-frame selector by **+2.32 points**. **Spending 4x the budget
beats choosing well at 1x the budget.**

*Quoted from the prior round, comparator not in the repo.* The selection **gain** ladder on
InternVL3-14B was reported as **+2.14 at 8 f, +0.51 (n.s.) at 16 f, +1.08 (p = 0.096) at 32 f**.
There is **no `uniform16` or `uniform32` arm in `results_baseline/`**, so this ladder cannot be
regenerated from any artefact here; it carries no n and no MDE, and its two nulls cannot be phrased
as bounded nulls. It is retained only as a marker of unfinished work (section 12.1, item 4). Either
the matched 16- and 32-frame uniform controls get run, or this ladder is dropped from the paper.

The claim the shipped evidence licenses is **"better frame *choice* at a small budget"**. It does not
license "better at any budget", and no such claim is made.

---

## 4. Path 1 — the mediation result

**Figure:** `fig_path1_evidence_mediation.png` / `.pdf` (source: `make_figures.py`; the same script
with `--windows raw` writes the raw-window robustness twin `fig_path1_evidence_mediation_rawwindows.*`).
Panel (a) evidence-window hit rate by mode with 95% Wilson CIs and the paired exact-McNemar contrasts
bracketed; panel (b) the exact decomposition of the referent-vs-random accuracy gain as a waterfall —
the part routed through the hit rate against the part routed through conditional accuracy — each with
key-clustered bootstrap CIs. The whole figure is stamped ORACLE-WINDOW DIAGNOSTIC.

The question this section answers: **when query-conditioned frame selection wins, what is it winning
at?** The naive answer — "it looks in the right place" — is testable, and it is only partly right.

### 4.1 It does look in the right place

All arms 8 frames, n = 2,962 questions with a window, paired exact McNemar on the hit indicator:

| mode | hit rate % | 95% Wilson CI |
|---|---|---|
| referent (Path 1) | **57.73** | [55.94, 59.50] |
| chunk (Path 2) | 51.76 | [49.95, 53.55] |
| uniform | 42.51 | [40.74, 44.29] |
| random | 41.39 | [39.63, 43.17] |

| contrast | delta (points) | p | n |
|---|---|---|---|
| referent - random | **+16.34** | 1.51e-38 | 2,962 |
| referent - uniform | **+15.23** | 1.29e-35 | 2,962 |
| referent - chunk | **+5.98** | 4.73e-10 | 2,962 |
| chunk - random | **+10.36** | 7.11e-16 | 2,962 |
| chunk - uniform | **+9.25** | 3.68e-13 | 2,962 |

Unlike the previous round, the `referent` hit-rate advantage is now established against **both**
controls, not only against `random` (raw-window robustness: against `uniform` it is a null, +1.62,
p = 0.0582, **no difference larger than 2.35 points at 80% power** — see section 2). The two `chunk`
rows do not carry over as cleanly: on raw windows `chunk - uniform` is **-1.62 (p = 0.0798), a sign
flip**, and `chunk - random` shrinks to +2.70 (p = 0.00292). Both `chunk` rows here are properties of
the tolerance-corrected windows (section 2, limitation 5).

### 4.2 But looking in the right place is only 37% of it

The decomposition is an exact identity — total gain = (change in hit rate) x (hit-minus-miss accuracy
gap) + (average hit rate x change in acc-given-hit + average miss rate x change in acc-given-miss) —
so the residual is 0 by construction (measured: 0.00e+00). InternVL3-14B, referent vs its random
control, n = 2,962 paired questions with a window, 8 frames per arm:

| quantity | referent | random | delta |
|---|---|---|---|
| accuracy | 27.18 | 24.17 | **+3.00** (exact McNemar p = 1.14e-05) |
| P(hit) | 0.5773 | 0.4139 | **+0.1634** |
| accuracy given hit | 31.17 (n = 1,710) | 26.67 (n = 1,226) | +4.50 |
| accuracy given miss | 21.73 (n = 1,252) | 22.41 (n = 1,736) | -0.68 |

| channel | accuracy points | 95% CI (key-clustered bootstrap, 10,000 draws) |
|---|---|---|
| explained by the **hit rate** | **+1.12** | [+0.73, +1.55] |
| explained by **conditional accuracy (hit + miss strata)** | **+1.88** | [+0.50, +3.25] |
| total | +3.00 | [+1.65, +4.32] |
| **percent mediated by hitting** | **37.3%** | [21.4%, 70.6%] |

**Prediction (b), the primary test.** If hitting were the whole mechanism, acc-given-hit would be
**flat across modes**. It is not: on the **762 questions where BOTH modes hit** — the only stratum
where the same item is seen under both modes — referent beats random by **+2.62 points
(b = 57, c = 37, exact McNemar p = 0.0495, n = 762)**. A within-question permutation test on the same
stratum corroborates (p = 0.0484, 10,000 seeded draws). The unpaired Fisher test on the two hit
strata (p = 0.0085) is a **footnote only**: the strata overlap by 762 questions = 44.6% of the
referent stratum and 62.2% of the random stratum, so its independence assumption is violated and its
p is uncalibrated.

**Honest conclusion.** Query conditioning improves **WHERE we look** (+16.34 points of hit rate,
p = 1.51e-38) and, **separately, WHAT we get when we are there** (+2.62 points of acc-given-hit at
matched hit status, p = 0.0495). Hitting the evidence accounts for **37.3% [21.4, 70.6]** of the
accuracy gain, so the **unmediated share is 62.7%, 95% CI [29.4, 78.6]** — a wide interval, and the
number itself is what is measured; the reason for it is not. One untested explanation is that the
frames co-selected with an in-window frame are themselves better, because the same score that put one
frame in the window put its neighbours on the same scene, the same shot, the same faces. **This is a
conjecture; nothing measured here tests it**, and it is not used to support any claim in this
document.

**Replication and scope.** Direction replicates on all backbones (pct mediated: InternVL3-14B 37.3%
[21.4, 70.6]; InternVL3-8B 41.6% [20.8, 110.0], n = 1,604; Ovis2.5-9B 53.5% [30.4, 134.8];
Qwen2.5-VL-7B 36.9% [15.2, 131.3]; VideoChat-Flash-7B 40.3% [22.9, 88.1] `[PROVISIONAL]` — this one
mediation number was computed on that backbone's partial arms and has **not** been recomputed since
they finished, unlike its accuracy contrasts in section 9, which are final at n = 3,233). The
prediction-(b) test reaches significance only on InternVL3-14B; on the others it is a bounded null
(Ovis +0.66, p = 0.699, **no difference larger than 3.76 points**; Qwen +0.79, p = 0.624, **no
difference larger than 3.71 points**; InternVL3-8B +2.04, p = 0.322, **no difference larger than 4.94
points**). So "acc-given-hit is not flat" is demonstrated on the strongest backbone and is
**unrefuted, not established**, on the others.

---

## 5. Path 2 — the concentration result

**Figure:** `fig_path2_concentration_dose.png` / `.pdf` (source: `make_figures.py`; `--windows raw`
writes the robustness twin `fig_path2_concentration_dose_rawwindows.*`, in which chunk sits **below**
uniform-8 — the sign flip disclosed in section 2).
Panel (a) plots the four modes on one axis — distinct chunks touched, reversed so that concentration
increases rightward — against **hit rate** with 95% CIs, with marker area proportional to that mode's
accuracy and the accuracy printed in each label; the inverted U with `referent` at the peak is the
finding, and it is a finding about hit rate, not about accuracy (note 2 in section 5.3). Panel (b) is
the dose curve with Wilson CIs, n per point and the chance line, annotated with the mode-stratified
depth >= 2 vs depth == 1 result. Stamped ORACLE-WINDOW DIAGNOSTIC.

### 5.1 First, the refutation: there is no breadth/depth trade-off

The figure this section replaces was designed as a trade-off plane: Path 1 buys breadth by giving up
depth, Path 2 the reverse, and breadth wins. **The data refute that framing outright.** All arms
8 frames, n = 2,962:

| mode | hit rate % | E[depth given hit] | E[depth] | spread | distinct chunks touched |
|---|---|---|---|---|---|
| referent | **57.73** | **2.002** | **1.156** | 0.5759 | 4.113 |
| chunk | 51.76 | 1.862 | 0.964 | 0.4049 | 2.000 |
| random | 41.39 | 1.433 | 0.593 | 0.8008 | 5.448 |
| uniform | 42.51 | 1.189 | 0.505 | 1.0000 | 8.000 |

**`referent` DOMINATES `chunk` on both axes** — higher hit rate (+5.98, p = 4.73e-10, n = 2,962) *and*
higher depth (2.002 vs 1.862 frames inside the window given a hit). A trade-off plane drawn through
dominating points is a figure a reviewer kills. The trade-off framing is withdrawn.

### 5.2 And depth is not worthless either, so the hit-rate loss buys nothing

If extra in-window frames were worthless, chunk-level could be excused: it would be paying for
something it does not need. They are not worthless. The test is **mode-stratified**, because at
depth 1 the cells are dominated by one set of modes and at depth 4+ by another, so any slope fitted
across a pooled row mixes dose with mode:

| estimator | what it estimates | value | inference |
|---|---|---|---|
| **mode-balanced stratified (PRIMARY)** | **depth >= 2 vs depth == 1** — a *binary* contrast on the extra frames taken together, **not** a per-frame slope | **+4.57 pts** | 95% key-clustered bootstrap CI [2.72, 6.50], p = 1e-4, MDE 2.69, n = 5,667 pairs / 894 questions |
| FE conditional model, `(depth-1)` coefficient | **per extra frame** | **+2.13 pts/frame** | se 0.42, p = 4.40e-07, MDE 1.18, n = 35,544 obs / 2,962 clusters |
| Mantel-Haenszel over the 12 mode-pair cells | depth >= 2 vs depth == 1 | +3.83 pts | p = 1.10e-15, heterogeneity Q = 15.58, df = 11, p = 0.157 |
| mode-stratified WLS slope over the dose curve | per extra frame | +1.525 pts/frame | 95% key-clustered CI **[0.00, 3.13] — does NOT exclude zero** (`dose_slope_mode_stratified_excludes_zero = false`) |

**Verdict: `saturates = FALSE`; extra in-window frames DO pay — as a step, not as a linear slope.**
The primary +4.57 is the value of **having at least two in-window frames rather than exactly one**;
it is a contingency contrast between deep and shallow cells, and there is no "each" in it. Read as a
per-frame number it would be wrong by 2-3x and would contradict the dose curve below, which rises
24.25 -> 27.92 -> 30.69 (steps of +3.67 and +2.77) and then **falls** to 27.69 at 4+. The genuine
per-extra-frame estimates are smaller and one of them includes zero: **+2.13 pts/frame** (FE,
p = 4.40e-07) and **+1.53 pts/frame with CI [0.00, 3.13]** (mode-stratified WLS). So *depth pays* is
established as a **step from 1 to >= 2 in-window frames**, and **not** as a linear per-frame dose.
For comparison, the *first* in-window frame is worth **+4.79 points** (mode-balanced, 95% CI
[3.90, 5.69], p = 1e-4, MDE 1.28) — so the extra frames **taken together** are worth about as much as
the first one, which is a statement about the pair of steps and not about any single frame.

**The rest of that FE model, since one coefficient from it is quoted above.** In the same fit,
`mode[chunk] = -0.21` (se 0.43, p = 0.629, MDE 1.21): **conditional on hit and on depth, chunk and
referent are statistically indistinguishable**, so the referent-minus-chunk gap runs through hit rate
and depth rather than through any residual "mode" effect. `mode[random] = -1.05` (se 0.50, p = 0.037)
and `mode[uniform] = -0.97` (se 0.50, p = 0.051) stay negative, which is the query-conditioning
channel showing up where it should. On raw windows the same coefficients are -0.49 (p = 0.258; no difference larger than 1.21 points
at 80% power),
-2.00 (p = 6.7e-05) and -2.18 (p = 8.4e-06).

The dose curve behind it (pooled over the three backbones, all arms 8 frames, Wilson CIs; the pooled
CIs are nominal because rows share questions):

| frames inside the evidence window | accuracy % | n | 95% CI |
|---|---|---|---|
| 0 (miss) | 19.06 | 18,360 | [18.50, 19.63] |
| 1 | 24.25 | 11,118 | [23.46, 25.05] |
| 2 | 27.92 | 3,378 | [26.43, 29.45] |
| 3 | 30.69 | 1,323 | [28.26, 33.23] |
| 4+ | 27.69 | 1,365 | [25.38, 30.13] |

Because depth pays, Path 2's lower hit rate cannot be re-described as a deliberate purchase of depth
— referent is the deeper arm as well (2.002 vs 1.862), so there is nothing bought. **It is a loss of
hit rate that buys nothing.** Its *accuracy* cost is a weaker statement: keyframe - chunk is a bounded
null on two of the four backbones (+0.43, p = 0.492, and +0.80, p = 0.233) and on all three backbones
on the n = 2,962 mechanism subset (section 5.5).

### 5.3 The live account: one axis, concentration, and hit rate is non-monotone in it

The four modes are ordered on a single axis — how many distinct chunks of the video the 8 frames
touch:

| | uniform | random | referent | chunk |
|---|---|---|---|---|
| distinct chunks touched | 8.00 | 5.45 | **4.11** | 2.00 |
| hit rate % | 42.51 | 41.39 | **57.73** | 51.76 |
| accuracy % (mean of 3 backbones, n = 2,962) | 21.19 | 21.26 | **23.70** | 23.00 |

**Hit rate is non-monotone in concentration and peaks in the middle, at `referent`.** Both extremes
lose on hit rate; on accuracy the same shape is **suggestive only** (note 2 below), and on raw windows
the left arm of the U does not survive at all (section 2). Uniform spreads 8 frames across the whole
runtime — one frame every ~20.7 s against a ~9.9 s target — and mostly samples empty video. Chunk
commits the entire budget to 2 of 8 segments and, when it misses, misses further away: on **misses**,
the nearest selected frame is a median **25.3 s** from the window **centre** for chunk vs **12.1 s**
for referent (n = 1,429 / 1,252 misses). Provenance: the medians are `nearest_s.{chunk,referent}.miss_only.median` in
`path1_stats_tol.json`; only the miss counts and the `uniform` comparison are log-only, from the
`HOW CLOSE DOES A SELECTION GET?` block of `p1t.log` (lines 202-217). This supersedes the earlier
claim that the number appears in none of the four stats
JSONs: it is the `HOW CLOSE DOES A SELECTION GET?` block of the primary console log
`/tmp/claude-1238/-home-ab260989-gen-reid/p1t.log` (lines 202-217), and `nearest_s` there is the
distance to the window **centre**, not to the window, so a hit can carry a non-zero value
(limitation 9). The same block also shows `uniform` with the *smallest* miss distance of the four
(median 8.9 s), so miss distance tracks dispersion rather than accuracy: it illustrates the chunk
failure mode, it does not establish it.

Two honesty notes on this table, both of which a reviewer will otherwise raise:

1. **Concentration is confounded with query conditioning across all four modes** — `uniform` and
   `random` are not query-conditioned at all. The clean, unconfounded contrast on the concentration
   axis is **referent vs chunk**: same signal, same query, same pool, same budget, only the unit
   differs. There, concentration is the only variable, and referent wins the hit rate by +5.98
   (p = 4.73e-10, n = 2,962).
2. **On accuracy, the descent from the peak is significant on one backbone of three**
   (Qwen +1.33, p = 0.0274; InternVL **no difference larger than 1.64 points**, p = 0.492; Ovis **no
   difference larger than 1.82 points**, p = 0.233). The inverted U is established on **hit rate**
   and is **suggestive, not established, on accuracy**.

Mode-free support for the dispersion story, from within-mode terciles of spread (which remove the
mode main effect entirely): inside `chunk`, the most-spread tercile scores **-3.45 points** below the
least-spread one (p = 0.00108, n = 3,504 / 2,898); inside `referent` the same contrast is -1.36
(p = 0.223, **no difference larger than 3.12 points at 80% power**) and inside `random` -1.51
(p = 0.156, **no difference larger than 2.98 points**). The raw pooled spread-vs-accuracy table is
**not** usable for this: spread is near-deterministic given the mode (uniform is 1.000 by
construction), so the pooled table mostly re-expresses the mode contrasts.

### 5.4 The mechanism: scale matching

A chunk is **20.7 s** (median). The evidence is **9.9 s** (median, after the tolerance correction),
3.4% of the video. A fixed one-eighth grid is therefore **the wrong granularity by roughly a factor of
two, and it is the same wrong granularity for every question**, whether the evidence is a
half-second glance or a minute-long scene. (The 9.9 s is the tolerance-corrected median; the raw
*span* windows have a median of 10.34 s, so the ratio is not manufactured by the widening — but the
1,491 raw *point* windows have no duration at all, so "a factor of two" is a statement about the
widened windows, which are the only ones on which every question has a duration.) Committing to top-2 chunks spends the whole budget on
41.4 s = 25% of the video, chosen before anything is known about how localised this question's
evidence is.

CLIP top-k is **score-driven, not geometry-driven**. Because CLIP similarity is temporally
autocorrelated — frames near the evidence score highly together — a top-k rule concentrates itself
**exactly as much as this question's evidence warrants**, adaptively, per question, with **no
pre-commitment**. That is why `referent` reaches chunk-level depth (2.00 vs 1.86 frames in-window
given a hit) while still touching **twice as many distinct chunks** (4.11 vs 2.00). It gets
concentration for free where concentration is warranted, and keeps coverage where it is not.

**So Path 2 is not "the worse container". It is the same retrieval signal forced to pre-commit its
granularity before it knows what the question needs.** That is the sentence the head-to-head
measures, and it is the reason the comparison is interesting rather than a bake-off.

Scope of this claim: the scale-matching account is **consistent with** all the measured geometry
(dominance, non-monotonicity, the miss-distance gap, the within-chunk spread tercile), and it is
**not independently tested** — the decisive experiment would sweep chunk count (2, 4, 8, 16, 32
chunks) at fixed budget and show accuracy peaking where chunk duration matches evidence duration.
That experiment has not been run (section 12.1, item 5).

### 5.5 What the gap decomposition does and does not say

For completeness, the chunk-minus-keyframe accuracy gap decomposed into a breadth term
(change in P(hit) x value of a hit) and a depth term (change in E[depth given hit] x value per extra
frame), n = 2,962 per backbone:

| backbone | gap | breadth term | depth term | unexplained remainder | McNemar p on the gap |
|---|---|---|---|---|---|
| InternVL3-14B | -0.30 | -0.42 | -0.24 | +0.36 | 0.658 — **no difference larger than 1.71 points at 80% power** |
| Qwen2.5-VL-7B | -1.22 | -0.26 | -0.24 | -0.72 | 0.056 — **no difference larger than 1.73 points at 80% power** |
| Ovis2.5-9B | -0.57 | -0.52 | -0.31 | +0.25 | 0.420 — **no difference larger than 1.88 points at 80% power** |

The remainder is **defined** as gap minus breadth minus depth, so the three adding up to the gap is
an identity of the definition and is **not evidence of anything**; it absorbs the breadth x depth
interaction and every unmodelled channel. On the subset the gap itself is a bounded null on all
three backbones. The decomposition is reported because it is what a reviewer will ask for, and it is
labelled as descriptive.

A related question, asked and answered negatively: **does the chunk penalty hurt weak models more?**
Weakest (Qwen, uniform-8 16.91 on the subset) vs strongest (InternVL3-14B, 24.38): chunk-minus-keyframe
-1.22 vs -0.30, difference-in-differences **-0.91, 95% paired bootstrap CI [-2.53, +0.61], p = 0.269**.
**Not supported at this n.** It is not narrated.

---

## 6. Path 2's second implementation: the memory bank

InternVL3-14B only, n = 759, **all arms budget-matched at 8,192 visual tokens = 32 frames x 256
tokens per frame**. Write phase: video -> 8 chunks -> 32 frames per chunk encoded **once** into
LLM-embedding space. Read phase: question -> CLIP -> rank chunks -> splice the winning chunks'
*stored* tokens -> answer. The video is never re-decoded.

| Arm | budget | accuracy | contrast | delta | p | MDE |
|---|---|---|---|---|---|---|
| `mb_uniform32` | 8,192 tok (32 f) | **30.96** | strong baseline | — | — | — |
| `mb_rand1` | 8,192 tok (32 f) | 28.46 | control, 1 random chunk | vs `mb_uniform32` -2.50 | 0.104 | 4.09 |
| `mb_rand2` | 8,192 tok (32 f) | 28.46 | control, 2 random chunks | vs `mb_uniform32` -2.50 | 0.0842 | 3.85 |
| `mb_top1` | 8,192 tok (32 f) | 30.04 | vs `mb_rand1` | +1.58 | 0.335 | 4.21 |
| `mb_top1` | 8,192 tok (32 f) | 30.04 | vs `mb_uniform32` | -0.92 | 0.586 | 4.06 |
| `mb_top2` | 8,192 tok (32 f) | 32.28 | vs `mb_rand2` | **+3.82** | 0.00934 | 3.99 |
| `mb_top2` | 8,192 tok (32 f) | 32.28 | vs `mb_uniform32` | +1.32 | 0.353 | 3.58 |
| `mb_oracle` | 8,192 tok (32 f) | 35.05 | **CEILING — answer-informed, NEVER a method** | +4.08 vs `mb_uniform32` | 0.00353 | 3.82 |

**Retrieval quality — ORACLE DIAGNOSTIC, not a method result** (chunk-hit is scored against the
human-annotated evidence chunk, so it is answer-informed and cannot justify a method):

| arm | chunk-hit % | n | contrast | delta | p | MDE |
|---|---|---|---|---|---|---|
| `mb_rand1` | 12.25 | 759 | — | — | — | — |
| `mb_rand2` | 25.56 | 759 | — | — | — | — |
| `mb_top1` | 29.12 | 759 | vs `mb_rand1` | **+16.86** | 3.67e-16 | 5.88 |
| `mb_top2` | 48.62 | 759 | vs `mb_rand2` | **+23.06** | 3.88e-21 | 6.95 |
| `mb_oracle` | 100.0 | 759 | CEILING | — | — | — |

Accuracy given a chunk hit is 36.23% vs 26.91% given a miss (pooled over the two 2-chunk arms,
n = 1,518; **DIAGNOSTIC**) — landing in the right chunk is worth **+9.32 points** at this budget too,
which replicates the frame-level "arriving pays" finding on a different budget and a different
mechanism.

**The three limits, which must be read together and never selectively:**

1. **The retrieval-to-accuracy effect is present at k = 2 and undetectable at k = 1.**
   `mb_top2 - mb_rand2 = +3.82` (p = 0.00934) but `mb_top1 - mb_rand1 = +1.58` — **no difference
   larger than 4.21 points at 80% power** (p = 0.335, n = 759).
2. **The random control is itself worse than the trivial baseline.** `mb_rand2` 28.46 vs
   `mb_uniform32` 30.96 = **-2.50** (p = 0.0842; **no difference larger than 3.85 points at 80%
   power**, but the point estimate is negative). So the headline +3.82 is **partly the control being
   bad**, not the method being good. This caveat travels with the +3.82 everywhere.
3. **Against the strong baseline the memory bank is a null.** `mb_top2 - mb_uniform32 = +1.32`:
   **no difference larger than 3.58 points at 80% power** (p = 0.353, n = 759). This is the
   load-bearing null of Path 2.

Footnote, because every reader assumes a bug: `mb_rand1` and `mb_rand2` both read 28.46% by
**coincidence** — 216/759 correct in each, but their predictions differ on 180/759 keys and their
correctness flags on 76/759.

### 6.1 The one matched-budget cross-path comparison

At 32 frames both paths can be compared directly (InternVL3-14B, anon-to-real id mapping applied,
paired, exact McNemar; recomputed by the adversarial review, `review/REVIEW_NARRATIVE.md` section G):

| contrast | budget | n | acc A | acc B | delta (A - B) | p | MDE |
|---|---|---|---|---|---|---|---|
| `kf_q_t32` vs `mb_top2` | 32 f both | 686 | 31.05 | 31.49 | -0.44 | 0.832 | 3.85 |
| `kf_q_t32` vs `mb_uniform32` | 32 f both | 686 | 31.05 | 30.17 | +0.87 | 0.581 | 3.70 |
| `kf_q_t32` vs `mb_rand2` | 32 f both | 686 | 31.05 | 27.55 | **+3.50** | 0.0250 | 4.20 |

**At a matched 32-frame budget the granularity axis is a null: no difference between frame-level and
chunk-level larger than 3.85 points at 80% power** (p = 0.832, n = 686). A stated null with an MDE is
a legitimate finding, and this is the honest form of the "two paths bracket one axis" claim: *the
axis was measured at two budgets; at 8 frames frame-level is ahead on all four backbones and
significantly so on two, at 32 frames the two granularities are indistinguishable to within 3.85
points.* The data do not support granularity being a settled ordering.

### 6.2 Amortisation, quantified — and it does not pay off here

Stated in the natural unit, frames pushed through the vision tower (no wall-clock or FLOP has been
measured; see the honesty notes):

| | write phase, once per video | read phase, per question | total for Q questions |
|---|---|---|---|
| **Path 1** (frame selection) | 0 | 8 frames encoded | **8Q frames** |
| **Path 2** (memory bank) | 8 chunks x 32 frames = **256 frames encoded** | ~0 frames (CLIP text encode + top-2 argsort + tensor splice) | **256 + ~0 x Q ~ 256 frames** |

Break-even is `8Q = 256`, i.e. **Q > 32 questions per video**.

This benchmark has **8.16 questions per video** (449 videos; median 8, max 17); on the evaluated
subset, **7.98** (3,233 questions over 405 videos; median 8, min 1, max 17). **0% of videos reach
break-even** — the most-questioned video in the corpus asks 17, about half of what is needed.

**Therefore, at this question density Path 1 is BOTH cheaper AND more accurate.** Amortisation is a
**property of the design**, realisable only in a deployment that asks many questions of one video
(an assistant re-queried over the same footage). It is **not a benefit demonstrated by these
results**, and this document does not present it as one. Honesty notes: (i) frames encoded is a
proxy — no wall-clock second, FLOP or byte was measured, and the memory bank's **storage footprint**
is not counted at all; (ii) the ~0 read cost ignores the CLIP text encode, which is small but not
zero.

---

## 7. How the two paths differ, in one table

| Axis | Path 1 — frame-level (`referent`) | Path 2 — chunk-level (`chunk` / memory bank) |
|---|---|---|
| Retrieval unit | a single frame | a contiguous chunk, one eighth of the video (median 20.7 s) |
| Ranking signal | CLIP similarity, per frame | CLIP similarity, per chunk, aggregated by **max** |
| Granularity decision | **deferred to the score**, per question | **pre-committed**, identical for every question |
| Concentration achieved | 4.11 distinct chunks touched (adaptive) | 2.00 by construction |
| Budget | 8 frames | frame-rendered twin: 8 frames (**matched**). Memory bank: 8,192 visual tokens = 32 frames (**NOT matched** to the 8-frame arms; the one matched cross-path test is section 6.1) |
| Per-question visual encode | full — re-select and re-encode 8 frames | ~0 in the memory bank (stored tokens spliced); full in the frame-rendered twin |
| Re-decode required | **yes**, every question | **no** in the memory bank; yes in the twin |
| Portability | full — renders mp4, runs on every backbone | twin: full. Memory bank: **InternVL only** (embedding splice is backbone-specific), so granularity is confounded with implementation in the memory-bank rows |
| Amortisation over Q questions | O(Q) visual encode | O(1) visual encode + O(Q) cheap reads; break-even **Q > 32**, corpus density **8.16** |
| Accuracy, 8 f, n = 3,233 (InternVL / Qwen / Ovis / VideoChat-Flash) | 27.37 / 19.27 / 25.36 / 23.01 | 26.94 / 17.94 / 24.56 / 20.88 |
| Head-to-head, keyframe - chunk, 8 f, n = 3,233 | — | +0.43 (p = 0.492) / **+1.33 (p = 0.0274)** / +0.80 (p = 0.233) / **+2.10 (p = 0.00108)** |
| Head-to-head at matched 32 f, n = 686 | — | -0.44 (p = 0.832; **no difference larger than 3.85 points**) |

---

## 8. Both original hypotheses are dead

Stated explicitly so that neither is reintroduced by a later draft.

* **DEAD HYPOTHESIS 1 — "keyframe wins because it hits the evidence."** Only partly. Hitting explains
  **37.3% [21.4, 70.6]** of the gain, and acc-given-hit is *also* significantly higher for referent
  (+2.62 points on the 762 both-hit questions, p = 0.0495). The mechanism has two channels, not one.
* **DEAD HYPOTHESIS 2 — "breadth vs depth, and breadth wins."** There is **no trade-off**: referent
  dominates chunk on hit rate (57.73 vs 51.76, p = 4.73e-10) **and** on depth (2.002 vs 1.862). And
  depth is not worthless — having **>= 2** in-window frames rather than exactly one is worth
  **+4.57 points** (mode-balanced, CI [2.72, 6.50], p = 1e-4, MDE 2.69, n = 5,667 pairs / 894
  questions). That is a **step, not a per-frame slope**: the per-extra-frame estimates are +2.13
  (FE, p = 4.40e-07) and +1.53 with CI [0.00, 3.13] (mode-stratified WLS), the second of which does
  not exclude zero (section 5.2). A trade-off plane through dominating points is a figure that gets a
  paper rejected.
* **WHAT REPLACED THEM.** One axis — **concentration** — with **non-monotone hit rate** peaking at
  `referent` (established on hit rate; on accuracy the descent from the peak is significant on 1 of 3
  backbones and a bounded null on all three on the n = 2,962 mechanism subset, and on raw windows the
  left arm of the U does not survive — sections 2 and 5.3), and **scale matching** as the mechanism
  (section 5.4).

---

## 9. VideoChat-Flash-7B: the method holds; the rendered-mp4 delivery path does not

An earlier draft reported *"keyframe LOSES on VideoChat-Flash-7B, -3.04, a characterised failure
case"*. That number was measured against the model's **own native video pipeline**, which is **not
budget-matched** and records no frame count. Those arms have since finished. At **n = 3,233** — the
same key set the other three backbones run on — all four budget-matched contrasts on this backbone
are positive and three of them are significant:

| contrast | budget-matched? | n | delta | p | MDE |
|---|---|---|---|---|---|
| keyframe - `kf_uniform8` (uniform-8 through the **identical** mp4 pipeline) | **yes** | 3,233 | **+2.94** | 2.44e-05 | 1.93 |
| keyframe - `random` (8 frames from the same 64-frame pool) | **yes** | 3,233 | **+2.88** | 6.28e-05 | 2.00 |
| keyframe - chunk | **yes** | 3,233 | **+2.10** | 0.00108 | 1.78 |
| chunk - `kf_uniform8` | **yes** | 3,233 | +0.84 | 0.231 | **no difference larger than 1.88 points at 80% power** |
| keyframe vs `results_video_v2` (VCF's **own** native pipeline, frame count unknown) | **no** | 3,160 | **-2.97** | 2.6e-05 | — |

Accuracies on that 3,233-key set: keyframe **23.01**, chunk 20.88, `random` 20.14, `kf_uniform8`
20.07.

**So the headline contrast `keyframe - random` is positive and significant on 4 of 4 backbones**
(p = 9.03e-06 / 0.01015 / 3.21e-04 / 6.28e-05), and VideoChat-Flash is no longer a failure case or an
odd one out: it carries the largest budget-matched selection gain of the four (+2.94 against
uniform-8, against +2.75 / +2.13 / +2.51 elsewhere), though on the `keyframe - random` contrast it is
second to InternVL3-14B (+2.88 vs +2.91). Its `chunk - uniform` contrast is a
bounded null, as on Qwen. The sign of the old claim was an artefact of an unmatched baseline, and the
paper treats the budget-matched rows as the claim because budget matching is the declared central
discipline of the study.

*Provenance of the numbers in this table.* The deltas and p-values are the completed-arm numbers from
the run owner. The MDE column and the accuracy line were recomputed here directly from the four
prediction files (`results_baseline/kf_{referent,chunk,random,uniform8}/videochat-flash-7b/`). That
recomputation reproduces the two headline rows exactly (+2.94, p = 2.44e-05; +2.88, p = 6.28e-05) and
differs by a **single discordant pair** on the two chunk contrasts (+0.80, p = 0.249 for
`chunk - uniform`; +2.13, p = 0.000898 for `keyframe - chunk`). Neither reading changes a verdict, but
the difference should be traced before submission (section 12.1, item 1).

**The second finding on this backbone is bigger than the first: the rendered-mp4 delivery path costs
VideoChat-Flash-7B about 6 points.** Same arm, same questions, only the delivery path differs:

| backbone | `kf_uniform8` (rendered mp4, 8 f) | native pipeline | delta | p | n |
|---|---|---|---|---|---|
| VideoChat-Flash-7B | 20.25 | 26.20 | **-5.95** | 2.22e-16 | 3,160 |
| InternVL3-14B | 24.84 | 25.54 | -0.70 (quoted provisionally as -0.80) | 0.137 — **no difference larger than 1.25 points at 80% power** | 3,160 |
| Qwen2.5-VL-7B | 17.37 | 17.53 | -0.16 | 0.668 — **no difference larger than 0.83 points at 80% power** | 3,160 |
| Ovis2.5-9B | 23.10 | 22.97 | +0.13 | 0.747 — **no difference larger than 0.82 points at 80% power** | 3,160 |

(An earlier computation of the VideoChat-Flash row on the then-partial arms gave -5.75, p = 1.5e-11,
n = 2,277: same sign, same magnitude, smaller subset.)

On the other three backbones the delivery path is free to within 0.82-1.25 points at 80% power; on
VideoChat-Flash-7B it costs **-5.95 points**, **roughly twice** the largest selection effect measured
anywhere in this study (+2.94, on this same backbone) — not an order of magnitude, but still about
twice the gain it swamps. **This is a limitation of how we deliver frames to the model, not of which
frames we choose.** Every VideoChat-Flash number here is measured on a harness that handicaps that
backbone; the budget-matched contrasts remain valid because both arms pay the same penalty, but the
absolute accuracies are not comparable to that model's published numbers. The cause (codec /
container / resolution of the mp4v re-render interacting with that model's tokenizer) is **not
diagnosed**.

**What is still open on this backbone.** The four 8-frame arms are complete and hold the same 3,233
keys as the other backbones, so the contrasts above are final and mutually comparable — the earlier
`[PROVISIONAL]` labels on them are withdrawn. Three items remain, and they are bookkeeping rather
than measurement: (i) the collector (`RESULTS_MASTER.md`, `results_master.csv`) is still **stale for
this backbone** (generated at n = 2,341) and still shows the superseded partial-arm figure
+2.96 / n = 2,772; (ii) the native-pipeline comparison is keyed by *real* video id and runs on
n = 3,160, a different and smaller intersection than the budget-matched rows, so those two blocks are
not on one question set; (iii) the mediation percentage quoted for this backbone in section 4.2 was
computed on the partial arms and has not been recomputed, so it keeps its `[PROVISIONAL]` label.
Regenerate and reconcile before any VCF number is quoted in the paper (section 12.1, item 1).

---

## 10. Limitations

Flat, unsoftened, all of them in the paper.

1. **The absolute numbers are low and the gains are small.** Budget-matched baselines sit at
   **17.14-24.62%** against a **12.5% chance floor**; the method moves them by **+2 to +3 points**
   (n = 3,233 per contrast). Real, paired-significant improvements on a hard benchmark — and
   improvements *near chance*, not a solved task.
2. **The gain decays with budget, and spending more frames beats choosing better.** 8 to 16 frames:
   **no difference larger than 1.53 points at 80% power** (+0.46, p = 0.429, n = 3,233); 32 frames
   beats the 8-frame selector by **+2.32** (p = 8.79e-05, n = 3,059). The prior round's gain ladder
   (+2.14 / +0.51 n.s. / +1.08 p = 0.096 at 8 / 16 / 32 f) has **no matched comparator in the repo**
   and no n or MDE; it is a marker, not a result.
3. **The memory bank does not beat a budget-matched uniform baseline.** `mb_top2 - mb_uniform32 =
   +1.32`: **no difference larger than 3.58 points at 80% power** (p = 0.353, n = 759). Its +3.82 vs
   random is partly the control being 2.50 points below uniform-32. At k = 1 the effect vanishes
   (+1.58, p = 0.335, **no difference larger than 4.21 points**). And it has no demonstrated
   efficiency win either (break-even Q > 32 vs corpus density 8.16). It is a measured design point
   with a stated null.
4. **All mechanism analysis is conditioned on ORACLE, answer-informed evidence windows and is a
   DIAGNOSTIC, never a method.** Hit rate, depth, dose, mediation, chunk-hit and `mb_oracle` explain
   an already-measured gain; none can select frames at test time and none is an achievable accuracy.
   `mb_oracle` (35.05) is a ceiling. In `RESULTS_MASTER.md` it still renders in a column format
   identical to the method rows and must be moved to a separate ceiling block.
5. **Every mechanism quantity depends on the window-widening choice, and one contrast changes
   sign.** The raw-window robustness column (section 2) must be read alongside the primary column.
   Four quantities change status between them: referent - uniform hit rate (+15.23, p = 1.29e-35
   primary vs +1.62, **no difference larger than 2.35 points at 80% power**, raw); **chunk - uniform
   hit rate, which flips sign — +9.25 (p = 3.68e-13) primary against -1.62 (p = 0.0798, no difference
   larger than 2.54 points at 80% power) raw, i.e. on raw windows uniform-8 out-hits chunk**; the
   both-hit acc test (p = 0.0495 vs p = 0.0854); and the mediated fraction (37.3% vs 9.5%). The mode
   ordering on depth is stable across both columns (the saturation verdict is NOT: it is
   underpowered on raw windows), and so are the signs
   of every `referent` contrast — but **not** the sign of `chunk - uniform`. The magnitudes are not
   stable at all.
6. **The concentration inverted-U is established on hit rate, not on accuracy — and only on the
   tolerance-corrected windows.** referent - chunk is +5.98 points of hit rate (p = 4.73e-10,
   n = 2,962) but only +0.43 / +1.33 / +0.80 points of accuracy, significant on **one backbone of
   three**, and on the n = 2,962 mechanism subset the gap is a **bounded null on all three**
   (McNemar p = 0.658 / 0.056 / 0.420; section 5.5). On raw windows the left arm of the U disappears
   as well: chunk - uniform is -1.62 (p = 0.0798), so uniform-8 out-hits chunk (limitation 5,
   section 2).
7. **Scale matching is an account, not a tested claim.** The chunk-count sweep that would test it
   (2 / 4 / 8 / 16 / 32 chunks at fixed budget) has not been run.
8. **Unexplained attrition.** 3,667 declared MCQ vs 3,233 evaluated in every arm — **434 questions
   (11.8%) unaccounted for**. Also, the selection dump covers **405 of 449 videos (90.2%)** and is
   appended in sorted video order, so it is a **prefix, not a random sample**; the slice is close to
   the corpus on the one check available (InternVL3-14B uniform-8: 24.38% on the subset vs 24.62%
   corpus-wide, -0.25 points), which is reassurance, not proof.
9. **Evidence-window hygiene is not finished.** `dump_selections.py` silently skips videos with no
   mp4 (14 of 455 anon ids) and ingests `evidence_windows.json` with **no confidence filter**
   (3,216 `high` / 3 `medium` / 26 `low` / **54 `failed`**). The dropped set must be logged and the
   filter stated or applied. Note also that `nearest_s` is the distance from the window **centre**,
   not from the window, so a hit can carry `nearest_s > 0`; any caption saying "distance to the
   evidence window" must be relabelled. The docstring of `widen_windows.py` still quotes a stale
   corpus step of 125 frames; the computed value is 145.7.
10. **Two key spaces.** Frame-rendered arms are keyed by *anonymised* video id, memory-bank and
    native-video results by *real* video id; a naive join gives **zero** intersection. All cross-path
    joins go through `video_id_mapping.json`, always **anon-to-real** (`real_to_anon` is lossy,
    445 vs 455 entries). Collapsing anon-to-real drops 502 rows because 9 real video ids carry 2-3
    anon ids; 64 (real video, question) pairs collide, **40 of them with a different gold letter**.
    Duplicates are resolved deterministically by first occurrence.
11. **VideoChat-Flash-7B's 8-frame arms are now final at n = 3,233** and are on the same key set as
    the other three backbones, so its selection contrasts are no longer provisional (section 9).
    Three things about that backbone still are: the collector table is stale at n = 2,341, the
    native-pipeline comparison runs on a different key space (real-keyed, n = 3,160), and its
    mediation percentage in section 4.2 was computed on the partial arms. There is also an
    unreconciled one-discordant-pair difference between the run owner's numbers and an independent
    recomputation on its two chunk contrasts (section 9).

---

## 11. Is "two paths" a real motivation, or a rationalisation of having run two experiments?

The honest sentence: **we ran the two experiments first and found the axis afterwards, and at the
only budget where the two paths can be compared directly the axis is a null (-0.44, p = 0.832, n =
686, no difference larger than 3.85 points).**

What makes the framing more than a rationalisation is verifiable and stated above: the two paths
share the retrieval signal, the query text (byte-identical, verified in code), the candidate pool,
the budget in the frame-rendered arms, and both controls — so the 8-frame head-to-head really does
vary one thing. What keeps it from being a settled result is equally clear: the memory bank exists on
one backbone, at a different budget; the only budget-matched chunk arm is the weaker frame-rendered
twin; and the accuracy ordering is significant on two backbones of four at one budget and a bounded
null on the other two. The finding
that earns the framing is not the ordering — it is the **geometry**: referent dominating chunk on
both axes while touching twice as many chunks, which is a fact about what a score-driven retriever
does that a grid-driven one cannot.

---

## 12. Reproduction pointers

| What | Where |
|---|---|
| Both paths, frame-rendered (`referent`, `question`, `chunk`, `random`, `uniform`) | `gen_keyframe_clips.py` |
| True memory bank (InternVL, embedding splice) | `analysis3/membank/run_membank.py`; index in `analysis3/membank/build_clip_index.py` |
| Per-question selected frame indices x evidence windows | `analysis3/selanal/dump_selections.py` -> `analysis3/selanal/selections.jsonl` |
| **Window widening (PRIMARY data)** | `analysis3/selanal/widen_windows.py` -> `analysis3/selanal/selections_tol.jsonl` |
| Mechanism stats, PRIMARY | `analysis3/selanal/path1_stats_tol.json`, `path2_stats_tol.json` (+ `path2_stats_tol_extra.json`) |
| Mechanism stats, RAW-WINDOW ROBUSTNESS | `analysis3/selanal/path1_stats.json`, `path2_stats.json` (+ `path2_stats_extra.json`, which carries the `chunk - uniform` sign flip of section 2) |
| Full printed analyses | `/tmp/claude-1238/-home-ab260989-gen-reid/p1t.log`, `p2t.log` (primary); `p1.log`, `p2.log` (raw) |
| **Figures, PRIMARY (tolerance-corrected data)** | `analysis3/selanal/make_figures.py` -> `fig_path1_evidence_mediation.{png,pdf}`, `fig_path2_concentration_dose.{png,pdf}` |
| **Figures, RAW-WINDOW ROBUSTNESS pair** | `make_figures.py --windows raw` -> `fig_path1_evidence_mediation_rawwindows.{png,pdf}`, `fig_path2_concentration_dose_rawwindows.{png,pdf}` — the pair in which chunk sits below uniform-8 (section 2) |
| Oracle evidence windows (human-verified 92-94%) | `analysis3/evidence_windows.json` |
| Frame-level accuracy (ANON-keyed) | `results_baseline/{kf_referent,kf_question,kf_chunk,kf_random,kf_uniform8,kf_q_t16,kf_q_t32}/<backbone>/predictions.jsonl` |
| Memory-bank accuracy (REAL-keyed) | `analysis3/membank/results/*.jsonl` |
| Native video baseline, NOT budget-matched (REAL-keyed) | `results_video_v2/<backbone>/predictions.jsonl` |
| Anon-to-real id map (use `anon_to_real` only) | `video_id_mapping.json` |
| Master table + MDEs | `analysis3/selanal/collect_results.py` -> `RESULTS_MASTER.md`, `results_master.csv` |
| Adversarial reviews this document answers | `analysis3/selanal/review/REVIEW_FINAL.md` (findings A1-A2, B1-B3, C2, C4, D1-D2, E1-E3, F1 and the section-G overclaiming table are applied in this revision; residuals R1-R7 from the confirmation pass are applied on top), `review/REVIEW_NARRATIVE.md`, `review/REVIEW_STATS.md`, `review/FIG2_REFRAME.md` |

### 12.1 Open work register

| # | Item | Blocks |
|---|---|---|
| 1 | Regenerate `RESULTS_MASTER.md` / `results_master.csv` on the completed VideoChat-Flash arms (still stale at n = 2,341, still showing the superseded +2.96 / n = 2,772), and trace the one-discordant-pair difference on that backbone's two chunk contrasts (+0.84 / p = 0.231 vs +0.80 / p = 0.249; +2.10 / p = 0.00108 vs +2.13 / p = 0.000898) | every VCF quotation, section 9, limitation 11 |
| 2 | Explain the 434 missing questions (3,667 declared vs 3,233 evaluated) | section 1, every n |
| 3 | Finish the selection dump to 449 videos; log the 14 dropped mp4s; state or apply the window-confidence filter | sections 4-5, limitation 9 |
| 4 | Run matched `uniform16` / `uniform32` controls, or drop the prior-round gain ladder | section 3.2, limitation 2 |
| 5 | Chunk-count sweep (2/4/8/16/32 chunks at fixed budget) to test scale matching directly | section 5.4, limitation 7 |
| 6 | Measure amortisation in wall-clock and bytes, not frames encoded; count memory-bank storage | section 6.2 |
| 7 | Move `mb_oracle` out of the method-row format in `RESULTS_MASTER.md` | limitation 4 |
| 8 | Diagnose the VideoChat-Flash mp4-render penalty | section 9 |
