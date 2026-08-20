# Benchmark data

Four files define the benchmark. The videos themselves are not redistributed; this directory
contains the questions, the metadata and the annotations that reference them.

## Files

### persistqa.json

The question set, keyed by anonymous video identifier. This is the canonical file used by the
evaluation harness and by both solutions.

Structure:

```
{
  "videos": [
    {
      "video_id": "vid_0001",
      "questions": [
        {
          "question_id": "q14",
          "question_text": "...",
          "options": {"A": "...", "B": "...", ...},   // eight options
          "correct_answer": "C",
          "metadata": {
            "capability": "location",                  // what is asked about the referent
            "reid_canonical": "cross_scene_reid",      // the identity-tracking demand
            "temporal_anchor": "...",                  // the referring phrase, when present
            "difficulty": "hard"
          }
        }
      ]
    }
  ]
}
```

The `capability` axis has five values and the `reid_canonical` axis has ten. Both are defined in
the paper appendix. `temporal_anchor` is the phrase that names the referent; it is what the
referent-conditioned query mode in `solutions/lantern` appends to the question.

### persistqa_real_ids.json

The same questions keyed by the source-corpus video identifier instead of the anonymous one.

**Two identifier spaces exist and they do not interchange.** The evaluation harness and Lantern are
keyed on anonymous identifiers; Cairn and the evidence windows are keyed on real identifiers. A
join across the two without mapping produces an empty intersection rather than an error, which is
silent and has caused real mistakes. Always map before joining, and always map in the direction
anonymous to real: the reverse mapping is lossy, because several anonymous identifiers can share
one real video.

### video_id_mapping.json

```
{"anon_to_real": {"vid_0001": "--UgPWRVt8A", ...},
 "real_to_anon": {"--UgPWRVt8A": "vid_0001", ...}}
```

`anon_to_real` is complete. `real_to_anon` has fewer entries than `anon_to_real` for the reason
above, so treat it as a convenience only.

### evidence_windows.json

For each question, when the answer is visible in the video.

```
{"<real_video_id>|<question_id>": {
    "key": "...",
    "dense_frames": [20, 21],       // indices on the dense annotation grid
    "video_frames": [3017, 3176],   // the same moments as absolute video frame indices
    "t0": 125.83, "t1": 132.47,     // seconds
    "video_nframes": 4924}}
```

Two properties matter when using these.

**They are answer-informed.** The annotation was produced by showing a model the question and its
correct answer and asking only when the answer becomes visible. Anything conditioned on these
windows is a diagnostic that explains a measured result, or a ceiling. It is never a method and
never an achievable accuracy.

**They carry the resolution of their grid, not of the video.** Windows were localized on a dense
frame grid whose median step is 146 video frames. Half the questions name a single grid point, so
`t0` equals `t1` and the raw window is one frame wide. Treating such a window as an exact instant
measures nothing: asking whether eight sampled frames land on one exact frame out of several
thousand returns noise. `solutions/shared/analysis/widen_windows.py` widens every window by the
grid half-step, which is the resolution the annotation actually has. Analyses report results under
both the raw and the widened windows.

## Obtaining the videos

The 449 videos come from three public corpora: CinePile, MovieChat-1k and LVU. Obtain each from
its source and place the files in one directory named by identifier, as `<video_id>.mp4`, then
point the harness at it:

```bash
export PERSISTQA_VIDEO_DIR=/path/to/videos
```

Either identifier space works for filenames; `persistqa.paths.video_path` tries the real
identifier first and falls back to the anonymous one.

Thirty videos cannot be decoded by the evaluation stack. They are excluded for every model so that
all models are scored on the same question set.
