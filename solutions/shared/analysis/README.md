# Mechanism analysis

This directory answers why the two methods work and why one of them works better, and produces the
figures in the paper. Nothing here needs a GPU; the whole sequence runs on CPU in a few minutes
once the inputs exist.

Everything here conditions on the evidence windows described in `benchmark/data/README.md`. Those
windows are answer-informed. Every quantity produced here is therefore a **diagnostic** that
explains a measured accuracy difference, or a **ceiling**. None is a method and none is an
achievable accuracy. The scripts print this with their output and it must survive into any text
that quotes them.

## Inputs

- Prediction files for the method and control arms of both methods.
- `benchmark/data/evidence_windows.json`.
- The rendered clip sets, only for the selection dump below, which recomputes what each mode chose.

## Sequence

Run in this order. Each step writes a file the next one reads.

### 1. Recover what each selection mode actually chose

```bash
python -m solutions.shared.analysis.dump_selections
```

The selectors render their choice to a clip and record only the clip path, so the frame indices
are not written down anywhere. This step recomputes them. The selection is deterministic given the
CLIP features, and the script imports the selector's own function rather than reimplementing it, so
what is dumped is by construction what was rendered. It rebuilds the candidate pool with the
selector's own rule; the Cairn index uses a different grid and must not be substituted here.

Output: one row per question and mode, with the absolute frame indices chosen, the evidence
window, and the overlap between them.

### 2. Correct the windows to their own resolution

```bash
python -m solutions.shared.analysis.widen_windows
```

Half the windows are one frame wide, for the reason given in `benchmark/data/README.md`. Scoring
an eight-frame selection against a one-frame target measures nothing, and the raw numbers confirm
it: on those questions the uniform control scores highest, which is the signature of noise. This
step widens every window by the annotation grid half-step, derived per video from that video's own
grid. The tolerance is a property of the annotation and is never tuned against accuracy.

Both the raw and the widened files are kept, and every analysis below is reported under both.

### 3. Analyse each method

```bash
python -m solutions.shared.analysis.analyze_lantern --selections solutions/shared/analysis/selections_tol.jsonl
python -m solutions.shared.analysis.analyze_cairn   --selections solutions/shared/analysis/selections_tol.jsonl
```

`analyze_lantern.py` decomposes the accuracy gain into the part explained by landing on the
evidence and the part that is not, as an exact identity whose terms sum to the total. It tests
whether accuracy conditional on landing differs between the method and its control using a paired
test on the questions where both arms landed, which is the only stratum where the same item is seen
under both.

`analyze_cairn.py` measures the geometry of each selection mode, how concentrated it is and how
deeply it covers the evidence, and tests whether extra evidence frames pay. That test is stratified
on the ordered pair of modes being compared, because an unstratified version silently compares
modes rather than depths.

Both cluster on the question key wherever an observation appears more than once, and phrase every
non-significant result as a bound.

### 4. Assemble the results table

```bash
python -m solutions.shared.analysis.collect_results
```

Recomputes every accuracy and contrast from the prediction files rather than trusting any recorded
summary, and reports discrepancies against previously quoted values. Run this before quoting a
number anywhere.

### 5. Figures

```bash
python -m solutions.shared.analysis.make_figures                  # per-method, with a raw-window twin
python -m solutions.shared.analysis.make_figures --windows raw
python -m solutions.shared.analysis.make_paper_figure             # the combined figure in the paper
```

Figures are written as vector PDF with embedded Type 42 fonts at the final printed width, so the
point sizes in the file are the point sizes on the page. Colours are colourblind-safe and the
mapping is stated in each script's docstring and held identical across figures.

## Conventions enforced here

- Paired comparisons on the key intersection, exact McNemar, sample size reported every time.
- Every null stated as no difference larger than a stated bound at 80 percent power.
- Any quantity derived from the evidence windows labelled as a diagnostic or a ceiling, inline.
- The frame or token budget stated on every row.
