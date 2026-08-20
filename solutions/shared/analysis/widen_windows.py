#!/usr/bin/env python
"""Re-derive the evidence-hit fields at the ANNOTATION'S OWN RESOLUTION.

Why this is necessary, not a fudge
----------------------------------
The oracle windows in benchmark/data/evidence_windows.json were localised on a DENSE FRAME GRID: an
annotation names one or more `dense_frames`, which map to `video_frames`. The grid's median step is
125 video frames (~5 s at 25 fps), so the procedure cannot resolve the evidence to better than
+/- ~62 frames, ever.

Half the questions (1,491 of 2,962) name a SINGLE dense frame, which makes t0 == t1 and yields a
window ONE FRAME wide. Scoring "did a selected frame land inside the window?" against a 1-frame
target asks 8 samples to hit an exact frame out of several thousand, and the measured hit rates say
exactly that:

    point windows (n=1,491):  referent 1.81  chunk 1.88  random 0.40  uniform 2.68   <- noise;
                                                                        uniform "wins"
    span  windows (n=1,471):  referent 55.40 chunk 48.81 random 44.87 uniform 51.26  <- signal

So the pooled hit rate is a 50/50 mixture of a real measurement and a near-zero one. That is not a
property of the selection methods; it is an artefact of treating a grid-resolution point estimate as
an exact instant, and it dilutes every quantity mediated through `hit`.

The fix is to widen EVERY window by the grid half-step -- span windows too, since their endpoints
come from the same dense frames and carry the same uncertainty. Applying the tolerance uniformly is
the consistent choice; applying it only to the degenerate half would bake in a different precision
for each group.

The tolerance is derived PER VIDEO from that video's own dense->video frame mapping wherever two
distinct dense frames are available, and falls back to the corpus median step otherwise. It is a
property of the annotation, never tuned against accuracy.

Both the raw and the widened files should be analysed and BOTH reported: the widened one is primary
(it measures at the resolution the annotation actually has), the raw one is the robustness check.

Usage:
  python solutions/shared/analysis/widen_windows.py \
      --in solutions/shared/analysis/selections.jsonl --out solutions/shared/analysis/selections_tol.jsonl
"""
import argparse, json, os
import numpy as np

from persistqa.paths import ROOT  # noqa: E402


def grid_steps(ev):
    """video_id -> median dense-grid step in video frames, plus the corpus-wide fallback."""
    per = {}
    for k, d in ev.items():
        df, vf = d.get("dense_frames") or [], d.get("video_frames") or []
        if len(df) >= 2 and len(vf) >= 2 and df[-1] != df[0]:
            per.setdefault(k.split("|")[0], []).append(abs(vf[-1] - vf[0]) / abs(df[-1] - df[0]))
    med = {v: float(np.median(s)) for v, s in per.items()}
    glob = float(np.median(list(med.values()))) if med else 125.0
    return med, glob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=f"{ROOT}/solutions/shared/analysis/selections.jsonl")
    ap.add_argument("--out", default=f"{ROOT}/solutions/shared/analysis/selections_tol.jsonl")
    ap.add_argument("--evidence", default=f"{ROOT}/benchmark/data/evidence_windows.json")
    args = ap.parse_args()

    ev = json.load(open(args.evidence))
    per_vid, glob = grid_steps(ev)
    print(f"[tol] dense-grid step: corpus median {glob:.0f} frames -> half-step +/-{glob/2:.0f} "
          f"frames; per-video steps for {len(per_vid)} videos", flush=True)

    n = wid = 0
    flips = {"miss->hit": 0, "hit->hit": 0}
    with open(args.out, "w") as fo:
        for line in open(args.inp):
            r = json.loads(line)
            if r.get("ev_f0") is None:
                fo.write(json.dumps(r) + "\n"); n += 1; continue
            # the real key is the one the evidence file is keyed on
            step = per_vid.get(r["real_key"].split("|")[0], glob)
            tol = int(round(step / 2.0))
            f0 = max(0, r["ev_f0"] - tol)
            f1 = min(r["n_total"] - 1, r["ev_f1"] + tol)
            was = bool(r["hit"])
            n_in = sum(1 for x in r["sel_frames"] if f0 <= x <= f1)
            near = min(abs(x - (f0 + f1) / 2.0) for x in r["sel_frames"]) / max(r["fps"], 1e-6)
            r.update({
                "ev_f0_raw": r["ev_f0"], "ev_f1_raw": r["ev_f1"], "tol_frames": tol,
                "ev_f0": f0, "ev_f1": f1, "ev_span_frames": int(f1 - f0 + 1),
                "n_in_window": int(n_in), "hit": bool(n_in > 0), "nearest_s": round(float(near), 3),
            })
            flips["miss->hit" if (not was and r["hit"]) else "hit->hit" if was else "x"] = \
                flips.get("miss->hit" if (not was and r["hit"]) else "hit->hit" if was else "x", 0) + 1
            fo.write(json.dumps(r) + "\n"); n += 1; wid += 1
    print(f"[tol] wrote {n} rows ({wid} with a window widened) -> {args.out}", flush=True)
    print(f"[tol] transitions: {flips}", flush=True)


if __name__ == "__main__":
    main()
