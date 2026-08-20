#!/usr/bin/env python
"""Recover WHICH frames each selection method chose, and whether they landed on the evidence.

Why this file has to exist
--------------------------
`solutions/lantern/select_frames.py` renders its choice straight to an mp4 and the manifest records only
key -> clip path. The frame indices are never written down, so after the fact there is no way to
ask the only question that explains the accuracy numbers: *did the method actually find the
moment the answer lives in?* Everything downstream -- hit rate, dose-response, the breadth-vs-depth
account of why chunking loses to keyframe -- needs the indices, so they are recomputed here.

The selection is deterministic given the CLIP features, and `select_pool_indices` is imported from
the renderer rather than reimplemented, so what is dumped is by construction what was rendered.
The candidate pool is rebuilt with the renderer's own rule (`linspace(0, N-1, n_candidates)`);
note this is NOT the pool in solutions/cairn/index, which is 8 frames inside each of 8 chunks and
therefore a different grid -- reusing that index here would silently score a selection that never
happened.

Runs on CPU in ~70 min for the corpus (measured: same CLIP pass as build_clip_index.py).

Out: one JSONL row per (key, mode) with the absolute frame indices, the evidence window, and the
overlap between them.
"""
import argparse, json, os, sys
import numpy as np

from persistqa.paths import ROOT  # noqa: E402
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/analysis3/mem")
sys.modules.setdefault("probe_video_judge_v2", type(sys)("probe_video_judge_v2"))
import importlib.util
_spec = importlib.util.spec_from_file_location("gkc", f"{ROOT}/solutions/lantern/select_frames.py")
gkc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(gkc)

MODES = ("referent", "chunk", "random", "uniform")


def main():
    import torch, decord, open_clip
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/solutions/shared/analysis/selections.jsonl")
    ap.add_argument("--n_candidates", type=int, default=gkc.N_CANDIDATES)
    ap.add_argument("--topk", type=int, default=gkc.TOPK)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from mem_common import load_evidence
    ev = load_evidence()                      # key -> {t0,t1,fps,...}, oracle windows
    a2r = json.load(open(f"{ROOT}/video_id_mapping.json")).get("anon_to_real", {})
    r2a = json.load(open(f"{ROOT}/video_id_mapping.json")).get("real_to_anon", {})
    bench = gkc.load_bench()                  # anon-keyed, same source the renderer used
    vids = sorted({k.split("|")[0] for k in bench})
    if args.limit: vids = vids[:args.limit]

    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try: done.add(json.loads(l)["key"] + "|" + json.loads(l)["mode"])
            except Exception: pass
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clip, _, prep = open_clip.create_model_and_transforms(gkc.CLIP_MODEL[0], pretrained=gkc.CLIP_MODEL[1])
    clip = clip.eval().to(dev); ctok = open_clip.get_tokenizer(gkc.CLIP_MODEL[0])
    print(f"[dump] {len(vids)} videos, modes={MODES}, dev={dev} -> {args.out}", flush=True)

    fout = open(args.out, "a"); n = 0
    for vi, anon in enumerate(vids):
        vp = None
        real = a2r.get(anon, anon)
        for cand in (f"/home/c3-0/datasets/moviechat1k-test/{real}.mp4",
                     f"/home/c3-0/datasets/moviechat1k-test/{anon}.mp4"):
            if os.path.isfile(cand): vp = cand; break
        if vp is None: continue
        try:
            vr = decord.VideoReader(vp); N = len(vr); fps = float(vr.get_avg_fps()) or 25.0
        except Exception as e:
            print(f"  [vid err] {anon}: {str(e)[:70]}", flush=True); continue
        pool = np.linspace(0, N - 1, min(args.n_candidates, N)).astype(int)   # renderer's own rule
        try:
            arr = vr.get_batch([int(i) for i in pool]).asnumpy()
        except Exception as e:
            print(f"  [dec err] {anon}: {str(e)[:70]}", flush=True); del vr; continue
        pil = [Image.fromarray(a).convert("RGB") for a in arr]
        with torch.no_grad():
            feats = torch.nn.functional.normalize(
                clip.encode_image(torch.stack([prep(p) for p in pil]).to(dev)).float(), dim=-1).cpu().numpy()
        for k in [kk for kk in bench if kk.split("|")[0] == anon]:
            # evidence windows are keyed on REAL ids; the renderer is keyed on ANON
            rk = f"{a2r.get(anon, anon)}|{k.split('|')[1]}"
            rec = ev.get(rk) or ev.get(k)
            f0 = f1 = None
            if rec and rec.get("t0") is not None:
                efps = float(rec.get("fps") or fps) or 25.0
                f0 = max(0, min(N - 1, int(round(float(rec["t0"]) * efps))))
                f1 = max(f0, min(N - 1, int(round(float(rec["t1"]) * efps))))
            sims_cache = {}
            for mode in MODES:
                if f"{k}|{mode}" in done: continue
                sims = None
                if mode in gkc.CLIP_MODES:
                    qt = gkc.query_text(bench[k], mode)
                    if qt not in sims_cache:
                        with torch.no_grad():
                            qf = torch.nn.functional.normalize(
                                clip.encode_text(ctok([qt[:300]]).to(dev)).float(), dim=-1).cpu().numpy()[0]
                        sims_cache[qt] = feats @ qf
                    sims = sims_cache[qt]
                sel = gkc.select_pool_indices(mode, k, len(pool), args.topk, sims)
                frames = [int(pool[i]) for i in sel]
                n_in = 0 if f0 is None else sum(1 for x in frames if f0 <= x <= f1)
                # distance from the window, in seconds, for the nearest selected frame
                near = None if f0 is None else min(abs(x - (f0 + f1) / 2.0) for x in frames) / max(fps, 1e-6)
                fout.write(json.dumps({
                    "key": k, "real_key": rk, "video_id": anon, "question_id": k.split("|")[1],
                    "mode": mode, "sel_pool": [int(x) for x in sel], "sel_frames": frames,
                    "n_total": int(N), "fps": round(fps, 3),
                    "ev_f0": f0, "ev_f1": f1,
                    "ev_span_frames": None if f0 is None else int(f1 - f0 + 1),
                    "n_in_window": int(n_in), "hit": None if f0 is None else bool(n_in > 0),
                    "nearest_s": None if near is None else round(float(near), 3),
                    # temporal dispersion of the selection, as a fraction of video length
                    "spread": round(float((max(frames) - min(frames)) / max(N - 1, 1)), 4),
                    "chunk_ids": sorted(set(int(x) * 8 // len(pool) for x in sel)),
                }) + "\n")
                n += 1
        fout.flush(); del vr, pil, arr
        if (vi + 1) % 25 == 0: print(f"  [{vi+1}/{len(vids)}] rows={n}", flush=True)
    fout.close(); print(f"[dump] wrote {n} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
