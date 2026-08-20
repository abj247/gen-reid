#!/usr/bin/env python3
"""
v3: question-anchored, identity-isolated event retrieval (the last training-free shot).

The v2 diagnostic showed the per-identity WHOLE-VIDEO summary almost never contains the
queried event (content-ceiling 13%, self-judged "has info" 6%). v3 makes the description
QUESTION-CONDITIONED and IDENTITY-ISOLATED:

  per question:
    1. GROUND  match the question's referent to one identity (CLIP text<->identity embed,
               from cached results/tracks_real/<anon>.json).
    2. LOCALIZE+MARK  take dense frames where ONLY that identity appears, box it (set-of-marks).
    3. DESCRIBE  give the describe-VLM the QUESTION (minus options; never the answer) as focus
               context and ask what the marked entity DOES around that moment.
    4. ANSWER   a frozen reader answers the MCQ from that description (text-only).

Isolating one identity removes the cross-identity binding ambiguity that causes the 95%
WHAT-errors, so this can beat the direct video+text ceiling (~24-29%). Reuses the cached
perception, so it is fast (no YOLO/CLIP re-run).

Usage:
    python build_v3.py --describe_model internvl3-8b \
        --test_models qwen2.5-vl-7b internvl3-8b --out_dir results/v3
"""
import argparse, gc, json, os
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="combined_all_hard_v3_retagged.json")
    ap.add_argument("--probe_ids", default="results/oracle_probe/probe_ids.json")
    ap.add_argument("--oracle_pred", default="results/oracle_probe/arm_predictions.json")
    ap.add_argument("--tracks_dir", default="results/tracks_real")
    ap.add_argument("--mapping", default="video_id_mapping.json")
    ap.add_argument("--video_dir", default="/home/c3-0/datasets/moviechat1k-test")
    ap.add_argument("--describe_model", default="internvl3-8b")
    ap.add_argument("--test_models", nargs="+", default=["qwen2.5-vl-7b", "internvl3-8b"])
    ap.add_argument("--n_marked", type=int, default=12)
    ap.add_argument("--out_dir", default="results/v3")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    doss_dir = out / "dossiers"; doss_dir.mkdir(exist_ok=True)

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from oracle_binding_probe import load_bench
    from build_dossier import render_marked_frame, select_dets
    from evaluate_vlm_bm import create_model

    bench = load_bench(args.bench)
    probe = json.load(open(args.probe_ids))
    a2r = json.load(open(args.mapping))["anon_to_real"]

    # covered = probe questions whose video has a cached tracks file with >=1 identity
    tracks = {}
    for k in probe:
        anon = k.split("|")[0]
        tp = Path(args.tracks_dir) / f"{anon}.json"
        if anon not in tracks and tp.exists():
            tj = json.load(open(tp))
            if tj.get("identities"):
                tracks[anon] = tj
    covered = [k for k in probe if k.split("|")[0] in tracks and tuple(k.split("|")) in bench]
    print(f"probe={len(probe)} | covered (have tracks)={len(covered)}", flush=True)
    if not covered:
        return

    def vpath(anon):
        real = a2r.get(anon, anon)
        for ext in (".mp4", ".avi", ".mkv", ".mov", ".webm"):
            c = Path(args.video_dir) / f"{real}{ext}"
            if c.is_file():
                return str(c)
        return None

    # ---- STAGE 1: question-conditioned describe (grounded, isolated, marked) ----
    import open_clip, decord
    clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    clip = clip.eval().cuda(); tok = open_clip.get_tokenizer("ViT-B-32-quickgelu")

    def ground(anon, question):
        ids = tracks[anon]["identities"]
        embs, keys = [], []
        for kk, s in ids.items():
            if s.get("embed"):
                embs.append(np.array(s["embed"], np.float32)); keys.append(kk)
        if not keys:
            return None
        with torch.no_grad():
            q = clip.encode_text(tok([question[:300]]).cuda()).float().cpu().numpy()[0]
        q /= (np.linalg.norm(q) + 1e-8)
        sims = [float(np.dot(q, e / (np.linalg.norm(e) + 1e-8))) for e in embs]
        return keys[int(np.argmax(sims))]

    PROMPT = ("The bright box marks ONE specific {label} across these {n} frames, shown in time "
              "order from a video. A question will be asked about this {label}:\n\"{q}\"\n"
              "Describe in detail what the marked {label} DOES and what it interacts with, "
              "especially around the moment the question refers to. Be specific and factual "
              "(2-4 sentences). Do NOT guess or state an answer; only describe what is visible.")

    print(f"loading describe model {args.describe_model} ...", flush=True)
    dm = create_model(args.describe_model, device="cuda"); dm.load_model()
    qdoss = {}
    for i, k in enumerate(sorted(covered)):     # sorted -> same video's questions are adjacent
        anon, qid = k.split("|"); b = bench[(anon, qid)]
        cache_f = doss_dir / f"{anon}__{qid}.json"
        if cache_f.exists():
            qdoss[k] = json.load(open(cache_f))["desc"]; continue
        gid = ground(anon, b["question"])
        if gid is None:
            qdoss[k] = ""; json.dump({"desc": ""}, open(cache_f, "w")); continue
        slot = tracks[anon]["identities"][gid]
        dets = select_dets(slot["dets"], args.n_marked)
        vp = vpath(anon)
        if vp is None:
            qdoss[k] = ""; json.dump({"desc": ""}, open(cache_f, "w")); continue
        vr = decord.VideoReader(vp)             # open/close per question -> bounded CPU RAM
        gframes = sorted({d["global_frame"] for d in dets})
        raw = {int(g): vr.get_batch([g]).asnumpy()[0] for g in gframes}
        del vr
        imgs = [render_marked_frame(raw[d["global_frame"]], d["box"]) for d in dets]
        prompt = PROMPT.format(label=slot["label"], n=len(imgs), q=b["question"])
        try:
            desc = dm.describe_images(imgs, prompt, max_new_tokens=200)
        except Exception as e:
            desc = f"(describe failed: {type(e).__name__})"
        qdoss[k] = desc
        del raw, imgs
        json.dump({"desc": desc, "gid": gid, "label": slot["label"]}, open(cache_f, "w"))
        if (i + 1) % 20 == 0:
            print(f"  describe [{i+1}/{len(covered)}]", flush=True)
    del dm, clip; gc.collect(); torch.cuda.empty_cache()

    # ---- STAGE 2: readers answer from the question-conditioned dossier ----
    results = {}
    for mkey in args.test_models:
        print(f"\n=== answering with {mkey} (v3) ===", flush=True)
        model = create_model(mkey, device="cuda"); model.load_model()
        rows = []
        for k in covered:
            anon, qid = k.split("|"); b = bench[(anon, qid)]
            gold = b["gold"] if b["gold"] in b["options"] else sorted(b["options"])[0]
            qtext = f"Scene description: {qdoss[k]}\n\n{b['question']}" if qdoss[k] else b["question"]
            try:
                pred = model.inference_text_only(qtext, b["options"])
            except Exception:
                pred = "ERROR"
            rows.append({"k": k, "pred": pred, "gold": gold, "correct": pred == gold,
                         "reid": b["reid"]})
        results[model.model_name] = rows
        del model; gc.collect(); torch.cuda.empty_cache()
    json.dump(results, open(out / "predictions.json", "w"), indent=2)

    # ---- report ----
    oracle = json.load(open(args.oracle_pred)); cov = set(covered)
    sub = lambda rs: 100.0 * sum(x["correct"] for x in rs if x["k"] in cov) / \
        max(len([x for x in rs if x["k"] in cov]), 1)
    lines = ["V3 QUESTION-ANCHORED EVENT RETRIEVAL vs ORACLE CEILING", "=" * 55,
             f"covered: {len(covered)}/{len(probe)}", ""]
    for mname, rows in results.items():
        d = 100.0 * sum(x["correct"] for x in rows) / max(len(rows), 1)
        ok = next((m for m in oracle if m == mname), None)
        a = sub(oracle[ok]["A_baseline"]) if ok else float("nan")
        bb = sub(oracle[ok]["B_oracle"]) if ok else float("nan")
        rec = 100.0 * (d - a) / (bb - a) if ok and bb > a else float("nan")
        lines += [f"{mname}",
                  f"  A_baseline   : {a:6.2f}%",
                  f"  D_v3         : {d:6.2f}%   <-- pipeline",
                  f"  B_oracle     : {bb:6.2f}%",
                  f"  -> lift D-A = {d-a:+.1f}pp | ceiling recovery = {rec:.0f}%", ""]
    lines.append("per-category D_v3:")
    for mname, rows in results.items():
        by = defaultdict(lambda: [0, 0])
        for x in rows:
            by[x["reid"]][0] += x["correct"]; by[x["reid"]][1] += 1
        lines.append(f"  {mname}: " + " | ".join(
            f"{c}: {100*v[0]/v[1]:.0f}% ({v[0]}/{v[1]})" for c, v in sorted(by.items())))
    rep = "\n".join(lines); print("\n" + rep, flush=True)
    (out / "report.txt").write_text(rep + "\n")


if __name__ == "__main__":
    main()
