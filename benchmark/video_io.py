#!/usr/bin/env python
"""
GATE PROBE B (tightened) -- disambiguate the RED from the 8B/uniform-frame judge.

Probe B was RED (frozen InternVL2-8B judge, 6 frames spread over the whole track:
video 2-way = 0.167, video<text). Two confounds: (1) judge strength, (2) the frames
were not localized to the queried moment. This version controls both:

  JUDGE = InternVL3-14B (newer + bigger; cached; loaded 4-bit, model_id overridden).
  Stage 1 (CLIP): for the TRUE (v3-grounded) person, CLIP-rank that identity's frames by
    similarity to the question+temporal_anchor and keep the top-N as MOMENT-LOCALIZED frames.
  Stage 2 (VLM judge), per question, forced gold-vs-magnet:
    V2_loc  : video 2-way, MOMENT-LOCALIZED frames     (the real test)
    V2_uni  : video 2-way, uniform-over-track frames    (the old setting, for the delta)
    T2      : text 2-way, black image (prior control)
    V8_loc  : video 8-way, localized frames
  chance: 2-way 0.50, 8-way ~0.125.

Decision:
  V2_loc >> 0.5 (>=~0.70) AND >> T2 -> verifier exists once moment+judge are right -> PURSUE.
  V2_loc still ~ chance / < T2       -> RED is robust -> pivot before writer-RL; diagnosis hardens.
  V2_loc > V2_uni meaningfully       -> localization mattered (moment-coverage confound was real).

Frozen, offline. InternVL3-14B 4-bit on one 16GB GPU. ~30-60 min.
Run: CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python -m benchmark.video_io
"""
import json, os, re, random
import numpy as np
from PIL import Image, ImageDraw

import os as _os
from pathlib import Path as _Path
ROOT = _os.environ.get("PERSISTQA_ROOT") or str(_Path(__file__).resolve().parents[1])
BENCH = f"{ROOT}/combined_all_hard_v3_retagged.json"
PROBE_IDS = f"{ROOT}/results/oracle_probe/probe_ids.json"
BIND_ERR = f"{ROOT}/results/binding_errors.json"
TRACKS_DIR = f"{ROOT}/results/tracks_real"
V3_DIR = f"{ROOT}/results/v3/dossiers"
MAPPING = f"{ROOT}/video_id_mapping.json"
VIDEO_DIR = "/home/c3-0/datasets/moviechat1k-test"
OUT_DIR = f"{ROOT}/results/probe_video_judge"
os.makedirs(OUT_DIR, exist_ok=True)
JUDGE_ID = os.environ.get("JUDGE_MODEL_ID", "OpenGVLab/InternVL3-14B")
N_LOC = 8           # localized frames
N_UNI = 8           # uniform frames (same budget as localized -> isolates localization)
CAND_CAP = 24       # candidate frames CLIP-ranks over
EXTS = (".mp4", ".avi", ".mkv", ".mov", ".webm")


def load_bench(path):
    raw = json.load(open(path)); out = {}
    for v in raw["videos"]:
        vid = v["video_id"]
        for q in v.get("questions", []):
            qid = q.get("question_id") or q.get("qid")
            if not qid:
                continue
            gold = (q.get("correct_answer") or q.get("answer") or "").strip().upper()[:1]
            md = q.get("metadata", {})
            out[f"{vid}|{qid}"] = {"options": q.get("options", {}), "gold": gold,
                                   "qtext": q.get("question_text", ""),
                                   "anchor": md.get("temporal_anchor", "") or "",
                                   "reid": md.get("reid_canonical", "?")}
    return out


def load_magnets(path):
    out = {}
    for r in json.load(open(path)):
        out[f"{r['video_id']}|{r['question_id']}"] = (r.get("magnet") or "").strip().upper()[:1]
    return out


def resolve_video(vid, a2r):
    for stem in (a2r.get(vid, vid), vid):
        for e in EXTS:
            p = os.path.join(VIDEO_DIR, f"{stem}{e}")
            if os.path.exists(p):
                return p
    return None


def spread(dets, k):
    ds = sorted(dets, key=lambda d: d["global_frame"])
    if len(ds) <= k:
        return ds
    return [ds[i] for i in np.linspace(0, len(ds) - 1, k).astype(int)]


def render(frame_np, box):
    img = Image.fromarray(frame_np).convert("RGB")
    x1, y1, x2, y2 = [int(v) for v in box]
    dr = ImageDraw.Draw(img)
    for w in range(5):
        dr.rectangle([x1 - w, y1 - w, x2 + w, y2 + w], outline=(50, 255, 50))
    return img


def parse_letter(resp, valid):
    m = re.search(r"[A-H]", (resp or "").strip().upper())
    return m.group(0) if (m and m.group(0) in valid) else "INVALID"


def main():
    import torch, decord
    rng = random.Random(0)
    bench = load_bench(BENCH)
    mag = load_magnets(BIND_ERR)
    a2r = json.load(open(MAPPING)).get("anon_to_real", {})
    probe_ids = json.load(open(PROBE_IDS))
    limit = int(os.environ.get("PROBE_LIMIT", "0"))
    if limit > 0:
        probe_ids = probe_ids[:limit]; print(f"[smoke] limited to {limit}", flush=True)

    # ---- gather valid items ----
    items = []
    skipped = {"no_bench": 0, "no_magnet": 0, "no_v3": 0, "no_track": 0, "no_video": 0}
    for key in probe_ids:
        vid, qid = key.split("|")
        b = bench.get(key)
        if not b or b["gold"] not in b["options"]:
            skipped["no_bench"] += 1; continue
        magnet = mag.get(key, "")
        if magnet not in b["options"] or magnet == b["gold"]:
            skipped["no_magnet"] += 1; continue
        v3p, trp = f"{V3_DIR}/{vid}__{qid}.json", f"{TRACKS_DIR}/{vid}.json"
        if not (os.path.exists(v3p) and os.path.exists(trp)):
            skipped["no_v3"] += 1; continue
        gid = json.load(open(v3p)).get("gid")
        tracks = json.load(open(trp)).get("identities", {})
        if gid not in tracks:
            skipped["no_track"] += 1; continue
        vpath = resolve_video(vid, a2r)
        if not vpath:
            skipped["no_video"] += 1; continue
        items.append({"key": key, "b": b, "magnet": magnet, "vpath": vpath,
                      "dets": tracks[gid]["dets"]})
    print(f"[plan] valid items: {len(items)}  skipped={skipped}", flush=True)

    # ---- Stage 1: CLIP moment-localization ----
    print("[stage1] CLIP-localizing frames to the queried moment...", flush=True)
    import open_clip
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clip, _, prep = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    clip = clip.eval().to(dev)
    tok = open_clip.get_tokenizer("ViT-B-32-quickgelu")

    for it in items:
        b = it["b"]
        cand = spread(it["dets"], CAND_CAP)
        try:
            vr = decord.VideoReader(it["vpath"]); N = len(vr)
            crops, valid_dets = [], []
            for d in cand:
                gf = min(int(d["global_frame"]), N - 1)
                fr = vr[gf].asnumpy()
                x1, y1, x2, y2 = [int(v) for v in d["box"]]
                x1, y1 = max(0, x1), max(0, y1)
                crop = fr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crops.append(prep(Image.fromarray(crop).convert("RGB"))); valid_dets.append(d)
            del vr
            if not crops:
                it["loc"] = spread(it["dets"], N_LOC); it["uni"] = spread(it["dets"], N_UNI); continue
            with torch.no_grad():
                ie = clip.encode_image(torch.stack(crops).to(dev)).float()
                ie = torch.nn.functional.normalize(ie, dim=-1)
                txt = (b["qtext"] + ". " + b["anchor"])[:300]
                te = clip.encode_text(tok([txt]).to(dev)).float()
                te = torch.nn.functional.normalize(te, dim=-1)
                sims = (ie @ te.T).squeeze(1).cpu().numpy()
            order = np.argsort(-sims)[:N_LOC]
            it["loc"] = sorted([valid_dets[i] for i in order], key=lambda d: d["global_frame"])
            it["uni"] = spread(it["dets"], N_UNI)
        except Exception as e:
            print(f"  [stage1 err] {it['key']}: {e}", flush=True)
            it["loc"] = spread(it["dets"], N_LOC); it["uni"] = spread(it["dets"], N_UNI)
    del clip
    torch.cuda.empty_cache()

    # ---- Stage 2: VLM judge ----
    print(f"[stage2] loading judge {JUDGE_ID} (4-bit)...", flush=True)
    from evaluate_vlm_bm import create_model
    quant = os.environ.get("PROBE_QUANT", "1") == "1"   # 0 -> bf16 (use on >=48GB GPU)
    model = create_model("internvl3-14b", device="cuda", num_frames=N_LOC,
                         max_pixels=(448, 448), quantize_4bit=quant)
    model.model_id = JUDGE_ID            # override InternVL2-14B -> cached InternVL3-14B
    model.model_name = f"judge::{JUDGE_ID}"
    model.load_model()
    black = [Image.new("RGB", (448, 448), (0, 0, 0))]

    def pils(vpath, dets):
        vr = decord.VideoReader(vpath); N = len(vr); out = []
        for d in dets:
            gf = min(int(d["global_frame"]), N - 1)
            out.append(render(vr[gf].asnumpy(), d["box"]))
        del vr
        return out

    rows = []
    for i, it in enumerate(items):
        b, key, magnet = it["b"], it["key"], it["magnet"]
        gold = b["gold"]; gtext, mtext = b["options"][gold], b["options"][magnet]
        gold_is_A = rng.random() < 0.5
        a_t, b_t = (gtext, mtext) if gold_is_A else (mtext, gtext)
        gl2 = "A" if gold_is_A else "B"
        two = (f"These images show the SAME person, marked with a green box, at several moments. "
               f"{b['qtext']}\nWhich option correctly describes what the marked person does?\n"
               f"(A) {a_t}\n(B) {b_t}\nAnswer with only the letter A or B.")
        opt_block = "\n".join(f"({L}) {t}" for L, t in b["options"].items())
        eight = (f"These images show the SAME person, marked with a green box, at several moments. "
                 f"{b['qtext']}\nWhich option correctly describes what the marked person does?\n"
                 f"{opt_block}\nAnswer with only the letter.")
        try:
            ploc, puni = pils(it["vpath"], it["loc"]), pils(it["vpath"], it["uni"])
            v2_loc = parse_letter(model.describe_images(ploc, two, max_new_tokens=6), {"A", "B"})
            v2_uni = parse_letter(model.describe_images(puni, two, max_new_tokens=6), {"A", "B"})
            t2 = parse_letter(model.describe_images(black, two, max_new_tokens=6), {"A", "B"})
            v8_loc = parse_letter(model.describe_images(ploc, eight, max_new_tokens=6), set(b["options"]))
        except Exception as e:
            print(f"  [stage2 err] {key}: {e}", flush=True)
            torch.cuda.empty_cache(); continue
        torch.cuda.empty_cache()
        rows.append({"key": key, "reid": b["reid"], "n_loc": len(ploc), "n_uni": len(puni),
                     "v2_loc": int(v2_loc == gl2), "v2_uni": int(v2_uni == gl2),
                     "t2": int(t2 == gl2), "v8_loc": int(v8_loc == gold)})
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(items)}] scored={len(rows)}", flush=True)

    n = len(rows)
    rate = lambda f: float(np.mean([r[f] for r in rows])) if n else None
    summary = {"judge": JUDGE_ID, "n_scored": n, "skipped": skipped,
               "V2_loc_video2way": rate("v2_loc"), "V2_uni_video2way": rate("v2_uni"),
               "T2_text2way": rate("t2"), "V8_loc_video8way": rate("v8_loc"),
               "loc_minus_text": (rate("v2_loc") - rate("t2")) if n else None,
               "loc_minus_uni": (rate("v2_loc") - rate("v2_uni")) if n else None}
    tag = os.environ.get("OUT_TAG", "")
    outp = f"{OUT_DIR}/internvl3_14b_localized{('_' + tag) if tag else ''}.json"
    json.dump({"summary": summary, "rows": rows}, open(outp, "w"), indent=2)

    print("\n" + "=" * 64)
    print(f"GATE PROBE B (tightened) -- judge={JUDGE_ID}, moment-localized frames")
    print("=" * 64)
    ff = lambda x, s="+.3f": (format(x, s) if x is not None else "n/a")
    print(f"scored {n}")
    print(f"  V2_loc  video 2-way, LOCALIZED : {ff(summary['V2_loc_video2way'],'.3f')}   (chance 0.500)")
    print(f"  V2_uni  video 2-way, uniform   : {ff(summary['V2_uni_video2way'],'.3f')}   (chance 0.500)")
    print(f"  T2      text  2-way (control)  : {ff(summary['T2_text2way'],'.3f')}   (chance 0.500)")
    print(f"  V8_loc  video 8-way, LOCALIZED : {ff(summary['V8_loc_video8way'],'.3f')}   (chance ~0.125)")
    print(f"  -> localized - text   = {ff(summary['loc_minus_text'])}")
    print(f"  -> localized - uniform= {ff(summary['loc_minus_uni'])}  (did moment-localization help?)")
    vl, lt = summary["V2_loc_video2way"], summary["loc_minus_text"]
    verdict = ("GREEN: stronger judge + localized frames verify bindings from video -> PURSUE"
               if (vl and vl >= 0.70 and lt and lt >= 0.15) else
               "RED (robust): even 14B + moment-localized frames ~ chance / < text -> pivot; diagnosis hardens"
               if (vl and vl < 0.60) else
               "YELLOW: partial signal -> inspect by category")
    print(f"\n  VERDICT: {verdict}")
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
