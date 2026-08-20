#!/usr/bin/env python
"""MEMORY BANK, read phase: chunk the video, keep the encoded memory, answer from the chunks whose
memory is most similar to the question.

    write:  video -> 8 chunks -> 32 frames/chunk encoded ONCE into LLM-embedding space  (the bank)
    read:   question -> CLIP -> rank chunks -> splice the winning chunks' stored tokens -> answer

Every arm spends EXACTLY 32 frames x 256 tokens = 8,192 visual tokens, byte-matched to the
`uniform32` arm this project has been reporting since the VTM ladder. Budget parity is structural,
not something a caller has to remember.

    arm            frames drawn from                                   decisive contrast
    mb_uniform32   32 uniform over the whole video                     (the baseline, re-run here)
    mb_top1        32 from the single highest-scoring chunk            vs mb_rand1
    mb_top2        16 from each of the top-2 chunks                    vs mb_rand2
    mb_rand1       32 from ONE seeded-random chunk        MANDATORY control
    mb_rand2       16 from each of TWO seeded-random chunks MANDATORY control
    mb_oracle      32 from the chunk holding the evidence centre       ceiling, not a method

Why the baseline is re-run inside this file
-------------------------------------------
`uniform32` already exists in analysis3/memory/vtm_results/internvl3-14b.jsonl at the same token
budget, and comparing against it directly would have been free. It is re-run anyway because that
row came from a different file with a different prompt builder, and this project has already been
bitten twice by exactly that kind of pipeline gap -- the CLIP-keyframe arms are still carrying an
unresolved "rendered mp4 vs source video" confound, and the leaderboard-vs-VTM harness differed by
3.8 points on identical questions. One code path for every arm removes the question entirely. The
external `uniform32` becomes a cross-check rather than the comparator.

Why CLIP does the ranking and the VLM never does
------------------------------------------------
See build_clip_index.py. Short version: the VLM-self-scoring version of this idea is QuestMem, and
its chunk scoring never beat random chunks (+0.66, p=0.70); the VLM-embedding-similarity version is
qcond, and it LOST to its random control (-0.42, MDE 0.79). CLIP retrieval is the only ranking
signal in this project with evidence behind it (55.5% chunk hit vs the self-scout's 37.1%, and
+1.95/+2.01/+2.65 at frame level on three backbones).

Run:
  python analysis3/membank/run_membank.py --selftest --subset core
  python analysis3/membank/run_membank.py --arms mb_uniform32,mb_top2,mb_rand2 --subset core
Analyse:
  python analysis3/memory/analyze_gates.py --files 'analysis3/membank/results/*.jsonl' \
      --pairs mb_top2:mb_rand2,mb_top1:mb_rand1,mb_top2:mb_uniform32,mb_oracle:mb_top2
"""
import argparse
import glob
import json
import os
import sys
import time
import zlib

import numpy as np

from persistqa.paths import ROOT  # noqa: E402

sys.path.insert(0, str(ROOT / "solutions" / "cairn"))
sys.path.insert(0, str(ROOT / "solutions" / "shared"))

from prompt_blocks import build_query_blocks, score_prebuilt  # noqa: E402
from build_clip_index import chunk_bounds  # noqa: E402

LETTERS = "ABCDEFGH"
OUTD = f"{ROOT}/solutions/cairn/results"
INDEX_DIR = f"{ROOT}/solutions/cairn/index"

N_CHUNKS = 8
BANK_PER_CHUNK = 32          # frames encoded per chunk -> 256 frames of memory per video
N_FRAMES = 32                # frames every arm feeds
TOK = 256                    # native InternVL tokens/frame; 32 x 256 = 8,192 visual tokens
TOTAL_TOKENS = N_FRAMES * TOK

ARMS = ("mb_uniform32", "mb_top1", "mb_top2", "mb_rand1", "mb_rand2", "mb_oracle")

# Canonical names use the method prefix; the mb_ ("memory bank") names are the ones written
# into result files and are what every existing artefact and the analysis code join on, so
# both are accepted and the mb_ form remains what is emitted.
ARM_ALIASES = {a.replace("mb_", "cairn_"): a for a in ARMS}


def canonical_arm(name):
    """Accept either naming; always return the mb_ form used on disk."""
    return ARM_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# retrieval (pure, CPU-testable)
# ---------------------------------------------------------------------------

def chunk_scores(feats, chunk_id, qvec, pool="max"):
    """(n_chunks,) similarity of each chunk to the query.

    `max` by default, not `mean`. The evidence window here has a median span of 4.4 s inside a
    ~24 s chunk, so a chunk earns its place because ONE of its frames matches, not because all of
    them do -- averaging over 32 frames buries exactly the signal being looked for. `mean` is kept
    selectable so the choice is an ablation rather than an assumption.
    """
    sims = feats @ qvec
    out = np.full(N_CHUNKS, -1e9, dtype=np.float32)
    for c in range(N_CHUNKS):
        m = chunk_id == c
        if m.any():
            out[c] = sims[m].max() if pool == "max" else sims[m].mean()
    return out


def seeded_chunks(key, k, n_chunks=N_CHUNKS):
    """k distinct chunks from a question-key-seeded RNG. The control must be reproducible and must
    differ per QUESTION -- seeding per video would collapse it into a per-video control."""
    rng = np.random.RandomState(zlib.crc32(f"{key}|mb_rand".encode()) & 0xFFFFFFFF)
    return sorted(int(c) for c in rng.choice(n_chunks, size=k, replace=False))


def evidence_chunk(rec, n_total):
    """Chunk holding the CENTRE of the oracle evidence window, or None."""
    if rec is None or rec.get("t0") is None:
        return None
    fps = float(rec.get("fps") or 25.0)
    c = 0.5 * (float(rec["t0"]) + float(rec["t1"])) * fps
    c = int(max(0, min(n_total - 1, round(c))))
    for i, (lo, hi) in enumerate(chunk_bounds(n_total, N_CHUNKS)):
        if lo <= c <= hi:
            return i
    return N_CHUNKS - 1


def arm_chunks(arm, key, scores, ev_chunk):
    """Which chunks this arm reads, and how many frames it takes from each. None -> drop the key."""
    if arm == "mb_uniform32":
        return None                                  # handled separately: whole-video grid
    order = list(np.argsort(-scores))
    if arm == "mb_top1":
        return [(int(order[0]), N_FRAMES)]
    if arm == "mb_top2":
        return [(int(c), N_FRAMES // 2) for c in sorted(order[:2])]
    if arm == "mb_rand1":
        return [(seeded_chunks(key, 1)[0], N_FRAMES)]
    if arm == "mb_rand2":
        return [(c, N_FRAMES // 2) for c in seeded_chunks(key, 2)]
    if arm == "mb_oracle":
        return None if ev_chunk is None else [(int(ev_chunk), N_FRAMES)]
    raise ValueError(arm)


def bank_rows(plan, per_chunk=BANK_PER_CHUNK):
    """Positions INTO THE BANK for an arm's plan, ascending. Bank row = chunk*per_chunk + j.

    Frames are taken uniformly from within each chosen chunk's stored block, so an arm that reads
    one chunk covers it densely and an arm that reads two covers each half as densely -- the total
    is always N_FRAMES, which is what keeps the budget matched.
    """
    rows = []
    for c, n in plan:
        j = np.linspace(0, per_chunk - 1, n).astype(int)
        rows.extend(int(c * per_chunk + x) for x in j)
    rows = sorted(rows)
    if len(set(rows)) != len(rows):
        raise AssertionError(f"duplicate bank rows in plan {plan}")
    return rows


def frame_prompt(n):
    return "".join(f"Frame {i + 1}: <image>\n" for i in range(n))


# ---------------------------------------------------------------------------

def load_index(vid):
    p = os.path.join(INDEX_DIR, f"{vid}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return {"feats": z["feats"], "chunk_id": z["chunk_id"], "n_total": int(z["n_total"]),
            "fps": float(z["fps"])}


def selftest(args, arms):
    """Replay retrieval + budget geometry over every key. No GPU, no weights, no video decode."""
    from mem_common import load_benchmark, load_evidence
    import bench_filters
    bench = load_benchmark()
    ev = load_evidence()
    keys = [f"{v}|{q}" for v, qs in bench.items() for q in qs]
    keep = bench_filters.subset_keys(args.subset, args.subset_file)
    if keep is not None:
        keys = [k for k in keys if k in keep]
    und = bench_filters.project_undecodable()
    keys = sorted(k for k in keys if k.split("|")[0] not in und)
    if args.limit:
        keys = keys[: args.limit]

    n_ok = n_noidx = n_drop = 0
    hits = {a: 0 for a in arms if a not in ("mb_uniform32",)}
    scored = 0
    for key in keys:
        vid = key.split("|")[0]
        ix = load_index(vid)
        if ix is None:
            n_noidx += 1
            continue
        evc = evidence_chunk(ev.get(key), ix["n_total"])
        # a deterministic stand-in query vector: the selftest checks GEOMETRY, not retrieval quality
        qv = ix["feats"].mean(0)
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        sc = chunk_scores(ix["feats"], ix["chunk_id"], qv, args.pool)
        plans = {}
        bad = False
        for a in arms:
            if a == "mb_uniform32":
                plans[a] = None
                continue
            p = arm_chunks(a, key, sc, evc)
            if p is None:
                bad = True
                break
            plans[a] = p
        if bad:
            n_drop += 1
            continue
        for a, p in plans.items():
            n = N_FRAMES if p is None else sum(x[1] for x in p)
            assert n == N_FRAMES, f"{key}/{a}: {n} frames, expected {N_FRAMES}"
            if p is not None:
                rows = bank_rows(p)
                assert len(rows) == N_FRAMES and len(set(rows)) == N_FRAMES
                assert max(rows) < N_CHUNKS * BANK_PER_CHUNK
                if evc is not None and any(c == evc for c, _ in p):
                    hits[a] += 1
        scored += 1
        n_ok += 1

    print(f"[selftest] arms={list(arms)}")
    print(f"[selftest] {n_ok} keys scorable in every arm | {n_noidx} without a CLIP index | "
          f"{n_drop} dropped from ALL arms")
    print(f"[selftest] every arm feeds exactly {N_FRAMES} frames x {TOK} tok = {TOTAL_TOKENS} "
          f"visual tokens")
    if scored:
        print("[selftest] chunk-hit vs the ORACLE evidence chunk (random 1-of-8 = 12.5%, "
              "2-of-8 = 25.0%) -- note this uses a stand-in query vector, so it measures the "
              "GEOMETRY only, not the real retrieval:")
        for a, h in hits.items():
            print(f"             {a:12s} {100.0 * h / scored:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="internvl3-14b")
    ap.add_argument("--arms", default="mb_uniform32,mb_top2,mb_rand2,mb_top1,mb_rand1")
    ap.add_argument("--subset", default="core", choices=["core", "all"])
    ap.add_argument("--subset_file", default="")
    ap.add_argument("--pool", default="max", choices=["max", "mean"])
    ap.add_argument("--query_mode", default="referent", choices=["question", "referent"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--encode_batch", type=int, default=32)
    ap.add_argument("--tag", default="")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        assert a in ARMS, f"unknown arm {a}; expected one of {ARMS}"

    if args.selftest:
        selftest(args, arms)
        return

    import decord
    import torch
    import open_clip
    from PIL import Image
    from torchvision import transforms
    from mem_common import load_benchmark, load_evidence, VIDEO_DIR
    from evaluate_vlm_bm import create_model
    from contrastive_decode import LetterScorer
    from tgrpo_train import resolve_video
    from vtm_bank import encode_frames
    import bench_filters

    os.makedirs(OUTD, exist_ok=True)
    sub = "" if args.subset == "core" else "__full"
    out_path = os.path.join(
        OUTD, f"{args.model}{sub}__{args.pool}_{args.query_mode}"
              f"__{'_'.join(sorted(arms))}{args.tag}__membank.jsonl")
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                r = json.loads(line)
                done.add((r["key"], r["arm"]))
            except Exception:
                pass

    bench = load_benchmark()
    ev = load_evidence()
    keys = [f"{v}|{q}" for v, qs in bench.items() for q in qs]
    keep = bench_filters.subset_keys(args.subset, args.subset_file)
    if keep is not None:
        keys = [k for k in keys if k in keep]
    und = bench_filters.project_undecodable()
    keys = sorted(k for k in keys if k.split("|")[0] not in und)
    if args.limit:
        keys = keys[: args.limit]
    todo = [k for k in keys if any((k, a) not in done for a in arms)]
    print(f"[membank] {len(keys)} keys, {len(todo)} with work left | arms={arms} | "
          f"pool={args.pool} query={args.query_mode}", flush=True)
    print(f"[membank] bank {N_CHUNKS}x{BANK_PER_CHUNK}={N_CHUNKS*BANK_PER_CHUNK} frames/video; "
          f"every arm feeds {N_FRAMES}x{TOK}={TOTAL_TOKENS} visual tokens", flush=True)
    if not todo:
        return

    dev = "cuda"
    cm, _, cprep = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    cm = cm.eval().to(dev)
    ctok = open_clip.get_tokenizer("ViT-B-32-quickgelu")

    def qvec(q):
        t = q.get("question_text", "")
        if args.query_mode == "referent":
            t = f"{t} {q.get('metadata', {}).get('temporal_anchor', '') or ''}".strip()
        with torch.no_grad():
            f = torch.nn.functional.normalize(
                cm.encode_text(ctok([t[:300]]).to(dev)).float(), dim=-1)
        return f.cpu().numpy()[0]

    vlm = create_model(args.model, num_frames=32, max_pixels=(448, 448))
    vlm.load_model()
    sc = LetterScorer(vlm)
    assert sc.model.num_image_token == TOK, f"expected {TOK} tok/frame, got {sc.model.num_image_token}"
    tf = transforms.Compose([
        transforms.Resize((448, 448)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    by_video = {}
    for k in todo:
        by_video.setdefault(k.split("|")[0], []).append(k)

    n_scores = n_noidx = 0
    fout = open(out_path, "a")
    t0 = time.time()
    for vid, vkeys in sorted(by_video.items()):
        ix = load_index(vid)
        if ix is None:
            n_noidx += 1
            continue
        try:
            vr = decord.VideoReader(resolve_video(VIDEO_DIR, vid), num_threads=2)
            n_total = len(vr)
        except Exception as e:
            print(f"  [skip video] {vid}: {str(e)[:90]}", flush=True)
            continue

        # ---- WRITE PHASE: encode the bank ONCE for this video -------------------------------
        # 256 frames on the chunk grid, in LLM-embedding space. This is the memory; every arm and
        # every question of this video reads from it and nothing is ever re-encoded.
        bounds = chunk_bounds(n_total, N_CHUNKS)
        bank_idx = []
        for lo, hi in bounds:
            bank_idx.extend(int(x) for x in np.linspace(lo, hi, BANK_PER_CHUNK).astype(int))
        try:
            parts = []
            for i in range(0, len(bank_idx), args.encode_batch):
                arr = vr.get_batch(bank_idx[i: i + args.encode_batch]).asnumpy()
                pil = [Image.fromarray(a).convert("RGB") for a in arr]
                parts.append(encode_frames(sc, tf, pil, vlm.device, batch=args.encode_batch))
                del pil, arr
            bank = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
            del parts
            uni_idx = np.linspace(0, n_total - 1, N_FRAMES).astype(int)
            parts = []
            for i in range(0, len(uni_idx), args.encode_batch):
                arr = vr.get_batch([int(x) for x in uni_idx[i: i + args.encode_batch]]).asnumpy()
                pil = [Image.fromarray(a).convert("RGB") for a in arr]
                parts.append(encode_frames(sc, tf, pil, vlm.device, batch=args.encode_batch))
                del pil, arr
            uni_bank = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
            del parts
        except Exception as e:
            print(f"  [encode err] {vid}: {str(e)[:110]}", flush=True)
            del vr
            continue

        # ---- READ PHASE ---------------------------------------------------------------------
        for key in sorted(vkeys):
            qid = key.split("|", 1)[1]
            q = bench[vid][qid]
            opts = q["options"] if isinstance(q["options"], dict) else {
                LETTERS[i]: t for i, t in enumerate(q["options"])}
            gold = (q.get("correct_answer") or "")[:1].upper()
            mcq = "\n" + vlm.format_mcq_prompt(q["question_text"], opts)
            evc = evidence_chunk(ev.get(key), n_total)
            sc_chunks = chunk_scores(ix["feats"], ix["chunk_id"], qvec(q), args.pool)

            for arm in arms:
                if (key, arm) in done:
                    continue
                try:
                    plan = arm_chunks(arm, key, sc_chunks, evc)
                    if arm != "mb_uniform32" and plan is None:
                        continue                      # oracle with no evidence record
                    if plan is None:
                        vit = uni_bank
                        used, hit = [], None
                    else:
                        rows = bank_rows(plan)
                        vit = bank[rows]
                        used = [int(c) for c, _ in plan]
                        hit = None if evc is None else bool(evc in used)
                    F, T, C = vit.shape
                    assert F == N_FRAMES and T == TOK, f"{key}/{arm}: bank slice {vit.shape}"
                    query = build_query_blocks(sc, frame_prompt(F) + mcq, [TOK] * F)
                    _, lp = score_prebuilt(sc, query, vit.reshape(F * T, C))
                    pred = LETTERS[int(np.argmax(np.asarray(lp)[: len(opts)]))]
                except Exception as e:
                    print(f"  [score err] {key} {arm}: {str(e)[:110]}", flush=True)
                    continue

                fout.write(json.dumps({
                    "key": key, "video_id": vid, "question_id": qid, "arm": arm,
                    "n_frames": N_FRAMES, "tokens_per_frame": TOK, "visual_tokens": TOTAL_TOKENS,
                    "predicted": pred, "correct": gold, "is_correct": pred == gold,
                    "capability": q.get("metadata", {}).get("capability", "unknown"),
                    "chunks_used": used, "evidence_chunk": evc, "chunk_hit": hit,
                    "chunk_scores": [round(float(x), 4) for x in sc_chunks],
                    "pool": args.pool, "query_mode": args.query_mode,
                    "video_nframes": int(n_total),
                }) + "\n")
                fout.flush()
                n_scores += 1

        del bank, uni_bank, vr
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        el = max(time.time() - t0, 1e-3)
        print(f"  [{vid}] {n_scores} scores, {n_scores/el:.3f}/s", flush=True)

    fout.close()
    print(f"[summary] {n_scores} scores -> {out_path}", flush=True)
    print(f"[summary] {n_noidx} videos had no CLIP index (run build_clip_index.py first)",
          flush=True)
    print("[summary] decisive contrasts: mb_top2 - mb_rand2 and mb_top1 - mb_rand1. "
          "mb_oracle is a ceiling, never a method.", flush=True)


if __name__ == "__main__":
    main()
