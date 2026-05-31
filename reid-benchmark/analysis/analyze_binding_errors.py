#!/usr/bin/env python3
"""
Binding (who/what) error decomposition — the method-motivating analysis.

For each trap/hard question we take the GOLD option and the MAGNET distractor
(the single wrong option the model committee converges on), and use a local LLM
to decompose both into (WHO = referenced identity/entity) and (WHAT = action/
attribute/state). We then classify the committee's error:

  WHAT_error  : magnet keeps the SAME who, swaps the what  (right person, wrong attribute)
  WHO_error   : magnet keeps the SAME what, swaps the who  (right attribute, wrong person)
  BOTH_diff   : magnet differs in both
  UNCLEAR     : LLM could not decompose

A high WHAT_error rate is the empirical signature of a feature-conjunction /
binding failure: the model perceives the features but binds identity to the
wrong attribute. That motivates an explicit identity-attribute binding
mechanism rather than "more memory".

Runs in the `reid` env (Gemma-3-12B). Reads results_video_v2 predictions to find
the committee magnet per question.

Usage:
    python analyze_binding_errors.py \
        --bench combined_all_hard_v3_retagged.json \
        --video_dir results_video_v2 --mapping video_id_mapping.json \
        --model google/gemma-3-12b-it --out results/binding_errors.json
"""
import argparse, glob, json, re, time
from collections import Counter, defaultdict
from pathlib import Path
import torch


def load_bench(path):
    bench = json.load(open(path))
    meta, opts, gold, qtext = {}, {}, {}, {}
    for v in bench["videos"]:
        vid = v["video_id"]
        for q in v.get("questions", []):
            qid = q.get("question_id")
            if not qid: continue
            k = (vid, qid)
            m = q.get("metadata", {})
            meta[k] = {"capability": m.get("capability","?"), "reid": m.get("reid_canonical","?")}
            opts[k] = q.get("options", {})
            gold[k] = (q.get("correct_answer") or q.get("answer","")).strip().upper()[:1]
            qtext[k] = q.get("question_text", q.get("question",""))
    return meta, opts, gold, qtext


def load_preds(d, key_map):
    preds = {}
    for line in open(d):
        r = json.loads(line)
        vid = key_map.get(r.get("video_id"), r.get("video_id"))
        k = (vid, r.get("question_id"))
        if k in preds: continue
        preds[k] = r.get("predicted","?")
    return preds


PROMPT = """You are analyzing a multiple-choice video question about identifying people and their actions.

QUESTION: {q}

The CORRECT answer is: "{gold}"
A WRONG answer that models often choose instead is: "{magnet}"

Compare the WRONG answer to the CORRECT answer. Choose exactly one label:

WHAT_error: the wrong answer refers to the SAME person/entity as the correct answer, but a DIFFERENT action, attribute, state, or object (right who, wrong what).
WHO_error: the wrong answer refers to a DIFFERENT person/entity, but a SIMILAR action/attribute (wrong who, right what).
BOTH_diff: the wrong answer differs in BOTH the person/entity and the action/attribute.
UNCLEAR: cannot tell.

Reply with ONLY the label."""


def classify(text):
    t = text.strip().upper()
    for lab in ("WHAT_ERROR", "WHO_ERROR", "BOTH_DIFF", "UNCLEAR"):
        if lab.replace("_","") in t.replace("_","").replace(" ",""):
            return lab.lower()
    if "WHAT" in t: return "what_error"
    if "WHO" in t: return "who_error"
    if "BOTH" in t: return "both_diff"
    return "unclear"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="combined_all_hard_v3_retagged.json")
    ap.add_argument("--video_dir", default="results_video_v2")
    ap.add_argument("--mapping", default="video_id_mapping.json")
    ap.add_argument("--model", default="google/gemma-3-12b-it")
    ap.add_argument("--out", default="results/binding_errors.json")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    real_to_anon = json.load(open(args.mapping))["real_to_anon"]
    meta, opts, gold, qtext = load_bench(args.bench)
    VID = {d.split("/")[-2]: load_preds(d, real_to_anon)
           for d in sorted(glob.glob(f"{args.video_dir}/*/predictions.jsonl"))}
    models = sorted(VID); nm = len(models)
    common = set(meta)
    for m in VID: common &= set(VID[m])
    common = sorted(common)

    # Find the magnet distractor per question (most-voted wrong option), keep only
    # questions where a real magnet exists (>=1/3 of models share one wrong option).
    targets = []
    for k in common:
        votes = Counter(VID[m][k] for m in models)
        g = gold[k] if gold[k] in opts[k] else (list(opts[k])[0] if opts[k] else "A")
        wrong = Counter({o: c for o, c in votes.items() if o != g and o in opts[k]})
        if not wrong: continue
        mo, mc = wrong.most_common(1)[0]
        if mc >= nm/3 and g in opts[k] and mo in opts[k]:
            targets.append((k, g, mo, mc))
    if args.limit:
        targets = targets[:args.limit]
    print(f"Questions with a committee magnet distractor: {len(targets)}", flush=True)

    # Load Gemma-3-12B
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    proc = AutoProcessor.from_pretrained(args.model)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True).eval()
    tok = proc.tokenizer
    tok.padding_side = "left"
    print("Model loaded.", flush=True)

    results = []
    t0 = time.time()
    bs = args.batch_size
    for s in range(0, len(targets), bs):
        batch = targets[s:s+bs]
        prompts = [PROMPT.format(q=qtext[k], gold=opts[k][g], magnet=opts[k][mo])
                   for (k, g, mo, mc) in batch]
        msgs = [[{"role":"user","content":[{"type":"text","text":p}]}] for p in prompts]
        inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        plen = inputs["input_ids"].shape[1]
        outs = tok.batch_decode(gen[:, plen:], skip_special_tokens=True)
        for (k, g, mo, mc), o in zip(batch, outs):
            results.append({"video_id": k[0], "question_id": k[1],
                            "capability": meta[k]["capability"], "reid": meta[k]["reid"],
                            "gold": g, "magnet": mo, "magnet_votes": mc,
                            "label": classify(o), "raw": o.strip()[:24]})
        if (s//bs) % 10 == 0 or s+bs >= len(targets):
            done = min(s+bs, len(targets))
            print(f"  [{done}/{len(targets)}] {done/max(time.time()-t0,1e-3):.1f} q/s", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)

    # Summary
    dist = Counter(r["label"] for r in results)
    print("\n=== BINDING ERROR DECOMPOSITION ===", flush=True)
    tot = len(results)
    for lab in ("what_error","who_error","both_diff","unclear"):
        print(f"  {lab:11s}: {dist.get(lab,0):5d} ({100*dist.get(lab,0)/max(tot,1):.1f}%)", flush=True)
    decided = sum(dist.get(l,0) for l in ("what_error","who_error","both_diff"))
    if decided:
        print(f"\n  Among decided: WHAT(binding)={100*dist.get('what_error',0)/decided:.1f}%  "
              f"WHO={100*dist.get('who_error',0)/decided:.1f}%  BOTH={100*dist.get('both_diff',0)/decided:.1f}%", flush=True)
    # by reid category
    print("\n  WHAT_error (binding) rate by reid category (decided only):", flush=True)
    byr = defaultdict(lambda: [0,0])
    for r in results:
        if r["label"] in ("what_error","who_error","both_diff"):
            byr[r["reid"]][1]+=1
            if r["label"]=="what_error": byr[r["reid"]][0]+=1
    for c,(a,t) in sorted(byr.items(), key=lambda x:-x[1][0]/max(x[1][1],1)):
        if t>=20: print(f"    {c:24s} {100*a/t:5.1f}% ({a}/{t})", flush=True)
    print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
