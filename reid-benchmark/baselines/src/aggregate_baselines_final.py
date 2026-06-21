#!/usr/bin/env python
"""Aggregate the full re-ID baseline matrix and render publication figures.
Methods (identity-assignment strategies):
  raw           : VLM on the full untouched video (no tracking)
  CLIP-link     : tracker tracklets -> CLIP-feature agglomerative linking -> global IDs   (tracker-only)
  OSNet         : tracker + OSNet person-ReID for global IDs
  CLIP-ReID     : tracker + CLIP-ReID for global IDs
  TransReID     : tracker + TransReID for global IDs
  SOLIDER       : tracker + SOLIDER for global IDs
Each conditioned method renders a per-question clip showing only the grounded identity's marked frames.
All on the common QA set; chance = 12.5%.
"""
import json, glob, collections, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/ab260989/gen-reid"
TRK = ["botsort", "bytetrack", "deepocsort", "strongsort"]
# method -> pipeline-dir template ({t}=tracker) under results_baseline, or "RAW"
METHODS = [
    ("raw",       "RAW"),
    ("CLIP-link", "{t}"),
    ("OSNet",     "osnet__{t}"),
    ("CLIP-ReID", "clipreid__{t}"),
    ("TransReID", "transreid__{t}"),
    ("SOLIDER",   "solider__{t}"),
]
r2a = json.load(open(f"{ROOT}/video_id_mapping.json")).get("real_to_anon", {})

def kof(r):
    if "key" in r: return r["key"]
    v = r.get("video_id"); v = r2a.get(v, v); return f"{v}|{r.get('question_id')}"

def load(path):
    a = {}
    for line in open(path):
        try: r = json.loads(line)
        except: continue
        a[kof(r)] = bool(r.get("is_correct"))
    return a

# load raw
raw = {p.split("/")[-2]: load(p) for p in glob.glob(f"{ROOT}/results_video_v2/*/predictions.jsonl")}
MODELS = sorted(raw)

# load conditioned: data[method][model][tracker] = {key:correct}
data = collections.defaultdict(lambda: collections.defaultdict(dict))
for mname, tmpl in METHODS:
    if tmpl == "RAW": continue
    for t in TRK:
        pipe = tmpl.format(t=t)
        for m in MODELS:
            p = f"{ROOT}/results_baseline/{pipe}/{m}/predictions.jsonl"
            try: data[mname][m][t] = load(p)
            except FileNotFoundError: pass

# common QA = intersection across every file (raw + all conditioned)
sets = [set(raw[m]) for m in MODELS]
for mname, tmpl in METHODS:
    if tmpl == "RAW": continue
    for m in MODELS:
        for t in TRK:
            if t in data[mname][m]: sets.append(set(data[mname][m][t]))
common = set.intersection(*sets)
N = len(common)

def acc(d):
    ks = [k for k in common if k in d]
    return 100.0 * sum(d[k] for k in ks) / len(ks) if ks else float("nan")

# build per-(model,method) accuracy = mean over trackers (raw has no tracker)
method_names = [m for m, _ in METHODS]
A = {}  # A[model][method] = accuracy
for m in MODELS:
    A[m] = {"raw": acc(raw[m])}
    for mname, tmpl in METHODS:
        if tmpl == "RAW": continue
        vals = [acc(data[mname][m][t]) for t in TRK if t in data[mname][m]]
        A[m][mname] = float(np.mean(vals)) if vals else float("nan")

# ----- master CSV + console table -----
order = sorted(MODELS, key=lambda m: -A[m]["raw"])
with open(f"{ROOT}/baseline_matrix_final.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model"] + method_names + [f"{x}-raw" for x in method_names[1:]])
    for m in order:
        row = [m] + [f"{A[m][x]:.2f}" for x in method_names] + [f"{A[m][x]-A[m]['raw']:+.2f}" for x in method_names[1:]]
        w.writerow(row)
print(f"common QA = {N}  (chance 12.5%)\n")
hdr = f"{'model':18s}" + "".join(f"{x:>10s}" for x in method_names)
print(hdr); print("-"*len(hdr))
for m in order:
    print(f"{m:18s}" + "".join(f"{A[m][x]:10.1f}" for x in method_names))
print("-"*len(hdr))
means = {x: float(np.mean([A[m][x] for m in MODELS])) for x in method_names}
print(f"{'MEAN':18s}" + "".join(f"{means[x]:10.1f}" for x in method_names))
print("\nmean delta vs raw:")
for x in method_names[1:]:
    print(f"  {x:10s} {means[x]-means['raw']:+.2f}")

# ============================ PUBLICATION FIGURES ============================
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 140, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
COL = {"raw": "#222222", "CLIP-link": "#4C72B0", "OSNet": "#55A868",
       "CLIP-ReID": "#C44E52", "TransReID": "#8172B3", "SOLIDER": "#CCB974"}

# Fig 1: method means across all 17 models (the headline)
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ms = [means[x] for x in method_names]
ses = [np.std([A[m][x] for m in MODELS]) / np.sqrt(len(MODELS)) for x in method_names]
bars = ax.bar(method_names, ms, yerr=ses, capsize=4, color=[COL[x] for x in method_names], edgecolor="black", linewidth=0.6)
ax.axhline(means["raw"], ls="--", c="#222", lw=1, alpha=.7, label=f"raw = {means['raw']:.1f}%")
ax.axhline(12.5, ls=":", c="grey", lw=1, label="chance = 12.5%")
for b, v in zip(bars, ms): ax.text(b.get_x()+b.get_width()/2, v+0.15, f"{v:.1f}", ha="center", fontsize=10)
ax.set_ylabel("Accuracy (%)  — mean over 17 VLMs"); ax.set_ylim(11, max(ms)+2)
ax.set_title("Identity-conditioning vs. raw video: no method beats raw")
ax.legend(frameon=False, fontsize=10); plt.savefig(f"{ROOT}/figures_final_method_means.png"); plt.savefig(f"{ROOT}/figures_final_method_means.pdf"); plt.close()

# Fig 2: delta heatmap (model x method-minus-raw)
cond = method_names[1:]
M = np.array([[A[m][x]-A[m]["raw"] for x in cond] for m in order])
fig, ax = plt.subplots(figsize=(7.0, 7.4))
vmax = np.nanmax(np.abs(M))
im = ax.imshow(M, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(cond))); ax.set_xticklabels(cond, rotation=30, ha="right")
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=9)
for i in range(len(order)):
    for j in range(len(cond)):
        ax.text(j, i, f"{M[i,j]:+.1f}", ha="center", va="center", fontsize=7.5,
                color="black" if abs(M[i,j])<vmax*.6 else "white")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("Δ accuracy vs raw (pts)")
ax.set_title("Per-VLM gain/loss from identity conditioning\n(blue = worse than raw)")
plt.savefig(f"{ROOT}/figures_final_delta_heatmap.png"); plt.savefig(f"{ROOT}/figures_final_delta_heatmap.pdf"); plt.close()

# Fig 3: grouped bars raw vs each method, per model
fig, ax = plt.subplots(figsize=(13, 5.0))
x = np.arange(len(order)); w = 0.14
for i, mn in enumerate(method_names):
    ax.bar(x + (i-2.5)*w, [A[m][mn] for m in order], w, label=mn, color=COL[mn], edgecolor="black", linewidth=0.3)
ax.axhline(12.5, ls=":", c="grey", lw=1)
ax.set_xticks(x); ax.set_xticklabels(order, rotation=40, ha="right", fontsize=9)
ax.set_ylabel("Accuracy (%)"); ax.set_title("Per-VLM accuracy: raw vs. tracker+ID-method (chance 12.5%)")
ax.legend(frameon=False, ncol=6, fontsize=10, loc="upper right"); plt.savefig(f"{ROOT}/figures_final_per_model.png"); plt.savefig(f"{ROOT}/figures_final_per_model.pdf"); plt.close()

# Fig 4: tracker invariance — mean over models, per method x tracker
fig, ax = plt.subplots(figsize=(7.6, 4.4))
xt = np.arange(len(TRK)); w = 0.2
for i, mn in enumerate(["OSNet", "CLIP-ReID", "TransReID", "SOLIDER"]):
    ys = [np.mean([acc(data[mn][m][t]) for m in MODELS if t in data[mn][m]]) for t in TRK]
    ax.bar(xt + (i-1.5)*w, ys, w, label=mn, color=COL[mn], edgecolor="black", linewidth=0.3)
ax.axhline(means["raw"], ls="--", c="#222", lw=1, label=f"raw={means['raw']:.1f}%")
ax.set_xticks(xt); ax.set_xticklabels(TRK); ax.set_ylabel("Accuracy (%) — mean over 17 VLMs")
ax.set_ylim(11, means["raw"]+2); ax.set_title("Tracker choice is irrelevant (all below raw)")
ax.legend(frameon=False, fontsize=9, ncol=3); plt.savefig(f"{ROOT}/figures_final_tracker_invariance.png"); plt.savefig(f"{ROOT}/figures_final_tracker_invariance.pdf"); plt.close()

print("\nwrote: baseline_matrix_final.csv + figures_final_{method_means,delta_heatmap,per_model,tracker_invariance}.{png,pdf}")
