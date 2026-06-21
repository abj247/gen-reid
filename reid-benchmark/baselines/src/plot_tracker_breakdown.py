#!/usr/bin/env python
"""Two more publication figures:
 A) method x tracker table (mean over 17 VLMs) rendered as a styled, heatmap-shaded table.
 B) full per-model heatmap: 17 VLMs (rows) x [raw | 5 methods x 4 trackers] (cols), accuracy annotated.
"""
import json, glob, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = "/home/ab260989/gen-reid"
TRK = ["botsort", "bytetrack", "deepocsort", "strongsort"]
TRK_LBL = ["BoT-SORT", "ByteTrack", "DeepOCSORT", "StrongSORT"]
METHODS = [("CLIP-link", "{t}"), ("OSNet", "osnet__{t}"), ("CLIP-ReID", "clipreid__{t}"),
           ("TransReID", "transreid__{t}"), ("SOLIDER", "solider__{t}")]
r2a = json.load(open(f"{ROOT}/video_id_mapping.json")).get("real_to_anon", {})

def kof(r):
    if "key" in r: return r["key"]
    v = r.get("video_id"); v = r2a.get(v, v); return f"{v}|{r.get('question_id')}"
def load(p):
    a = {}
    for l in open(p):
        try: r = json.loads(l)
        except: continue
        a[kof(r)] = bool(r.get("is_correct"))
    return a

raw = {p.split("/")[-2]: load(p) for p in glob.glob(f"{ROOT}/results_video_v2/*/predictions.jsonl")}
MODELS = sorted(raw)
data = collections.defaultdict(lambda: collections.defaultdict(dict))
for mn, tm in METHODS:
    for t in TRK:
        for m in MODELS:
            try: data[mn][m][t] = load(f"{ROOT}/results_baseline/{tm.format(t=t)}/{m}/predictions.jsonl")
            except FileNotFoundError: pass
sets = [set(raw[m]) for m in MODELS] + [set(data[mn][m][t]) for mn, _ in METHODS for m in MODELS for t in TRK if t in data[mn][m]]
common = set.intersection(*sets)
def acc(d):
    ks = [k for k in common if k in d]; return 100*sum(d[k] for k in ks)/len(ks) if ks else float("nan")

mnames = [m for m, _ in METHODS]
order = sorted(MODELS, key=lambda m: -acc(raw[m]))
rawmean = np.mean([acc(raw[m]) for m in MODELS])

# ---- method x tracker means (over models) ----
MT = {mn: [np.mean([acc(data[mn][m][t]) for m in MODELS if t in data[mn][m]]) for t in TRK] for mn in mnames}

# ================= FIGURE A: styled table =================
plt.rcParams.update({"font.size": 12, "savefig.bbox": "tight", "figure.dpi": 150})
fig, ax = plt.subplots(figsize=(9.2, 3.4)); ax.axis("off")
cols = ["ID method"] + TRK_LBL + ["avg", "spread"]
rows = []
allvals = np.array([MT[mn] for mn in mnames])
vmin, vmax = allvals.min(), allvals.max()
cmap = LinearSegmentedColormap.from_list("rb", ["#b40426", "#f7f7f7", "#3b4cc0"])  # low=red high=blue
cell_text, cell_col = [], []
for mn in mnames:
    r = MT[mn]; avg = np.mean(r); spr = max(r)-min(r)
    cell_text.append([mn] + [f"{v:.2f}" for v in r] + [f"{avg:.2f}", f"{spr:.2f}"])
    rowc = ["white"]
    for v in r:
        nv = (v - vmin)/(vmax - vmin + 1e-9); rowc.append(cmap(nv))
    rowc += ["#eef3f8", "#fbeee6"]
    cell_col.append(rowc)
tbl = ax.table(cellText=cell_text, colLabels=cols, cellColours=cell_col,
               colColours=["#d9d9d9"]*len(cols), cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.6)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#888");
    if r == 0 or c == 0: cell.set_text_props(weight="bold")
ax.set_title(f"Accuracy (%) by ID method × tracker — mean over 17 VLMs   "
             f"(raw={rawmean:.1f}%, chance=12.5%, N={len(common)})\n"
             f"max tracker spread within a method = {max(max(MT[m])-min(MT[m]) for m in mnames):.2f} pts",
             fontsize=11, pad=10)
plt.savefig(f"{ROOT}/figures_final_tableA_method_x_tracker.png")
plt.savefig(f"{ROOT}/figures_final_tableA_method_x_tracker.pdf"); plt.close()

# ================= FIGURE B: full 17-model x (raw + 5x4) heatmap =================
col_keys = [("raw", None)] + [(mn, t) for mn in mnames for t in TRK]
def cell_acc(m, mn, t):
    return acc(raw[m]) if mn == "raw" else (acc(data[mn][m][t]) if t in data[mn][m] else np.nan)
H = np.array([[cell_acc(m, mn, t) for (mn, t) in col_keys] for m in order])
fig, ax = plt.subplots(figsize=(15.5, 8.2))
im = ax.imshow(H, cmap="viridis", aspect="auto", vmin=11, vmax=27)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=9)
xlabels = ["raw"] + [f"{TRK_LBL[TRK.index(t)]}" for (mn, t) in col_keys[1:]]
ax.set_xticks(range(len(col_keys))); ax.set_xticklabels(xlabels, rotation=90, fontsize=7.5)
for i in range(len(order)):
    for j in range(len(col_keys)):
        v = H[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.2,
                    color="white" if v < 20 else "black")
# method group separators + top labels
ax.axvline(0.5, color="white", lw=2)
for gi, mn in enumerate(mnames):
    x0 = 1 + gi*4
    ax.axvline(x0-0.5, color="white", lw=2)
    ax.text(x0+1.5, -1.1, mn, ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.text(0, -1.1, "raw", ha="center", va="bottom", fontsize=11, fontweight="bold")
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("Accuracy (%)")
ax.set_title("Full baseline matrix: 17 VLMs × (raw + 5 ID-methods × 4 trackers).  Chance = 12.5%.", fontsize=12, pad=26)
plt.savefig(f"{ROOT}/figures_final_tableB_full_model_x_tracker.png")
plt.savefig(f"{ROOT}/figures_final_tableB_full_model_x_tracker.pdf"); plt.close()
print("wrote figures_final_tableA_method_x_tracker.{png,pdf} and figures_final_tableB_full_model_x_tracker.{png,pdf}")
print(f"common QA={len(common)}, raw mean={rawmean:.2f}")
