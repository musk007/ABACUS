#!/usr/bin/env python3
"""Per-class perception figure (ICLR column width).

(a) 14 LLD cell classes: detection mAP50 (mean +/- std over folds) and cell-type
    head F1 (pooled over folds), classes sorted by cell-type support.
(b) 6 clinical groups: cell-group head F1 (pooled over folds).

Run from agentic_hematology/wbc_unified/cv:
    python plot_perception.py --det-dir runs/detector/train_folds/detection_per_class
Writes perception_per_class.pdf and perception_per_class.csv to --out-dir.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--det-dir", type=Path, required=True)
ap.add_argument("--celltype-root", type=Path, default=Path("runs/celltype"))
ap.add_argument("--out-dir", type=Path,
                default=Path("/home/roba.majzoub/research/AgenticHematology/outputs/paper_results/figures"))
args = ap.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)

norm = lambda s: s.strip().lower().replace("_", " ")

# ---- data
det = {norm(r["class"]): r for r in csv.DictReader(open(args.det_dir / "per_class_summary.csv")) if r["class"] != "all"}
ct = {norm(r["class"]): r for r in csv.DictReader(open(args.celltype_root / "dinobloom_C1C2_celltype" / "pooled_per_class.csv"))}
grp = list(csv.DictReader(open(args.celltype_root / "dinobloom_C1C2_group" / "pooled_per_class.csv")))

classes = sorted(set(ct) | set(det), key=lambda c: -int(ct[c]["support"]) if c in ct else 0)
rows = []
for c in classes:
    d, t = det.get(c), ct.get(c)
    rows.append({
        "class": c,
        "det_mAP50_mean": float(d["mAP50_mean"]) if d else np.nan,
        "det_mAP50_std": float(d["mAP50_std"]) if d else np.nan,
        "det_instances": int(d["instances"]) if d and d["instances"] else "",
        "celltype_f1": float(t["f1"]) if t else np.nan,
        "celltype_support": int(t["support"]) if t else "",
    })
grp_rows = sorted(({"group": g["class"], "f1": float(g["f1"]), "support": int(g["support"])} for g in grp),
                  key=lambda r: -r["support"])

with open(args.out_dir / "perception_per_class.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
with open(args.out_dir / "perception_per_group.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(grp_rows[0]))
    w.writeheader(); w.writerows(grp_rows)

# ---- figure
plt.rcParams.update({"font.size": 7})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.9), gridspec_kw={"width_ratios": [1.6, 1]})
DET, CT, GRP = "#9AA0A6", "#2E2A5E", "#2E2A5E"

y = np.arange(len(rows))
h = 0.38
cap = lambda s: s[:1].upper() + s[1:]
labels = [f"{cap(r['class'])} ({r['celltype_support']})" if r["celltype_support"] != "" else cap(r["class"]) for r in rows]
dm = np.array([r["det_mAP50_mean"] for r in rows]); ds = np.array([r["det_mAP50_std"] for r in rows])
cf = np.array([r["celltype_f1"] for r in rows])
ax1.barh(y - h / 2, np.nan_to_num(dm), h, xerr=np.nan_to_num(ds), color=DET, label="Detection mAP50",
         error_kw={"lw": 0.6, "capsize": 1.5})
ax1.barh(y + h / 2, np.nan_to_num(cf), h, color=CT, label="Cell-type F1")
ax1.set_yticks(y); ax1.set_yticklabels(labels, fontsize=6.5); ax1.invert_yaxis()
ax1.set_xlim(0, 1); ax1.set_xlabel("Score")
ax1.set_title("(a) 14 cell classes (support)", fontsize=7.5, loc="left")
fig.legend(*ax1.get_legend_handles_labels(), frameon=False, fontsize=6.5, ncol=2,
           loc="upper center", bbox_to_anchor=(0.35, 1.06))

yg = np.arange(len(grp_rows))
gl = [f"{r['group'].replace('_', ' ')} ({r['support']})" for r in grp_rows]
ax2.barh(yg, [r["f1"] for r in grp_rows], 0.6, color=GRP)
for yy, r in zip(yg, grp_rows):
    ax2.text(r["f1"] + 0.02, yy, f"{r['f1']:.2f}", va="center", fontsize=6.5)
ax2.set_yticks(yg); ax2.set_yticklabels(gl, fontsize=6.5); ax2.invert_yaxis()
ax2.set_xlim(0, 1.1); ax2.set_xticks([0, 0.5, 1.0]); ax2.set_xlabel("F1")
ax2.set_title("(b) 6 clinical groups", fontsize=7.5, loc="left")

for ax in (ax1, ax2):
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(args.out_dir / "perception_per_class.pdf", bbox_inches="tight")
print("Saved", args.out_dir / "perception_per_class.pdf")
print("Saved", args.out_dir / "perception_per_class.csv", "and perception_per_group.csv")
missing = [r["class"] for r in rows if np.isnan(r["det_mAP50_mean"]) or np.isnan(r["celltype_f1"])]
if missing:
    print("Classes missing in one of the two sources (name mismatch?):", missing)