#!/usr/bin/env python3
"""Collect perception-module results into one JSON for the paper.

Run from agentic_hematology/wbc_unified/cv:
    python collect_perception.py --det-dir <folder with per_class_summary.csv and per_class_by_fold.csv>
Writes <out>/perception.json (default: AgenticHematology/outputs/paper_results).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--det-dir", type=Path, required=True)
ap.add_argument("--celltype-root", type=Path, default=Path("runs/celltype"))
ap.add_argument("--attr-comparison", type=Path, default=Path("runs/attribute_probe_comparison.json"))
ap.add_argument("--out", type=Path, default=Path("/home/roba.majzoub/research/AgenticHematology/outputs/paper_results"))
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

# ---- detection: macro mAP ("all" row) + instance-weighted mAP50 per fold
summ = {r["class"]: r for r in csv.DictReader(open(args.det_dir / "per_class_summary.csv"))}
rows = list(csv.DictReader(open(args.det_dir / "per_class_by_fold.csv")))
weighted = []
for f in sorted({r["fold"] for r in rows}):
    cl = [r for r in rows if r["fold"] == f and r["class"] != "all"]
    weighted.append(sum(float(r["mAP50"]) * int(r["instances"]) for r in cl) / sum(int(r["instances"]) for r in cl))
detection = {
    "macro_over_classes": {k: float(v) for k, v in summ["all"].items() if k not in ("class", "n_folds", "instances")},
    "instance_weighted_mAP50": {"mean": st.mean(weighted), "std": st.pstdev(weighted), "per_fold": weighted},
    "source": str((args.det_dir / "per_class_summary.csv").resolve()),
}

# ---- cell type (14 classes) and groups (6), pooled over folds
cell_type = {}
for name in ("celltype", "group"):
    s = json.load(open(args.celltype_root / f"dinobloom_C1C2_{name}" / "pooled_summary.json"))
    cell_type[name] = {k: s[k] for k in ("n_cells", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")}
    if name == "group":
        i = s["labels"].index("blasts")
        cm = s["confusion_matrix"]
        tp = cm[i][i]
        prec = tp / max(sum(r[i] for r in cm), 1)
        rec = tp / max(sum(cm[i]), 1)
        cell_type[name]["blast_f1"] = 2 * prec * rec / (prec + rec)

# ---- attribute probes by training x test camera
attr = json.load(open(args.attr_comparison))
attributes = {tr: {te: {"mean_f1": v["mean_f1"], "n_cells": v["n_cells"],
                        "per_attribute_f1": v["per_attribute_f1"]} for te, v in d.items()}
              for tr, d in attr.items()}

res = {"detection": detection, "cell_type": cell_type, "attribute_probes": attributes}
(args.out / "perception.json").write_text(json.dumps(res, indent=2))
print(json.dumps({"detection_macro": detection["macro_over_classes"],
                  "detection_weighted_mAP50": round(detection["instance_weighted_mAP50"]["mean"], 4),
                  "cell_type": cell_type,
                  "attr_mean_f1_C2probe_on_C2": round(attributes["C2"]["C2"]["mean_f1"], 4)}, indent=1))
print("Saved", args.out / "perception.json")