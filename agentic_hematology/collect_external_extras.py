#!/usr/bin/env python3
"""Save the external-validation extras computed so far only in the chat:
LLD-chosen threshold (Youden's J on LLD CV), cAItomorph operating point at it,
and the paired bootstrap of cAItomorph AUROC (C1+C2 minus C2-only).

Run from agentic_hematology/:
    python collect_external_extras.py
Writes <out>/external_threshold_and_camera.json.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve

ap = argparse.ArgumentParser()
ap.add_argument("--ext-dir", type=Path, default=Path("output/external_cohort"))
ap.add_argument("--c1c2", default="crop_ensemble_c1c2")
ap.add_argument("--c2only", default="crop_ensemble_c2only")
ap.add_argument("--out", type=Path, default=Path("/home/roba.majzoub/research/AgenticHematology/outputs/paper_results"))
ap.add_argument("--n-boot", type=int, default=5000)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

ACUTE = ("ALL", "AML", "APML")
lld = list(csv.DictReader(open(args.ext_dir / args.c1c2 / "lld_crop_cv_per_patient.csv")))
y = np.array([r["true"] in ACUTE for r in lld])
p = np.array([float(r["p_acute"]) for r in lld])
fpr, tpr, thr = roc_curve(y, p)
t = float(thr[np.argmax(tpr - fpr)])


def load(name):
    return {r["patient_id"]: r for r in csv.DictReader(open(args.ext_dir / name / "caitomorph_per_patient.csv"))
            if r["lld_label"]}


a, b = load(args.c2only), load(args.c1c2)
ids = sorted(set(a) & set(b))
yc = np.array([a[i]["lld_label"] != "CML" for i in ids])
pa = np.array([float(a[i]["p_acute"]) for i in ids])
pb = np.array([float(b[i]["p_acute"]) for i in ids])
pred = pb >= t

rng = np.random.default_rng(0)
diffs = []
for _ in range(args.n_boot):
    s = rng.integers(0, len(ids), len(ids))
    if 0 < yc[s].sum() < len(s):
        diffs.append(roc_auc_score(yc[s], pb[s]) - roc_auc_score(yc[s], pa[s]))

res = {
    "lld_threshold_youden": t,
    "lld_bal_acc_at_threshold": float(balanced_accuracy_score(y, p >= t)),
    "caitomorph_c1c2_at_lld_threshold": {
        "n": len(ids), "n_acute": int(yc.sum()), "n_chronic": int((~yc).sum()),
        "bal_acc": float(balanced_accuracy_score(yc, pred)),
        "sensitivity": float(pred[yc].mean()), "specificity": float((~pred[~yc]).mean())},
    "caitomorph_auroc": {"c2only": float(roc_auc_score(yc, pa)), "c1c2": float(roc_auc_score(yc, pb))},
    "paired_bootstrap_auroc_c1c2_minus_c2only": {
        "mean": float(np.mean(diffs)), "ci95": [float(x) for x in np.percentile(diffs, [2.5, 97.5])],
        "n_boot": len(diffs), "seed": 0},
}
(args.out / "external_threshold_and_camera.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=1))
print("Saved", args.out / "external_threshold_and_camera.json")