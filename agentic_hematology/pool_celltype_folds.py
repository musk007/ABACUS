#!/usr/bin/env python3
"""Pool cell-type CV results across folds (final epoch of each fold).

Sums each fold's confusion matrix (aligned by class name) and reports pooled
per-class precision / recall / F1 / support plus overall accuracy, balanced
accuracy, macro-F1 (classes with support) and weighted F1. Also lists each
fold's own numbers.

    python pool_celltype_folds.py runs/celltype/dinobloom_C1C2 runs/celltype/dinobloom_C1C2_group
Writes pooled_per_class.csv and pooled_summary.json inside each project folder.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


def load_folds(project: Path) -> list[tuple[int, dict]]:
    out = []
    for p in sorted(project.glob("*_cv_fold*/history_fold*.json")):
        fold = int(p.stem.split("fold")[-1])
        out.append((fold, json.loads(p.read_text())[-1]["metrics"]))
    return sorted(out)


def pool(project: Path) -> None:
    folds = load_folds(project)
    if not folds:
        print(f"{project}: no history_fold*.json found")
        return
    labels = []
    for _, m in folds:
        labels += [c for c in m["labels_present"] if c not in labels]
    idx = {c: i for i, c in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for _, m in folds:
        pos = [idx[c] for c in m["labels_present"]]
        cm[np.ix_(pos, pos)] += np.array(m["confusion_matrix"], dtype=np.int64)

    support = cm.sum(1)
    predicted = cm.sum(0)
    tp = np.diag(cm)
    prec = np.divide(tp, predicted, out=np.zeros(len(tp)), where=predicted > 0)
    rec = np.divide(tp, support, out=np.zeros(len(tp)), where=support > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(len(tp)), where=(prec + rec) > 0)
    has = support > 0
    summary = {
        "folds": [f for f, _ in folds],
        "n_cells": int(support.sum()),
        "accuracy": float(tp.sum() / support.sum()),
        "balanced_accuracy": float(rec[has].mean()),
        "macro_f1": float(f1[has].mean()),
        "weighted_f1": float((f1 * support).sum() / support.sum()),
        "per_fold": {f: {k: m[k] for k in ("macro_f1", "weighted_f1", "balanced_accuracy")} for f, m in folds},
        "labels": labels,
        "confusion_matrix": cm.tolist(),
    }

    with (project / "pooled_per_class.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "support", "precision", "recall", "f1"])
        for i, c in enumerate(labels):
            w.writerow([c, int(support[i]), f"{prec[i]:.4f}", f"{rec[i]:.4f}", f"{f1[i]:.4f}"])
    (project / "pooled_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n== {project}  (folds {summary['folds']}, {summary['n_cells']} cells)")
    print(f"{'class':24s} {'support':>7s} {'prec':>6s} {'rec':>6s} {'f1':>6s}")
    for i in np.argsort(-support):
        print(f"{labels[i]:24s} {support[i]:7d} {prec[i]:6.3f} {rec[i]:6.3f} {f1[i]:6.3f}")
    print(f"accuracy {summary['accuracy']:.4f}  balanced_acc {summary['balanced_accuracy']:.4f}  "
          f"macro_f1 {summary['macro_f1']:.4f}  weighted_f1 {summary['weighted_f1']:.4f}")
    for f, m in folds:
        print(f"  fold{f}: macro_f1 {m['macro_f1']:.4f}  weighted_f1 {m['weighted_f1']:.4f}  "
              f"bal_acc {m['balanced_accuracy']:.4f}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        pool(Path(arg))