#!/usr/bin/env python3
"""Per-class detection metrics for each CV fold (YOLO best.pt on its held-out split).

YOLO's results.csv only stores all-class metrics, so each fold's best.pt is
re-validated on the same data/split/imgsz recorded in its args.yaml.

Writes:
  per_class_by_fold.csv   fold, class, instances, precision, recall, mAP50, mAP50-95
  per_class_summary.csv   class, mean and std across folds (+ total instances)

    python yolo_per_class.py --runs wbc_unified/cv/runs/detector/train_folds --out cv_detector_per_class
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import yaml
from ultralytics import YOLO

METRICS = ("precision", "recall", "mAP50", "mAP50-95")


def validate_fold(fold_dir: Path, device: str, split: str | None) -> list[dict]:
    args = yaml.safe_load((fold_dir / "args.yaml").read_text())
    model = YOLO(str(fold_dir / "weights" / "best.pt"))
    res = model.val(
        data=args["data"], imgsz=args.get("imgsz", 640), batch=args.get("batch", 16),
        split=split or "val", device=device, plots=False, verbose=False,
        project=str(fold_dir), name="val_per_class", exist_ok=True,
    )
    names = res.names
    counts = getattr(res, "nt_per_class", None)
    rows = []
    for i, c in enumerate(res.box.ap_class_index):
        p, r, ap50, ap = res.box.class_result(i)
        rows.append({
            "fold": fold_dir.name, "class": names[int(c)],
            "instances": int(counts[int(c)]) if counts is not None else "",
            "precision": p, "recall": r, "mAP50": ap50, "mAP50-95": ap,
        })
    rows.append({"fold": fold_dir.name, "class": "all", "instances": int(sum(counts)) if counts is not None else "",
                 "precision": res.box.mp, "recall": res.box.mr, "mAP50": res.box.map50, "mAP50-95": res.box.map})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True, help="Folder containing fold*/weights/best.pt + args.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--split", default=None, help="Dataset split to validate on (default: val, as in training)")
    args = ap.parse_args()

    rows = []
    for fold_dir in sorted(p for p in args.runs.glob("fold*") if (p / "weights" / "best.pt").is_file()):
        print(f"== {fold_dir.name}")
        rows += validate_fold(fold_dir, args.device, args.split)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "per_class_by_fold.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "class", "instances", *METRICS])
        w.writeheader()
        w.writerows(rows)

    by_class = defaultdict(list)
    for r in rows:
        by_class[r["class"]].append(r)
    with (args.out / "per_class_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "n_folds", "instances", *[f"{m}_{s}" for m in METRICS for s in ("mean", "std")]])
        for cls, rs in sorted(by_class.items(), key=lambda kv: (kv[0] == "all", kv[0])):
            inst = sum(r["instances"] for r in rs if r["instances"] != "")
            vals = [f"{fn([float(r[m]) for r in rs]):.4f}" for m in METRICS for fn in (mean, pstdev)]
            w.writerow([cls, len(rs), inst, *vals])
            print(f"{cls:24s} n_folds={len(rs)} inst={inst:5d} " +
                  " ".join(f"{m}={mean(float(r[m]) for r in rs):.3f}" for m in METRICS))
    print(f"Wrote {args.out}/per_class_by_fold.csv and per_class_summary.csv")


if __name__ == "__main__":
    main()