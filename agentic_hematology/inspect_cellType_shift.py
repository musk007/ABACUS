#!/usr/bin/env python3
"""Print the predicted cell-type distribution per diagnosis group on an
external cohort of pre-cropped cells, and compare it to the LLD ground truth.

Purpose: the patient-level features are percentages over cell types, so if the
cell-type head mis-types under domain shift, every downstream number is wrong
in a way that is invisible from accuracy alone. Healthy stem-cell donors are the
control — their smears should be dominated by neutrophils and lymphocytes with
essentially no blasts.

    python inspect_celltype_shift.py \\
      --data-root /l/users/roba.majzoub/cAItomorph/caitomorph_test_set \\
      --celltype-weights wbc_unified/cv/runs/celltype/dinobloom_celltype_train/best_celltype.pt \\
      --max-cells 100 --device 0 \\
      --out outputs/external/celltype_shift_dinobloom.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
CV_ROOT = HERE / "wbc_unified" / "cv"
for p in (HERE, HERE.parent, CV_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from wbc_unified.cv.eval_cellType import load_model  # noqa: E402
from wbc_unified.cv.train_cellType import embed_tensor_batch  # noqa: E402
from utils.e2e_attributes import crops_to_tensor  # noqa: E402

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
BLAST_CLASSES = {"Myeloblast", "Lymphoblast", "Monoblast", "Abnormal promyelocyte"}


def patient_crops(patients_dir: Path, pid: str, max_cells: int | None) -> list[Path]:
    d = patients_dir / pid
    if not d.is_dir():
        return []
    crops = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if max_cells and len(crops) > max_cells:
        idx = np.linspace(0, len(crops) - 1, max_cells).astype(int)
        crops = [crops[i] for i in idx]
    return crops


@torch.no_grad()
def type_crops(model, embedder, class_names, imgsz, device, paths, batch=64):
    out, confs = [], []
    for start in range(0, len(paths), batch):
        chunk = paths[start : start + batch]
        imgs = [Image.open(p).convert("RGB") for p in chunk]
        x = crops_to_tensor(imgs, imgsz)
        logits = model(embed_tensor_batch(embedder, x)) if embedder is not None else model(x.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        idx = probs.argmax(axis=1)
        out.extend(class_names[i] for i in idx)
        confs.extend(float(probs[k, i]) for k, i in enumerate(idx))
    return out, confs


def lld_reference(celltype_manifest: Path | None) -> dict[str, float]:
    """GT cell-type distribution over the LLD test split, for comparison."""
    if not celltype_manifest or not celltype_manifest.is_file():
        return {}
    from data.celltype_dataset import CLASS_NAMES

    counts: Counter = Counter()
    with celltype_manifest.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["split"] != "test":
                continue
            counts[CLASS_NAMES[int(row["class_id"])]] += 1
    total = sum(v for k, v in counts.items() if k != "None") or 1
    return {k: round(100.0 * v / total, 1) for k, v in counts.items() if k != "None"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--celltype-weights", type=Path, required=True)
    ap.add_argument("--celltype-manifest", type=Path, default=CV_ROOT / "generated" / "celltype_manifest.csv")
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-cells", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    model, embedder, class_names, imgsz, ckpt = load_model(args.celltype_weights, device)
    print(f"cell-type model: arch={ckpt.get('arch')} target={ckpt.get('target')} imgsz={imgsz}")

    with (args.data_root / "metadata.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    patients_dir = args.data_root / "patients"

    by_group: dict[str, Counter] = defaultdict(Counter)
    conf_by_group: dict[str, list[float]] = defaultdict(list)

    for r in tqdm(rows, desc="patients"):
        crops = patient_crops(patients_dir, r["patient_id"], args.max_cells)
        if not crops:
            continue
        types, confs = type_crops(model, embedder, class_names, imgsz, device, crops, args.batch)
        by_group[r["diagnosis_coarse"]].update(types)
        conf_by_group[r["diagnosis_coarse"]].extend(confs)

    ref = lld_reference(args.celltype_manifest)

    report = {}
    all_types = sorted({t for c in by_group.values() for t in c})
    for grp in sorted(by_group):
        c = by_group[grp]
        total = sum(c.values())
        informative = sum(v for k, v in c.items() if k not in ("None", "Unknown")) or 1
        blasts = sum(c.get(b, 0) for b in BLAST_CLASSES)
        print(f"\n=== {grp}  ({total} cells, {100 * (total - informative) / total:.1f}% None) ===")
        print(f"    mean typing confidence: {np.mean(conf_by_group[grp]):.3f}")
        print(f"    blast% of informative : {100 * blasts / informative:.1f}")
        for t in all_types:
            if c.get(t):
                print(f"      {t:24s} {100 * c[t] / informative:6.1f}%  ({c[t]})")
        report[grp] = {
            "n_cells": total,
            "pct_none": round(100 * (total - informative) / total, 2),
            "mean_confidence": float(np.mean(conf_by_group[grp])),
            "blast_pct": round(100 * blasts / informative, 2),
            "distribution_pct": {t: round(100 * c[t] / informative, 2) for t in all_types if c.get(t)},
        }

    if ref:
        print("\n=== LLD test split, GROUND TRUTH (% of informative) ===")
        for t, v in sorted(ref.items(), key=lambda kv: -kv[1]):
            print(f"      {t:24s} {v:6.1f}%")
        report["_lld_test_ground_truth"] = ref

    print("\nRead it like this: 'Healthy' should be neutrophil/lymphocyte dominated with")
    print("blast% near 0. A flat distribution across groups, or one class taking most")
    print("cells everywhere, means the head is not transferring to this domain.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()