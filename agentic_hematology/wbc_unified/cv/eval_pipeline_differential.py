#!/usr/bin/env python3
"""Compare cell-typing strategies at the level that actually matters downstream:
the per-patient differential percentages that feed the leukemia classifier.

Two modes over the SAME detector boxes, so the comparison isolates the cell
typing rather than the localisation:

  --mode joint       cell type = the YOLO class head (trained on 14 classes)
  --mode decoupled   cell type = crop classifier (DinoBloom or EfficientNet),
                     YOLO used for boxes only

Reported per patient and averaged over the split:
  - differential % per cell type (over informative cells, 'None' excluded),
    predicted vs ground truth, and the mean absolute error in percentage points
  - blast-pool % error, since HybridClassifier keys on the 20% threshold
  - detection count vs GT cell count

    python eval_pipeline_differential.py --mode joint \\
      --det-weights runs/detector/train/weights/best.pt \\
      --split test --json-out runs/eval/diff_joint_test.json --device 0

    python eval_pipeline_differential.py --mode decoupled \\
      --det-weights runs/detector/train/weights/best.pt \\
      --celltype-weights runs/celltype/dinobloom/celltype/train/best_celltype.pt \\
      --split test --json-out runs/eval/diff_decoupled_dino_test.json --device 0

    python eval_pipeline_differential.py --compare a.json b.json c.json \\
      --compare-out runs/eval/differential_comparison.csv
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
AGENTIC = ROOT.parent.parent
REPO_ROOT = AGENTIC.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT))

from data.celltype_dataset import CLASS_NAMES  # noqa: E402
from data.cell_dataset import load_manifest  # noqa: E402
from utils.e2e_attributes import crops_to_tensor, det_crops_from_xyxy  # noqa: E402

# Matches BLAST_CLASSES in agentic_hematology/aggregator.py
BLAST_CLASSES = {"Myeloblast", "Lymphoblast", "Monoblast", "Abnormal promyelocyte"}
NONE_CLASS = "None"


# ---------------------------------------------------------------------------

def gt_differentials(manifest: Path, split: str) -> dict[str, dict[str, float]]:
    """Per-patient GT differential % over informative cells."""
    rows = load_manifest(manifest, split)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        name = CLASS_NAMES[int(r["class_id"])]
        counts[r["patient_id"]][name] += 1
    out: dict[str, dict[str, float]] = {}
    for pid, c in counts.items():
        total = sum(v for k, v in c.items() if k != NONE_CLASS)
        if total == 0:
            continue
        out[pid] = {
            k: round(100.0 * v / total, 3) for k, v in c.items() if k != NONE_CLASS
        }
        out[pid]["_n_informative"] = total
        out[pid]["_n_total"] = sum(c.values())
    return out


def images_by_patient(manifest: Path, split: str) -> dict[str, list[str]]:
    rows = load_manifest(manifest, split)
    by_patient: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_patient[r["patient_id"]].add(r["image"])
    return {pid: sorted(paths) for pid, paths in by_patient.items()}


# ---------------------------------------------------------------------------

def load_celltype_classifier(weights: Path, device: torch.device):
    from eval_cellType import load_model

    model, embedder, class_names, imgsz, ckpt = load_model(weights, device)
    if ckpt["target"] != "celltype":
        raise SystemExit(
            f"This comparison needs a 14-class model; checkpoint target is {ckpt['target']!r}."
        )
    return model, embedder, class_names, imgsz, ckpt


@torch.no_grad()
def classify_crops(model, embedder, crops, device, imgsz, class_names, batch=64) -> list[str]:
    if not crops:
        return []
    out: list[str] = []
    for start in range(0, len(crops), batch):
        x = crops_to_tensor(crops[start : start + batch], imgsz)
        if embedder is not None:
            from train_cellType import embed_tensor_batch

            logits = model(embed_tensor_batch(embedder, x))
        else:
            logits = model(x.to(device))
        out.extend(class_names[i] for i in logits.argmax(1).cpu().numpy())
    return out


@torch.no_grad()
def predicted_differentials(
    det_weights: Path,
    manifest: Path,
    split: str,
    device: torch.device,
    *,
    mode: str,
    celltype_weights: Path | None,
    conf: float,
    iou_nms: float,
    max_det: int,
    imgsz_det: int,
    pad: float,
    det_device: str,
    batch: int,
) -> dict[str, dict[str, float]]:
    det = YOLO(str(det_weights))
    det_names = det.names  # {id: name}

    model = embedder = class_names = None
    imgsz_cls = 224
    if mode == "decoupled":
        if celltype_weights is None:
            raise SystemExit("--celltype-weights is required for --mode decoupled")
        model, embedder, class_names, imgsz_cls, _ = load_celltype_classifier(
            celltype_weights, device
        )

    by_patient = images_by_patient(manifest, split)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for pid, image_paths in tqdm(sorted(by_patient.items()), desc=f"{mode} inference"):
        for img_path_str in image_paths:
            img_path = Path(img_path_str)
            if not img_path.is_file():
                continue
            res = det.predict(
                source=str(img_path),
                conf=conf,
                iou=iou_nms,
                imgsz=imgsz_det,
                max_det=max_det,
                device=det_device,
                verbose=False,
            )[0]
            if res.boxes is None or not len(res.boxes):
                continue

            if mode == "joint":
                for cls_id in res.boxes.cls.cpu().numpy().astype(int):
                    counts[pid][str(det_names[int(cls_id)])] += 1
            else:
                xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
                image = Image.open(img_path).convert("RGB")
                crops = det_crops_from_xyxy(image, xyxy, pad=pad)
                for name in classify_crops(
                    model, embedder, crops, device, imgsz_cls, class_names, batch=batch
                ):
                    counts[pid][name] += 1

    out: dict[str, dict[str, float]] = {}
    for pid, c in counts.items():
        total = sum(v for k, v in c.items() if k != NONE_CLASS)
        if total == 0:
            out[pid] = {"_n_informative": 0, "_n_total": sum(c.values())}
            continue
        out[pid] = {k: round(100.0 * v / total, 3) for k, v in c.items() if k != NONE_CLASS}
        out[pid]["_n_informative"] = total
        out[pid]["_n_total"] = sum(c.values())
    return out


# ---------------------------------------------------------------------------

def blast_pct(diff: dict[str, float]) -> float:
    return round(sum(diff.get(c, 0.0) for c in BLAST_CLASSES), 3)


def summarise(gt: dict, pred: dict) -> dict:
    patients = sorted(set(gt) & set(pred), key=lambda x: (not x.isdigit(), x))
    cell_types = [c for c in CLASS_NAMES if c != NONE_CLASS]

    per_class_err: dict[str, list[float]] = {c: [] for c in cell_types}
    blast_err: list[float] = []
    per_patient: list[dict] = []

    for pid in patients:
        g, p = gt[pid], pred[pid]
        for c in cell_types:
            per_class_err[c].append(abs(p.get(c, 0.0) - g.get(c, 0.0)))
        gb, pb = blast_pct(g), blast_pct(p)
        blast_err.append(abs(pb - gb))
        per_patient.append({
            "patient_id": pid,
            "gt_blast_pct": gb,
            "pred_blast_pct": pb,
            "blast_abs_err_pp": round(abs(pb - gb), 3),
            "gt_blast_threshold_met": gb >= 20.0,
            "pred_blast_threshold_met": pb >= 20.0,
            "gt_n_informative": g.get("_n_informative", 0),
            "pred_n_informative": p.get("_n_informative", 0),
        })

    threshold_agree = sum(
        1 for r in per_patient
        if r["gt_blast_threshold_met"] == r["pred_blast_threshold_met"]
    )
    return {
        "n_patients": len(patients),
        "differential_mae_pp": {
            c: round(float(np.mean(v)), 3) for c, v in per_class_err.items() if v
        },
        "mean_differential_mae_pp": round(
            float(np.mean([np.mean(v) for v in per_class_err.values() if v])), 3
        ),
        "blast_pct_mae_pp": round(float(np.mean(blast_err)), 3) if blast_err else 0.0,
        "blast_threshold_agreement": f"{threshold_agree}/{len(per_patient)}",
        "per_patient": per_patient,
    }


def compare(paths: list[Path], out: Path | None) -> None:
    results = [json.loads(p.read_text()) for p in paths]
    names = [r["label"] for r in results]
    rows: list[dict] = []

    for key in ("mean_differential_mae_pp", "blast_pct_mae_pp"):
        rows.append({"metric": key, "class": "",
                     **{n: r["summary"][key] for n, r in zip(names, results)}})
    rows.append({"metric": "blast_threshold_agreement", "class": "",
                 **{n: r["summary"]["blast_threshold_agreement"] for n, r in zip(names, results)}})
    for c in [c for c in CLASS_NAMES if c != NONE_CLASS]:
        rows.append({"metric": "differential_mae_pp", "class": c,
                     **{n: r["summary"]["differential_mae_pp"].get(c)
                        for n, r in zip(names, results)}})

    print(f"\n{'metric':28s}" + "".join(f"{n:>24s}" for n in names))
    for row in rows:
        label = row["class"] or row["metric"]
        vals = "".join(
            (f"{row[n]:24.3f}" if isinstance(row[n], (int, float)) else f"{str(row[n]):>24s}")
            for n in names
        )
        print(f"{label:28s}{vals}")

    if out is None:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".json":
        out.write_text(json.dumps({"arms": names, "rows": rows}, indent=2))
    else:
        with out.open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["metric", "class"] + names)
            w.writeheader()
            w.writerows(rows)
    print(f"\nWrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("joint", "decoupled"))
    ap.add_argument("--det-weights", type=Path)
    ap.add_argument("--celltype-weights", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "dataset.yaml")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--label", default=None, help="Name for this arm in comparisons.")
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Detection confidence; 0.25 matches aggregate()'s default.")
    ap.add_argument("--iou-nms", type=float, default=0.6)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--compare", type=Path, nargs="+", default=None)
    ap.add_argument("--compare-out", type=Path, default=None)
    args = ap.parse_args()

    if args.compare:
        compare(args.compare, args.compare_out)
        return
    if not args.mode or not args.det_weights:
        raise SystemExit("--mode and --det-weights are required (or use --compare)")

    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    if args.manifest is not None:
        manifest = args.manifest
    else:
        cfg = yaml.safe_load(args.config.read_text())
        manifest = Path(cfg["celltype_manifest"])

    gt = gt_differentials(manifest, args.split)
    pred = predicted_differentials(
        args.det_weights, manifest, args.split, device,
        mode=args.mode,
        celltype_weights=args.celltype_weights,
        conf=args.conf,
        iou_nms=args.iou_nms,
        max_det=args.max_det,
        imgsz_det=args.imgsz,
        pad=args.pad,
        det_device=args.device,
        batch=args.batch,
    )
    summary = summarise(gt, pred)

    label = args.label or (
        args.mode if args.mode == "joint"
        else f"decoupled/{Path(args.celltype_weights).parent.parent.parent.name}"
    )
    print(f"\n=== {label} ({summary['n_patients']} patients) ===")
    print(f"mean differential MAE : {summary['mean_differential_mae_pp']:.3f} pp")
    print(f"blast %% MAE           : {summary['blast_pct_mae_pp']:.3f} pp")
    print(f"20%% threshold agree   : {summary['blast_threshold_agreement']}")
    print(f"\n{'cell type':24s} {'MAE pp':>9}")
    for c, v in sorted(summary["differential_mae_pp"].items(), key=lambda kv: -kv[1]):
        print(f"{c:24s} {v:9.2f}")

    print(f"\n{'patient':10s} {'GT blast%':>10} {'pred blast%':>12} {'err':>7}")
    for r in summary["per_patient"]:
        print(f"{r['patient_id']:10s} {r['gt_blast_pct']:10.1f} "
              f"{r['pred_blast_pct']:12.1f} {r['blast_abs_err_pp']:7.1f}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "label": label,
            "mode": args.mode,
            "split": args.split,
            "det_weights": str(args.det_weights.resolve()),
            "celltype_weights": str(args.celltype_weights.resolve()) if args.celltype_weights else None,
            "conf": args.conf,
            "summary": summary,
            "gt_differentials": gt,
            "pred_differentials": pred,
            "written_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()