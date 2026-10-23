#!/usr/bin/env python3
"""Evaluate DinoBloom + trained MLP attributes and write JSON metrics.

Mirrors ``scripts/pipeline_eval.py`` (EfficientNet) for a fair ablation:
  attribute — GT crop attribute metrics on the manifest split
  joint     — YOLO detection val + GT crops + YOLO→DinoBloom e2e attributes

Example:
  cd /home/roba.majzoub/agentic_hematology/wbc_unified/cv

  python scripts/pipeline_eval_dinobloom.py attribute \\
    --attr-weights runs/attribute_dinobloom/train/best_attr_dinobloom.pt \\
    --split test \\
    --json-out runs/eval/dinobloom_attribute_gt_test.json \\
    --device 0

  python scripts/pipeline_eval_dinobloom.py joint \\
    --det-weights runs/detector/train/weights/best.pt \\
    --attr-weights runs/attribute_dinobloom/train/best_attr_dinobloom.pt \\
    --split test \\
    --json-out runs/eval/dinobloom_joint_test.json \\
    --device 0
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import csv

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.dinobloom_attribute_eval import (  # noqa: E402
    eval_attributes_e2e_dinobloom,
    eval_attributes_gt_dinobloom,
)
from data.splits import load_folds  # noqa: E402

DEFAULT_ATTR_WEIGHTS = ROOT / "runs" / "attribute_dinobloom" / "train" / "best_attr_dinobloom.pt"
DEFAULT_DET_WEIGHTS = ROOT / "runs" / "detector" / "train" / "weights" / "best.pt"


def _write(out: Path, payload: dict) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload["written_at"] = datetime.now(timezone.utc).isoformat()
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote metrics -> {out}")
    return out


def _metrics_payload(metrics: dict) -> dict[str, dict[str, float]]:
    return {
        name: {k: float(v) for k, v in stats.items()}
        for name, stats in (metrics or {}).items()
    }


def _mean_accuracy(per_attr: dict[str, dict[str, float]]) -> float:
    accs = [v["accuracy"] for v in per_attr.values() if "accuracy" in v]
    return round(sum(accs) / len(accs), 4) if accs else 0.0

def _mean_f1(per_attr: dict[str, dict[str, float]]) -> float:
    vals = [v.get("f1", v.get("f1_macro")) for v in per_attr.values()]
    vals = [x for x in vals if x is not None]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _read_ckpt_meta(attr_weights: Path) -> dict:
    ckpt = torch.load(attr_weights, map_location="cpu", weights_only=False)
    return ckpt if isinstance(ckpt, dict) else {}

def _resolve_fold(args, manifest: Path) -> tuple[Path, str]:
    """With --fold: manifest copy where fold patients are 'test', all others 'train'."""
    if args.fold is None:
        return manifest, args.split
    test_pids = load_folds()[args.fold]
    out = args.json_out.parent / f"_{manifest.stem}_fold{args.fold}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open() as fin, out.open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            row["split"] = "test" if row["patient_id"] in test_pids else "train"
            writer.writerow(row)
    return out, "test"


def eval_attribute_gt(args) -> dict:
    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    cfg = yaml.safe_load(args.config.read_text())
    manifest = Path(cfg["attr_manifest"])
    manifest, split = _resolve_fold(args, manifest)
    ckpt = _read_ckpt_meta(args.attr_weights)

    metrics = eval_attributes_gt_dinobloom(
        args.attr_weights,
        manifest,
        split,
        device,
        args.batch,
        dinobloom_weights=args.dinobloom_weights,
        dinobloom_hub_dir=args.dinobloom_hub_dir,
    )
    per_attr = _metrics_payload(metrics)
    payload = {
        "stage": "dinobloom_attribute_gt_crops",
        "split": split,
        "fold": args.fold,
        "attr_weights": str(args.attr_weights.resolve()),
        "dinobloom_variant": ckpt.get("dinobloom_variant"),
        "dinobloom_weights": ckpt.get("dinobloom_weights"),
        "per_attribute": per_attr,
        "mean_accuracy": _mean_accuracy(per_attr),
        "mean_f1": _mean_f1(per_attr)
    }
    if args.json_out:
        _write(args.json_out, payload)
    return payload


def eval_joint(args) -> dict:
    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    cfg = yaml.safe_load(args.config.read_text())
    manifest = Path(cfg["attr_manifest"])
    manifest, split = _resolve_fold(args, manifest)
    det_yaml = ROOT / "configs" / f"dataset_fold{args.fold}.yaml" if args.fold is not None else args.config

    det = YOLO(str(args.det_weights))
    det_metrics = det.val(
        data=str(det_yaml),
        split="test" if args.fold is not None else args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou_nms,
        max_det=args.max_det,
        device=args.device,
        batch=8,
    )
    box = getattr(det_metrics, "box", det_metrics)

    gt_attr = eval_attributes_gt_dinobloom(
        args.attr_weights,
        manifest,
        split,
        device,
        args.batch,
        dinobloom_weights=args.dinobloom_weights,
        dinobloom_hub_dir=args.dinobloom_hub_dir,
    )
    e2e_metrics, e2e_stats, _ = eval_attributes_e2e_dinobloom(
        args.det_weights,
        args.attr_weights,
        manifest,
        split,
        device,
        conf=args.conf,
        iou_nms=args.iou_nms,
        max_det=args.max_det,
        imgsz_det=args.imgsz,
        attr_batch=args.batch,
        pad=args.pad,
        det_device=args.device,
        legacy=args.legacy_attr,
        dinobloom_weights=args.dinobloom_weights,
        dinobloom_hub_dir=args.dinobloom_hub_dir,
    )

    ckpt = _read_ckpt_meta(args.attr_weights)
    per_attr_gt = _metrics_payload(gt_attr)
    per_attr_e2e = _metrics_payload(e2e_metrics)

    payload = {
        "stage": "dinobloom_stage1_joint",
        "split": split,
        "fold": args.fold,
        "det_weights": str(args.det_weights.resolve()),
        "attr_weights": str(args.attr_weights.resolve()),
        "dinobloom_variant": ckpt.get("dinobloom_variant"),
        "dinobloom_weights": ckpt.get("dinobloom_weights"),
        "detection": {
            "map50": float(getattr(box, "map50", 0) or 0),
            "map50_95": float(getattr(box, "map", 0) or 0),
            "precision": float(getattr(box, "mp", 0) or 0),
            "recall": float(getattr(box, "mr", 0) or 0),
        },
        "attributes_gt_crops": {
            "per_attribute": per_attr_gt,
            "mean_accuracy": _mean_accuracy(per_attr_gt),
            "mean_f1": _mean_f1(per_attr_gt)
        },
        "attributes_e2e": {
            "per_attribute": per_attr_e2e,
            "mean_accuracy": _mean_accuracy(per_attr_e2e),
            "mean_f1": _mean_f1(per_attr_e2e),
            "n_matched": int(e2e_stats.get("n_matched", 0)),
            "n_gt_cells": int(e2e_stats.get("n_gt_cells", 0)),
            "n_skipped_no_det": int(e2e_stats.get("n_skipped_no_det", 0)),
            "mean_match_iou": float(e2e_stats.get("mean_match_iou", 0)),
        },
    }
    if args.json_out:
        _write(args.json_out, payload)
    return payload


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "dataset.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument(
        "--attr-weights",
        type=Path,
        default=DEFAULT_ATTR_WEIGHTS,
        help="Trained DinoBloom MLP checkpoint (best_attr_dinobloom.pt).",
    )
    parser.add_argument(
        "--dinobloom-weights",
        default="auto",
        help="Backbone weights path or 'auto' (read from MLP checkpoint).",
    )
    parser.add_argument("--dinobloom-hub-dir", default=None)
    parser.add_argument("--fold", type=int, default=None, help="Evaluate CV fold K's held-out patients")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate DinoBloom + MLP attribute head.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_attr = sub.add_parser("attribute", help="GT crop attribute metrics only.")
    _add_shared_args(p_attr)

    p_joint = sub.add_parser("joint", help="YOLO val + GT crops + e2e DinoBloom attributes.")
    _add_shared_args(p_joint)
    p_joint.add_argument("--det-weights", type=Path, default=DEFAULT_DET_WEIGHTS)
    p_joint.add_argument("--imgsz", type=int, default=640)
    p_joint.add_argument("--conf", type=float, default=0.001)
    p_joint.add_argument("--iou-nms", type=float, default=0.6)
    p_joint.add_argument("--max-det", type=int, default=300)
    p_joint.add_argument("--pad", type=float, default=0.15)
    p_joint.add_argument(
        "--legacy-attr",
        action="store_true",
        help="Use val.py cell filter + sklearn metrics for e2e scoring.",
    )

    args = ap.parse_args()
    if not args.attr_weights.is_file():
        raise SystemExit(
            f"DinoBloom MLP checkpoint not found: {args.attr_weights}\n"
            "Train with: python train_dinobloom_attributes_torch.py"
        )

    if args.cmd == "attribute":
        eval_attribute_gt(args)
    else:
        if not args.det_weights.is_file():
            raise SystemExit(f"YOLO weights not found: {args.det_weights}")
        eval_joint(args)


if __name__ == "__main__":
    main()
