#!/usr/bin/env python3
"""Run stage eval and write JSON metrics (full pipeline)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import eval_attributes_gt  # noqa: E402
from utils.e2e_attributes import eval_attributes_e2e  # noqa: E402



def _write(out: Path, payload: dict) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload["written_at"] = datetime.now(timezone.utc).isoformat()
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote metrics -> {out}")
    return out


def _mean_accuracy(per_attr: dict[str, dict[str, float]]) -> float:
    accs = [v["accuracy"] for v in per_attr.values() if "accuracy" in v]
    return round(sum(accs) / len(accs), 4) if accs else 0.0

def _mean_f1(per_attr: dict[str, dict[str, float]]) -> float:
    vals = [v.get("f1", v.get("f1_macro")) for v in per_attr.values()]
    vals = [x for x in vals if x is not None]
    return round(sum(vals) / len(vals), 4) if vals else 0.0

def eval_detector(args) -> dict:
    det = YOLO(str(args.weights))
    metrics = det.val(
        data=str(args.config),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou_nms,
        max_det=args.max_det,
        device=args.device,
        batch=8,
    )
    box = getattr(metrics, "box", metrics)
    payload = {
        "stage": "detector",
        "split": args.split,
        "weights": str(args.weights.resolve()),
        "config": str(args.config.resolve()),
        "map50": float(getattr(box, "map50", 0) or 0),
        "map50_95": float(getattr(box, "map", 0) or 0),
        "precision": float(getattr(box, "mp", 0) or 0),
        "recall": float(getattr(box, "mr", 0) or 0),
    }
    if args.json_out:
        _write(args.json_out, payload)
    return payload


def eval_attribute_gt(args) -> dict:
    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    cfg = yaml.safe_load(args.config.read_text())
    manifest = Path(cfg["attr_manifest"])
    m = eval_attributes_gt(args.weights, manifest, args.split, device, args.batch)
    per_attr = {
        name: {k: float(v) for k, v in stats.items()}
        for name, stats in (m or {}).items()
    }
    payload = {
        "stage": "attribute_gt_crops",
        "split": args.split,
        "weights": str(args.weights.resolve()),
        "per_attribute": per_attr,
        "mean_accuracy": _mean_accuracy(per_attr),
        "mean_f1": _mean_f1(per_attr),
    }
    if args.json_out:
        _write(args.json_out, payload)
    return payload


def eval_joint(args) -> dict:
    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    cfg = yaml.safe_load(args.config.read_text())
    manifest = Path(cfg["attr_manifest"])

    det = YOLO(str(args.det_weights))
    det_metrics = det.val(
        data=str(args.config),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou_nms,
        max_det=args.max_det,
        device=args.device,
        batch=8,
    )
    box = getattr(det_metrics, "box", det_metrics)
    gt_attr = eval_attributes_gt(args.attr_weights, manifest, args.split, device, args.batch)
    e2e_metrics, e2e_stats, _ = eval_attributes_e2e(
        args.det_weights,
        args.attr_weights,
        manifest,
        args.split,
        device,
        conf=args.conf,
        iou_nms=args.iou_nms,
        max_det=args.max_det,
        imgsz_det=args.imgsz,
        attr_batch=args.batch,
        pad=args.pad,
        det_device=args.device,
        legacy=False,
    )
    per_attr_gt = {
        name: {k: float(v) for k, v in stats.items()}
        for name, stats in (gt_attr or {}).items()
    }
    per_attr_e2e = {
        name: {k: float(v) for k, v in stats.items()}
        for name, stats in (e2e_metrics or {}).items()
    }
    
    payload = {
        "stage": "stage1_joint",
        "split": args.split,
        "det_weights": str(args.det_weights.resolve()),
        "attr_weights": str(args.attr_weights.resolve()),
        "detection": {
            "map50": float(getattr(box, "map50", 0) or 0),
            "map50_95": float(getattr(box, "map", 0) or 0),
            "precision": float(getattr(box, "mp", 0) or 0),
            "recall": float(getattr(box, "mr", 0) or 0),
        },
        "attributes_gt_crops": {
            "per_attribute": per_attr_gt,
            "mean_accuracy": _mean_accuracy(per_attr_gt),
            "mean_f1": _mean_f1(per_attr_gt),
        },
        "attributes_e2e": {
            "per_attribute": per_attr_e2e,
            "mean_accuracy": _mean_accuracy(per_attr_e2e),
            "mean_f1": _mean_f1(per_attr_e2e),
            "n_matched": int(e2e_stats.get("n_matched", 0)),
            "n_gt_cells": int(e2e_stats.get("n_gt_cells", 0)),
            "mean_match_iou": float(e2e_stats.get("mean_match_iou", 0)),
        },
    }
    if args.json_out:
        _write(args.json_out, payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_det = sub.add_parser("detector")
    p_det.add_argument("--weights", type=Path, required=True)
    p_det.add_argument("--config", type=Path, default=ROOT / "configs" / "dataset.yaml")
    p_det.add_argument("--split", default="test")
    p_det.add_argument("--json-out", type=Path, required=True)
    p_det.add_argument("--device", default="0")
    p_det.add_argument("--imgsz", type=int, default=640)
    p_det.add_argument("--conf", type=float, default=0.001)
    p_det.add_argument("--iou-nms", type=float, default=0.6)
    p_det.add_argument("--max-det", type=int, default=300)

    p_attr = sub.add_parser("attribute")
    p_attr.add_argument("--weights", type=Path, required=True)
    p_attr.add_argument("--config", type=Path, default=ROOT / "configs" / "dataset.yaml")
    p_attr.add_argument("--split", default="test")
    p_attr.add_argument("--json-out", type=Path, required=True)
    p_attr.add_argument("--device", default="0")
    p_attr.add_argument("--batch", type=int, default=64)

    p_joint = sub.add_parser("joint")
    p_joint.add_argument("--det-weights", type=Path, required=True)
    p_joint.add_argument("--attr-weights", type=Path, required=True)
    p_joint.add_argument("--config", type=Path, default=ROOT / "configs" / "dataset.yaml")
    p_joint.add_argument("--split", default="test")
    p_joint.add_argument("--json-out", type=Path, required=True)
    p_joint.add_argument("--device", default="0")
    p_joint.add_argument("--imgsz", type=int, default=640)
    p_joint.add_argument("--conf", type=float, default=0.001)
    p_joint.add_argument("--iou-nms", type=float, default=0.6)
    p_joint.add_argument("--max-det", type=int, default=300)
    p_joint.add_argument("--batch", type=int, default=64)
    p_joint.add_argument("--pad", type=float, default=0.15)

    args = ap.parse_args()
    if args.cmd == "detector":
        eval_detector(args)
    elif args.cmd == "attribute":
        eval_attribute_gt(args)
    else:
        eval_joint(args)


if __name__ == "__main__":
    main()
