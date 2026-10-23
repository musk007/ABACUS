#!/usr/bin/env python3
"""Evaluate a trained cell-type classifier on GT crops and write JSON metrics.

Works for either arm — the backbone is read from the checkpoint, so the same
command evaluates DinoBloom and EfficientNet models.

    python eval_celltype.py \
      --weights runs/celltype/dinobloom/celltype/train/best_celltype.pt \
      --split test --json-out runs/eval/celltype_dinobloom_test.json --device 0

Pass --compare A.json B.json to print a side-by-side table afterwards.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
AGENTIC = ROOT.parent.parent
REPO_ROOT = AGENTIC.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT))

from data.celltype_dataset import CellTypeDataset  # noqa: E402
from train_cellType import (  # noqa: E402
    build_efficientnet,
    build_probe_head,
    embed_tensor_batch,
)


def load_model(weights: Path, device: torch.device):
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "dinobloom")
    class_names = ckpt["class_names"]
    n_classes = int(ckpt["n_classes"])
    imgsz = int(ckpt.get("imgsz", 224))

    embedder = None
    if arch == "dinobloom":
        from agentic_hematology.detection_agent_dinobloom import (  # noqa: E402
            DinoBloomEmbedder,
            resolve_dinobloom_weights,
        )

        variant = ckpt.get("dinobloom_variant", "l")
        backbone_path = ckpt.get("dinobloom_weights") or resolve_dinobloom_weights("auto", variant)
        embedder = DinoBloomEmbedder(
            weights_path=str(backbone_path), variant=variant, device=str(device)
        )
        for p in embedder.model.parameters():
            p.requires_grad = False
        embedder.model.eval()
        model = build_probe_head(int(ckpt["embed_dim"]), n_classes)
    else:
        model = build_efficientnet(n_classes, backbone=arch, pretrained=False)

    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, embedder, class_names, imgsz, ckpt


@torch.no_grad()
def run_eval(model, embedder, loader, device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ys, ps, pids = [], [], []
    for x, y, meta in tqdm(loader, desc="evaluating", leave=False):
        if embedder is not None:
            logits = model(embed_tensor_batch(embedder, x))
        else:
            logits = model(x.to(device))
        ys.append(y.numpy())
        ps.append(logits.argmax(1).cpu().numpy())
        pids.extend(meta["patient_id"])
    return np.concatenate(ys), np.concatenate(ps), pids


def differential_error(y_true, y_pred, pids, class_names) -> dict:
    """Per-patient differential % error — what actually feeds the leukemia classifier.

    Percentages are over informative cells (class 'None' excluded), matching
    aggregate() / cell_percentages_clinical.
    """
    none_idx = class_names.index("None") if "None" in class_names else -1
    per_class_abs_err: dict[str, list[float]] = {c: [] for c in class_names}
    for pid in sorted(set(pids)):
        mask = np.array([p == pid for p in pids])
        t, p = y_true[mask], y_pred[mask]
        t_inf, p_inf = t[t != none_idx], p[p != none_idx]
        if len(t_inf) == 0 or len(p_inf) == 0:
            continue
        for c, name in enumerate(class_names):
            if c == none_idx:
                continue
            gt_pct = 100.0 * (t_inf == c).sum() / len(t_inf)
            pr_pct = 100.0 * (p_inf == c).sum() / len(p_inf)
            per_class_abs_err[name].append(abs(pr_pct - gt_pct))
    return {
        name: round(float(np.mean(errs)), 3)
        for name, errs in per_class_abs_err.items()
        if errs
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "dataset.yaml")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--compare", type=Path, nargs="+", default=None,
                    help="Print a side-by-side table from existing result JSONs and exit.")
    ap.add_argument("--compare-out", type=Path, default=None,
                help="Write the comparison table to .csv or .json.")
    args = ap.parse_args()

    if args.compare:
        compare(args.compare, args.compare_out)
        return
    if not args.weights:
        raise SystemExit("--weights is required (or use --compare)")

    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    if args.manifest is not None:
        manifest = args.manifest
    else:
        cfg = yaml.safe_load(args.config.read_text())
        manifest = Path(cfg["celltype_manifest"])

    model, embedder, class_names, imgsz, ckpt = load_model(args.weights, device)
    ds = CellTypeDataset(manifest, args.split, imgsz=imgsz, target=ckpt["target"])
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    print(f"{ckpt['arch']} / target={ckpt['target']} / {len(ds)} cells in split={args.split}")

    y_true, y_pred, pids = run_eval(model, embedder, loader, device)
    present = sorted(set(y_true.tolist()))
    per_class = f1_score(y_true, y_pred, average=None, labels=present, zero_division=0)

    payload = {
        "stage": "celltype_gt_crops",
        "arch": ckpt["arch"],
        "target": ckpt["target"],
        "weights": str(args.weights.resolve()),
        "split": args.split,
        "n_cells": int(len(y_true)),
        "n_patients": len(set(pids)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=present, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=present, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float((y_true == y_pred).mean()),
        "per_class_f1": {class_names[c]: float(f) for c, f in zip(present, per_class)},
        "support": {class_names[c]: int((y_true == c).sum()) for c in present},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=present).tolist(),
        "labels_present": [class_names[c] for c in present],
        "differential_mae_pp": differential_error(y_true, y_pred, pids, class_names),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }

    print(f"\nmacro_f1={payload['macro_f1']:.4f}  weighted_f1={payload['weighted_f1']:.4f}  "
          f"bal_acc={payload['balanced_accuracy']:.4f}  acc={payload['accuracy']:.4f}")
    print(f"\n{'class':24s} {'F1':>7} {'n':>6} {'diff MAE pp':>12}")
    for name in payload["labels_present"]:
        mae = payload["differential_mae_pp"].get(name)
        mae_s = f"{mae:12.2f}" if mae is not None else " " * 12
        print(f"{name:24s} {payload['per_class_f1'][name]:7.4f} {payload['support'][name]:6d}{mae_s}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.json_out}")


def compare(paths: list[Path], out: Path | None = None) -> None:
    results = [json.loads(p.read_text()) for p in paths]
    names = [f"{r['arch']}/{r['target']}" for r in results]
    all_classes = sorted({c for r in results for c in r["per_class_f1"]})

    rows: list[dict] = []
    for key in ("macro_f1", "weighted_f1", "balanced_accuracy", "accuracy"):
        rows.append({"metric": key, "class": "", **{n: r[key] for n, r in zip(names, results)}})
    for c in all_classes:
        rows.append({"metric": "per_class_f1", "class": c,
                     **{n: r["per_class_f1"].get(c) for n, r in zip(names, results)}})
        rows.append({"metric": "differential_mae_pp", "class": c,
                     **{n: r.get("differential_mae_pp", {}).get(c) for n, r in zip(names, results)}})
        rows.append({"metric": "support", "class": c,
                     **{n: r["support"].get(c) for n, r in zip(names, results)}})

    for group, title in (("per_class_f1", "per-class F1"),
                         ("differential_mae_pp", "differential MAE (pp)")):
        pass  # printing handled below

    print(f"\n{'metric':24s}" + "".join(f"{n:>22s}" for n in names))
    for row in rows:
        if row["class"]:
            continue
        print(f"{row['metric']:24s}" + "".join(f"{row[n]:22.4f}" for n in names))

    for group, title, fmt in (("per_class_f1", "per-class F1", "{:22.4f}"),
                              ("differential_mae_pp", "differential MAE (pp)", "{:22.2f}")):
        print(f"\n{title:24s}" + "".join(f"{n:>22s}" for n in names))
        for row in rows:
            if row["metric"] != group:
                continue
            cells = "".join(
                fmt.format(row[n]) if row[n] is not None else f"{'-':>22s}" for n in names
            )
            print(f"{row['class']:24s}{cells}")

    if out is None:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".json":
        out.write_text(json.dumps({"arms": names, "rows": rows}, indent=2))
    else:
        import csv as _csv
        with out.open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["metric", "class"] + names)
            w.writeheader()
            w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()