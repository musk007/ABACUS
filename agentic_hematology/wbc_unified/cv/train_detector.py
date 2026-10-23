#!/usr/bin/env python3
"""Train YOLO11 (Ultralytics) WBC detector on prepared det_dataset."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def count_images(img_dir: Path) -> int:
    if not img_dir.exists():
        return 0
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    n = 0
    for p in img_dir.iterdir():
        if p.suffix.lower() in exts:
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "dataset.yaml")
    ap.add_argument("--model", type=str, default="yolo11m.pt", help="yolo11n/s/m/l or path to .pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="Total batch size (split across GPUs when multi-GPU)")
    ap.add_argument("--device", type=str, default=None, help="e.g. 0 or 0,1,2,3; default from NGPUS env")
    ap.add_argument("--ngpus", type=int, default=None, help="Override GPU count (else len(device) or NGPUS env)")
    ap.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("DET_WORKERS", "0")),
        help="Dataloader workers per GPU (0 = main process only, most stable on Slurm)",
    )
    ap.add_argument("--cache", type=str, default=os.environ.get("DET_CACHE", "false"), help="false|ram|disk")
    ap.add_argument("--project", type=str, default=str(ROOT / "runs" / "detector"))
    ap.add_argument("--name", type=str, default="train")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--val",
        action="store_true",
        default=os.environ.get("DET_VAL", "0").lower() in ("1", "true", "yes"),
        help="Run validation each epoch (DDP 4-GPU often hangs here; use evaluate.py after train)",
    )
    ap.add_argument(
        "--save-period",
        type=int,
        default=int(os.environ.get("DET_SAVE_PERIOD", "5")),
        help="Save checkpoint every N epochs (reduces large last.pt writes on shared FS)",
    )
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument(
        "--plots",
        action="store_true",
        default=os.environ.get("DET_PLOTS", "0").lower() in ("1", "true", "yes"),
    )
    args = ap.parse_args()

    ngpus = args.ngpus
    if ngpus is None and os.environ.get("NGPUS"):
        ngpus = int(os.environ["NGPUS"])
    if args.device is None:
        ngpus = ngpus if ngpus else 1
        args.device = ",".join(str(i) for i in range(ngpus))
    elif ngpus is None:
        ngpus = len([x for x in args.device.split(",") if x.strip()])
    ngpus = max(int(ngpus), 1)

    if args.workers > 0:
        max_total_workers = int(os.environ.get("MAX_DATALOADER_WORKERS", "16"))
        if args.workers * ngpus > max_total_workers:
            args.workers = max(1, max_total_workers // ngpus)
            print(f"Capped dataloader workers to {args.workers}/GPU ({args.workers * ngpus} total)")

    if ngpus > 1 and args.batch % ngpus != 0:
        raise SystemExit(f"DET batch={args.batch} must be divisible by ngpus={ngpus}")

    cfg = yaml.safe_load(args.config.read_text())
    data_yaml = args.config
    det_path = Path(cfg["path"])
    train_img = det_path / "images" / "train"
    n_train = count_images(train_img)
    per_gpu = args.batch // max(ngpus, 1)
    print(f"Detection data: {data_yaml}")
    print(f"  path={det_path}  train_images={n_train}")
    cache = False if args.cache.lower() in ("false", "0", "no", "") else args.cache
    save_dir = Path(args.project) / args.name
    last_pt = save_dir / "weights" / "last.pt"
    weights = args.model
    do_resume = False
    if args.resume:
        if last_pt.is_file():
            weights = str(last_pt)
            do_resume = True
            print(f"  resume: {last_pt} (epoch checkpoint)")
        else:
            print(f"  WARNING: --resume but missing {last_pt}; training from {args.model}")

    print(
        f"  device={args.device}  ngpus={ngpus}  batch={args.batch} ({per_gpu}/GPU)  "
        f"workers={args.workers}/GPU  cache={cache}  val={args.val}  save_period={args.save_period}"
    )
    if n_train == 0:
        raise SystemExit(
            f"No images in {train_img}. Run: python data/prepare_dataset.py\n"
            "Ensure DATA_ROOT images are visible on this node (NFS mount)."
        )

    model = YOLO(weights)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        resume=do_resume,
        patience=args.patience if args.val else 0,
        save=True,
        save_period=args.save_period,
        plots=args.plots,
        val=args.val,
        cache=cache,
        amp=True,
        single_cls=False,
        degrees=180.0,
        flipud=0.5,
        close_mosaic=10,
        deterministic=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Done. Best weights: {best}")


if __name__ == "__main__":
    main()
