#!/usr/bin/env python3
from __future__ import annotations
"""Train attribute classifier on GT cell crops (EfficientNet-B0)."""
"""
VAL_PATIENTS = {"2", "3", "13", "38",      # ALL  — 365 cells (20.0%)
                "17", "33", "37", "44"}    # AML  — 545 cells (20.0%)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.splits import N_FOLDS, test_patients_for_fold  
from data.cell_dataset import CellAttributeDataset, compute_pos_weights, compute_size_class_weights  # noqa: E402
from models.attribute_net import attribute_loss, build_attribute_model  # noqa: E402
from utils.labels import BINARY_ATTRS, ATTR_NAMES  # noqa: E402
from utils.metrics import attribute_metrics, cell_size_metrics  # noqa: E402

def init_distributed() -> tuple[int, int, int, bool]:
    """Return (rank, local_rank, world_size, is_distributed)."""
    if "RANK" not in os.environ:
        return 0, 0, 1, False
    import torch.distributed as dist

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, True


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed:
        import torch.distributed as dist

        dist.destroy_process_group()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps, sz_t, sz_p = [], [], [], []
    for x, y, s, _ in loader:
        bin_logits, size_logits = model(x.to(device))
        ys.append(y.numpy())
        ps.append(torch.sigmoid(bin_logits).cpu().numpy())
        sz_t.append(s.numpy())
        sz_p.append(size_logits.argmax(1).cpu().numpy())
    if not ys:
        return {}
    m = attribute_metrics(np.concatenate(ys), np.concatenate(ps), BINARY_ATTRS)
    size_m = cell_size_metrics(np.concatenate(sz_t), np.concatenate(sz_p))
    if size_m:
        m["Cell_Size"] = size_m
    return m


def mean_attr_f1(metrics: dict) -> float:
    if not metrics:
        return 0.0
    return sum(m.get("f1", m.get("f1_macro", 0.0)) for m in metrics.values()) / len(metrics)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "dataset.yaml")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64, help="Global batch size (split across GPUs under DDP)")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", type=str, default="0", help="Used only for single-GPU; DDP uses LOCAL_RANK")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--local-rank", type=int, default=-1, help="Set by torch.distributed.run")
    ap.add_argument("--backbone", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "resnet18"])
    ap.add_argument("--project", type=Path, default=ROOT / "runs" / "attribute")
    ap.add_argument("--name", type=str, default="train")
    ap.add_argument(
        "--fold",
        type=int,
        default=None,
        help=f"CV fold to hold out (0..{N_FOLDS - 1}). Omit for the final full-data run.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    val_patients = test_patients_for_fold(args.fold) if args.fold is not None else frozenset()
    is_cv = args.fold is not None
    run_name = args.name or (f"cv_fold{args.fold}" if is_cv else "train")

    rank, local_rank, world_size, is_distributed = init_distributed()
    is_main = rank == 0

    cfg = yaml.safe_load(args.config.read_text())
    manifest = Path(cfg["attr_manifest"])
    if not manifest.is_file():
        raise SystemExit(f"Missing manifest {manifest}. Run: python data/prepare_dataset.py")

    save_dir = args.project / args.name
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)

    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)

    if world_size > 1 and args.batch % world_size != 0:
        raise SystemExit(f"ATTR batch={args.batch} must be divisible by world_size={world_size}")
    batch_per_gpu = args.batch // world_size

    max_total_workers = int(os.environ.get("MAX_DATALOADER_WORKERS", "32"))
    if args.workers * world_size > max_total_workers:
        args.workers = max(2, max_total_workers // world_size)
        if is_main:
            print(f"Capped dataloader workers to {args.workers}/GPU ({args.workers * world_size} total)")

    train_ds = CellAttributeDataset(manifest, "train", imgsz=args.imgsz, fold=args.fold)
    val_loader = None
    if is_cv:
        val_ds = CellAttributeDataset(manifest, "test", imgsz=args.imgsz, fold=args.fold)
        val_loader = DataLoader(
            val_ds, batch_size=batch_per_gpu, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )
    if is_main:
        if is_cv:
            print(f"CV fold {args.fold}: held-out patients {sorted(val_patients, key=int)}")
            print(f"  train cells: {len(train_ds)}  val cells: {len(val_ds)}")
        else:
            print(f"FINAL run on all train patients: {len(train_ds)} cells, "
                  f"{args.epochs} epochs, no held-out selection")

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_per_gpu,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    # Weights come from this run's training subset only.
    pos_weight = compute_pos_weights(manifest, "train", fold=args.fold).to(device)
    size_weight = compute_size_class_weights(manifest, "train", fold=args.fold).to(device)
    if is_main:
        print(f"  pos_weight  {dict(zip(BINARY_ATTRS, [round(v, 3) for v in pos_weight.tolist()]))}")
        print(f"  size_weight {[round(v, 3) for v in size_weight.tolist()]}  (small, medium, large)")

    model = build_attribute_model(
        num_binary=len(BINARY_ATTRS), num_size=3, backbone=args.backbone, pretrained=True
    ).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history: list[dict] = []

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}", leave=False, disable=not is_main)
        for x, y, s, _ in pbar:
            x, y, s = x.to(device), y.to(device), s.to(device)
            opt.zero_grad(set_to_none=True)
            bin_logits, size_logits = model(x)
            loss = attribute_loss(
                bin_logits, y, size_logits, s,
                pos_weight=pos_weight, size_weight=size_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.item())
            n_batches += 1
        sched.step()

        eval_model = model.module if is_distributed else model
        metrics = evaluate(eval_model, val_loader, device) if (is_main and val_loader) else {}
        row = {
            "epoch": epoch + 1,
            "train_loss": running / max(n_batches, 1),
            "mean_attr_f1": mean_attr_f1(metrics),
            "per_attr": metrics,
        }
        if is_main:
            history.append(row)
            if metrics:
                per_attr = "  ".join(
                    f"{k}={v.get('f1', v.get('f1_macro', 0)):.3f}" for k, v in metrics.items()
                )
                print(f"epoch {epoch+1}: loss={row['train_loss']:.4f}  "
                      f"mean_attr_f1={row['mean_attr_f1']:.4f}  |  {per_attr}")
            else:
                print(f"epoch {epoch+1}: loss={row['train_loss']:.4f}")

    # Only the final epoch is saved: in CV the weights are throwaway (we want
    # the curve), and in the final run E* already came from cross-validation.
    if is_main:
        ckpt = {
            "model": (model.module if is_distributed else model).state_dict(),
            "backbone": args.backbone,
            "attr_names": ATTR_NAMES,
            "binary_attrs": BINARY_ATTRS,
            "num_size": 3,
            "imgsz": args.imgsz,
            "pos_weight": pos_weight.cpu(),
            "size_weight": size_weight.cpu(),
            "fold": args.fold,
            "val_patients": sorted(val_patients, key=int) if val_patients else [],
            "epochs_trained": args.epochs,
            "seed": args.seed,
            "metrics": history[-1]["per_attr"] if history else {},
        }
        out_name = "best_attr.pt" if not is_cv else f"fold{args.fold}_attr.pt"
        torch.save(ckpt, save_dir / out_name)
        hist_name = f"history_fold{args.fold}.json" if is_cv else "history.json"
        (save_dir / hist_name).write_text(json.dumps(history, indent=2))
        print(f"Saved {save_dir / out_name} and {save_dir / hist_name}")

    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
