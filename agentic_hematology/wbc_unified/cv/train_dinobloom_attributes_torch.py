#!/usr/bin/env python3
"""Train DinoBloom MLP attribute head on GT cell crops (frozen MarrLab backbone).

Mirrors ``train_attributes.py`` (EfficientNet) for a fair ablation:
  - same ``attr_manifest.csv``, same patient-grouped CV folds
  - same 6 binary morphology attributes + masked BCE, same 3-class Cell_Size head
  - same pos_weight / size_weight, computed on each run's training subset only
  - frozen foundation model + trainable MLP head only

Because the backbone is frozen, crop embeddings are computed once and cached
(``--cache-embeddings``), so the five CV folds cost seconds of MLP training each
instead of five full passes through DinoBloom.

Two modes:
  --fold K   CV run: hold out fold K's patients, record per-epoch val metrics.
  (no --fold) final run on all train patients, no held-out set, no selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
AGENTIC = ROOT.parent.parent
REPO_ROOT = AGENTIC.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT))

from data.splits import N_FOLDS, test_patients_for_fold  # noqa: E402
from data.cell_dataset import CellAttributeDataset, compute_pos_weights, compute_size_class_weights  # noqa: E402
from models.attribute_net import attribute_loss   # noqa: E402
from utils.labels import ATTR_NAMES, BINARY_ATTRS  # noqa: E402
from utils.metrics import attribute_metrics, cell_size_metrics  # noqa: E402

from agentic_hematology.detection_agent_dinobloom import (  # noqa: E402
    DINOBLOOM_VARIANTS,
    DinoBloomEmbedder,
    build_dinobloom_attribute_head,
    resolve_dinobloom_weights,
)


@torch.no_grad()
def embed_tensor_batch(embedder: DinoBloomEmbedder, x: torch.Tensor) -> torch.Tensor:
    """Embed pre-normalized crop tensors (same layout as CellAttributeDataset)."""
    x = x.to(embedder.device)
    model = embedder.model
    if hasattr(model, "forward_features"):
        feats = model.forward_features(x)
        if isinstance(feats, dict):
            if "x_norm_clstoken" in feats:
                emb = feats["x_norm_clstoken"]
            elif "cls_token" in feats:
                emb = feats["cls_token"]
            else:
                emb = next(iter(feats.values()))
        else:
            emb = feats
    else:
        emb = model(x)

    if emb.ndim > 2:
        emb = emb[:, 0] if emb.shape[1] > 1 else emb.mean(dim=1)
    return emb


@torch.no_grad()
def embed_split(
    embedder: DinoBloomEmbedder,
    manifest: Path,
    split: str,
    *,
    imgsz: int,
    batch: int,
    workers: int,
    val_patients,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Run the frozen backbone once over a split. Returns (emb, y, size, pids)."""
    ds = CellAttributeDataset(manifest, split, imgsz=imgsz, val_patients=val_patients)
    loader = DataLoader(
        ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True
    )
    embs, ys, szs, pids = [], [], [], []
    for x, y, s, meta in tqdm(loader, desc=f"embedding {split}", leave=False):
        embs.append(embed_tensor_batch(embedder, x).cpu())
        ys.append(y)
        szs.append(s)
        pids.extend(meta["patient_id"])
    if not embs:
        raise SystemExit(f"No cells for split={split}")
    return torch.cat(embs), torch.cat(ys), torch.cat(szs), pids


def build_or_load_cache(
    cache_path: Path | None,
    embedder_factory,
    manifest: Path,
    *,
    imgsz: int,
    batch: int,
    workers: int,
) -> dict:
    """Embed all train+test cells once (no fold filter) and cache to disk.

    Folds are applied afterwards by filtering on patient_id, so one cache
    serves every fold and the final run.
    """
    if cache_path is not None and cache_path.is_file():
        print(f"Loading cached embeddings from {cache_path}")
        cache = torch.load(cache_path, map_location="cpu")
        if cache.get("imgsz") != imgsz:
            raise SystemExit(
                f"Cache imgsz={cache.get('imgsz')} != requested {imgsz}. "
                f"Delete {cache_path} and re-run."
            )
        return cache

    embedder = embedder_factory()
    cache = {"imgsz": imgsz}
    for split in ("train", "test"):
        emb, y, sz, pids = embed_split(
            embedder, manifest, split,
            imgsz=imgsz, batch=batch, workers=workers, val_patients=None,
        )
        cache[split] = {"emb": emb, "y": y, "size": sz, "patient_ids": pids}
        print(f"  embedded {split}: {tuple(emb.shape)}")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
        print(f"Wrote embedding cache: {cache_path}")
    return cache


def subset(entry: dict, patient_ids: set[str] | None, *, keep: bool) -> TensorDataset:
    """Slice a cached split by patient. keep=True -> only those patients."""
    pids = entry["patient_ids"]
    if not patient_ids:
        idx = torch.arange(len(pids))
    else:
        mask = [(p in patient_ids) == keep for p in pids]
        idx = torch.tensor([i for i, m in enumerate(mask) if m], dtype=torch.long)
    return TensorDataset(entry["emb"][idx], entry["y"][idx], entry["size"][idx])


@torch.no_grad()
def evaluate(head, loader, device, nb: int) -> dict:
    head.eval()
    ys, ps, sz_t, sz_p = [], [], [], []
    for emb, y, s in loader:
        logits = head(emb.to(device))
        ys.append(y.numpy())
        ps.append(torch.sigmoid(logits[:, :nb]).cpu().numpy())
        sz_t.append(s.numpy())
        sz_p.append(logits[:, nb:].argmax(1).cpu().numpy())
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
    ap = argparse.ArgumentParser(description="Train DinoBloom attribute MLP on GT crops.")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "dataset.yaml")
    ap.add_argument("--manifest", type=Path, default=None, help="Override attr manifest CSV.")
    ap.add_argument("--dinobloom-weights", type=str, default="auto")
    ap.add_argument("--dinobloom-variant", choices=["s", "b", "l", "g"], default="l")
    ap.add_argument("--dinobloom-hub-dir", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--project", type=Path, default=ROOT / "runs" / "attribute_dinobloom")
    ap.add_argument("--name", type=str, default=None, help="Run dir name (default: train / cv_fold{K})")
    ap.add_argument(
        "--fold",
        type=int,
        default=None,
        help=f"CV fold to hold out (0..{N_FOLDS - 1}). Omit for the final full-data run.",
    )
    ap.add_argument(
        "--cache-embeddings",
        type=Path,
        default=None,
        help="Path to store/reuse frozen-backbone embeddings "
             "(default: <project>/embeddings_<variant>_<imgsz>.pt). Use --no-cache to disable.",
    )
    ap.add_argument("--no-cache", action="store_true", help="Recompute embeddings every run.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    val_patients = test_patients_for_fold(args.fold) if args.fold is not None else frozenset()
    is_cv = args.fold is not None
    run_name = args.name or (f"cv_fold{args.fold}" if is_cv else "train")

    if args.manifest is not None:
        manifest = Path(args.manifest)
        if not manifest.is_file() and not manifest.is_absolute():
            manifest = (ROOT / manifest).resolve()
    else:
        cfg = yaml.safe_load(args.config.read_text())
        manifest = Path(cfg["attr_manifest"])

    if not manifest.is_file():
        raise SystemExit(
            f"Missing manifest {manifest}. Run: python data/prepare_dataset.py "
            f"--data-root <LeukemiaDataset_Organized>"
        )

    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    save_dir = args.project / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    _, embed_dim = DINOBLOOM_VARIANTS[args.dinobloom_variant]
    weights_path = resolve_dinobloom_weights(args.dinobloom_weights, args.dinobloom_variant)

    def make_embedder() -> DinoBloomEmbedder:
        print(f"Loading frozen DinoBloom-{args.dinobloom_variant.upper()} from {weights_path}")
        emb = DinoBloomEmbedder(
            weights_path=weights_path,
            variant=args.dinobloom_variant,
            device=str(device),
            hub_dir=args.dinobloom_hub_dir,
        )
        for param in emb.model.parameters():
            param.requires_grad = False
        emb.model.eval()
        return emb

    cache_path = None
    if not args.no_cache:
        cache_path = args.cache_embeddings or (
            args.project / f"embeddings_{args.dinobloom_variant}_{args.imgsz}.pt"
        )
    cache = build_or_load_cache(
        cache_path, make_embedder, manifest,
        imgsz=args.imgsz, batch=args.batch, workers=args.workers,
    )

    if is_cv:
        tr, te = cache["train"], cache["test"]
        pool = {
            "emb": torch.cat([tr["emb"], te["emb"]]),
            "y": torch.cat([tr["y"], te["y"]]),
            "size": torch.cat([tr["size"], te["size"]]),
            "patient_ids": list(tr["patient_ids"]) + list(te["patient_ids"]),
        }
        train_set = subset(pool, set(val_patients), keep=False)
        val_set = subset(pool, set(val_patients), keep=True)
        val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False)
        print(f"CV fold {args.fold}: held-out patients {sorted(val_patients, key=int)}")
        print(f"  train cells: {len(train_set)}  test cells: {len(val_set)}")
    else:
        train_set = subset(cache["train"], None, keep=False)
        val_loader = None
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, drop_last=False)
    

    pos_weight = compute_pos_weights(manifest, "train", fold=args.fold).to(device)
    size_weight = compute_size_class_weights(manifest, "train", fold=args.fold).to(device)
    print(f"  pos_weight  {dict(zip(BINARY_ATTRS, [round(v, 3) for v in pos_weight.tolist()]))}")
    print(f"  size_weight {[round(v, 3) for v in size_weight.tolist()]}  (small, medium, large)")

    nb = len(BINARY_ATTRS)
    head = build_dinobloom_attribute_head(embed_dim, nb + 3).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    nb = len(BINARY_ATTRS)
    best_f1 = -1.0
    history: list[dict] = []

    for epoch in range(args.epochs):
        head.train()
        running = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False)
        for emb, y, s in pbar:
            emb, y, s = emb.to(device), y.to(device), s.to(device)
            opt.zero_grad(set_to_none=True)
            logits = head(emb)
            bin_logits, size_logits = logits[:, :nb], logits[:, nb:]
            loss = attribute_loss(
                bin_logits, y, size_logits, s,
                pos_weight=pos_weight, size_weight=size_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            opt.step()
            running += float(loss.item())
            n_batches += 1
        sched.step()

        metrics = evaluate(head, val_loader, device, nb) if val_loader else {}
        row = {
            "epoch": epoch + 1,
            "train_loss": running / max(n_batches, 1),
            "mean_attr_f1": mean_attr_f1(metrics),
            "per_attr": metrics,
        }
        history.append(row)
        if metrics:
            per_attr = "  ".join(
                f"{k}={v.get('f1', v.get('f1_macro', 0)):.3f}" for k, v in metrics.items()
            )
            print(f"epoch {epoch + 1}: loss={row['train_loss']:.4f}  "
                  f"mean_attr_f1={row['mean_attr_f1']:.4f}  |  {per_attr}")
        else:
            print(f"epoch {epoch + 1}: loss={row['train_loss']:.4f}")
    ckpt = {
        "model": head.state_dict(),
        "embed_dim": embed_dim,
        "num_attrs": nb + 3,
        "dinobloom_variant": args.dinobloom_variant,
        "dinobloom_weights": str(weights_path),
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
    out_name = f"fold{args.fold}_attr_dinobloom.pt" if is_cv else "best_attr_dinobloom.pt"
    torch.save(ckpt, save_dir / out_name)
    hist_name = f"history_fold{args.fold}.json" if is_cv else "history.json"
    (save_dir / hist_name).write_text(json.dumps(history, indent=2))
    print(f"Saved {save_dir / out_name} and {save_dir / hist_name}")


if __name__ == "__main__":
    main()
