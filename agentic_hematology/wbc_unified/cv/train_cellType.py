#!/usr/bin/env python3
"""Train a cell-type classifier on GT crops — the classification half of the
decoupled pipeline (YOLO localises class-agnostically, this head types the cell).

Two backbones, mirroring the attribute-head ablation:

  --backbone dinobloom     frozen MarrLab DinoBloom + MLP probe. Embeddings are
                           computed once and cached, so folds cost seconds.
  --backbone efficientnet  EfficientNet-B0 finetuned end-to-end.

Targets:
  --target celltype   14 LLD classes (default)
  --target group      6 coarse groups (None + the CLINICAL_GROUPS pooling).
                      Worth trying: Basophil has 16 training cells and
                      Abnormal promyelocyte 115, so several fine classes are
                      barely learnable on a patient-grouped split.

Modes:
  --fold K    CV run over the patient folds; writes history_fold{K}.json
  (no --fold) final run on all train patients, no held-out set
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
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchvision import models
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
AGENTIC = ROOT.parent.parent
REPO_ROOT = AGENTIC.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT))

from data.celltype_dataset import (  # noqa: E402
    CellTypeDataset,
    compute_class_weights,
    sample_weights_for_balancing,
    target_names,
)
from data.splits import N_FOLDS, test_patients_for_fold  # noqa: E402




def build_probe_head(embed_dim: int, n_classes: int, hidden: int = 512, dropout: float = 0.3) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, n_classes),
    )

def build_efficientnet(n_classes: int, backbone: str = "efficientnet_b0", pretrained: bool = True) -> nn.Module:
    weights = "DEFAULT" if pretrained else None
    if backbone in ("efficientnet", "efficientnet_b0"):
        net = models.efficientnet_b0(weights=weights)
        feat_dim = net.classifier[1].in_features
        net.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim, n_classes))
    elif backbone == "resnet18":
        net = models.resnet18(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, n_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return net
# ---------------------------------------------------------------------------
# DinoBloom embedding cache
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_tensor_batch(embedder, x: torch.Tensor) -> torch.Tensor:
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


def embed_split(embedder, manifest, split, *, imgsz, batch, workers, target):
    ds = CellTypeDataset(manifest, split, imgsz=imgsz, target=target)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    embs, ys, pids, doms = [], [], [], []
    for x, y, meta in tqdm(loader, desc=f"embedding {split}", leave=False):
        embs.append(embed_tensor_batch(embedder, x).cpu())
        ys.append(y)
        pids.extend(meta["patient_id"])
        doms.extend(meta.get("domain", ["?"] * len(y)))
    if not embs:
        raise SystemExit(f"No cells for split={split}")
    return torch.cat(embs), torch.cat(ys), pids, doms


def build_or_load_cache(cache_path, embedder_factory, manifest, *, imgsz, batch, workers, target):
    if cache_path is not None and cache_path.is_file():
        print(f"Loading cached embeddings from {cache_path}")
        cache = torch.load(cache_path, map_location="cpu")
        if cache.get("imgsz") != imgsz or cache.get("target") != target:
            raise SystemExit(
                f"Cache was built with imgsz={cache.get('imgsz')} target={cache.get('target')!r}; "
                f"requested imgsz={imgsz} target={target!r}. Delete {cache_path} and re-run."
            )
        return cache

    embedder = embedder_factory()
    cache = {"imgsz": imgsz, "target": target}
    for split in ("train", "test"):
        emb, y, pids, doms = embed_split(
            embedder, manifest, split, imgsz=imgsz, batch=batch, workers=workers, target=target
        )
        cache[split] = {"emb": emb, "y": y, "patient_ids": pids, "domains": doms}
        print(f"  embedded {split}: {tuple(emb.shape)}")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
        print(f"Wrote embedding cache: {cache_path}")
    return cache


def cache_subset(entry: dict, patient_ids: set[str], *, keep: bool, domain: str | None = None):
    pids = entry["patient_ids"]
    doms = entry.get("domains", ["?"] * len(pids))
    sel = []
    for i, p in enumerate(pids):
        if patient_ids and (p in patient_ids) != keep:
            continue
        if domain and doms[i] != domain:
            continue
        sel.append(i)
    idx = torch.tensor(sel, dtype=torch.long)
    return TensorDataset(entry["emb"][idx], entry["y"][idx]), entry["y"][idx].numpy()

# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(forward_fn, loader, class_names) -> dict:
    ys, ps = [], []
    for batch in loader:
        x, y = batch[0], batch[1]
        logits = forward_fn(x)
        ys.append(y.numpy())
        ps.append(logits.argmax(1).cpu().numpy())
    if not ys:
        return {}
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    present = sorted(set(y_true.tolist()))
    per_class = f1_score(y_true, y_pred, average=None, labels=present, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=present, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=present, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class_f1": {
            class_names[c]: float(f)
            for c, f in zip(
                present,per_class)},
        "support": {class_names[c]: int((y_true == c).sum()) for c in present},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=present).tolist(),
        "labels_present": [class_names[c] for c in present],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a cell-type classifier on GT crops.")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "dataset.yaml")
    ap.add_argument("--manifest", type=Path, default=None, help="Override celltype manifest CSV.")
    ap.add_argument("--target", choices=("celltype", "group"), default="celltype")
    ap.add_argument("--backbone", choices=("dinobloom", "efficientnet", "resnet18"), default="dinobloom")
    # DinoBloom-only
    ap.add_argument("--dinobloom-weights", type=str, default="auto")
    ap.add_argument("--dinobloom-variant", choices=["s", "b", "l", "g"], default="l")
    ap.add_argument("--dinobloom-hub-dir", type=str, default=None)
    ap.add_argument("--cache-embeddings", type=Path, default=None)
    ap.add_argument("--no-cache", action="store_true")
    # shared
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--lr", type=float, default=None,
                    help="Default 1e-3 for dinobloom probe, 3e-4 for finetuning.")
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--project", type=Path, default=ROOT / "runs" / "celltype" / "DINO_multiCamera")
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--fold", type=int, default=None,
                    help=f"CV fold to hold out (0..{N_FOLDS - 1}). Omit for the final run.")
    ap.add_argument("--balance", choices=("sampler", "loss", "both", "none"), default="sampler")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-domain", default=None, help="C1, C2, or omit for both")
    ap.add_argument("--test-domain", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.lr is None:
        args.lr = 1e-3 if args.backbone == "dinobloom" else 3e-4

    val_patients = test_patients_for_fold(args.fold) if args.fold is not None else frozenset()
    is_cv = args.fold is not None
    run_name = args.name or (f"cv_fold{args.fold}" if is_cv else "train")

    if args.manifest is not None:
        manifest = Path(args.manifest)
    else:
        cfg = yaml.safe_load(args.config.read_text())
        if "celltype_manifest" not in cfg:
            raise SystemExit(
                "dataset.yaml has no 'celltype_manifest'. Re-run data/prepare_dataset.py."
            )
        manifest = Path(cfg["celltype_manifest"])
    if not manifest.is_file():
        raise SystemExit(f"Missing manifest {manifest}. Run: python data/prepare_dataset.py")

    class_names = target_names(args.target)
    n_classes = len(class_names)

    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    dom_tag = f"_{args.train_domain or 'all'}2{args.test_domain or 'all'}"
    save_dir = args.project / f"{args.backbone}_{args.target}{dom_tag}_{run_name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- data + model, per backbone -------------------------------------
    if args.backbone == "dinobloom":
        from agentic_hematology.detection_agent_dinobloom import (  # noqa: E402
            DINOBLOOM_VARIANTS,
            DinoBloomEmbedder,
            resolve_dinobloom_weights,
        )

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
            for p in emb.model.parameters():
                p.requires_grad = False
            emb.model.eval()
            return emb

        cache_path = None
        if not args.no_cache:
            cache_path = args.cache_embeddings or (
                args.project / f"embeddings_{args.dinobloom_variant}_{args.imgsz}_{args.target}_{manifest.stem}.pt"
            )
        cache = build_or_load_cache(
            cache_path, make_embedder, manifest,
            imgsz=args.imgsz, batch=args.batch, workers=args.workers, target=args.target,
        )

        if is_cv:
            # folds span both manifest splits: merge them, then split by fold patients
            tr, te = cache["train"], cache["test"]
            pool = {
                "emb": torch.cat([tr["emb"], te["emb"]]),
                "y": torch.cat([tr["y"], te["y"]]),
                "patient_ids": list(tr["patient_ids"]) + list(te["patient_ids"]),
                "domains": list(tr.get("domains", [])) + list(te.get("domains", [])),
            }
            train_set, train_labels = cache_subset(pool, set(val_patients), keep=False, domain=args.train_domain)
            val_set, _ = cache_subset(pool, set(val_patients), keep=True, domain=args.test_domain)
        else:
            train_set, train_labels = cache_subset(cache["train"], set(), keep=False, domain=args.train_domain)
            val_set = None

        model = build_probe_head(embed_dim, n_classes).to(device)
        loader_workers = 0  # tensors already in memory
    else:

        train_ds = CellTypeDataset(
            manifest, "train", imgsz=args.imgsz, target=args.target,
            domain=args.train_domain, fold=args.fold,
        )
        train_labels = train_ds.labels
        train_set = train_ds

        val_set = None
        if is_cv:
            val_set = CellTypeDataset(
                manifest, "val", imgsz=args.imgsz, val_patients=val_patients, target=args.target, domain=args.test_domain
            )
        model = build_efficientnet(n_classes, backbone=args.backbone).to(device)
        embed_dim = None
        weights_path = None
        loader_workers = args.workers

    def forward_fn(x):
        return model(x.to(device))

    # ---- loaders ---------------------------------------------------------
    if args.balance in ("sampler", "both"):
        sw = sample_weights_for_balancing(train_labels, n_classes)
        sampler = WeightedRandomSampler(sw, num_samples=len(train_labels), replacement=True)
        train_loader = DataLoader(
            train_set, batch_size=args.batch, sampler=sampler, num_workers=loader_workers
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=args.batch, shuffle=True, num_workers=loader_workers
        )
    val_loader = (
        DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=loader_workers)
        if val_set is not None
        else None
    )

    if is_cv:
        print(f"CV fold {args.fold}: held-out patients {sorted(val_patients, key=int)}")
        print(f"  train cells: {len(train_set)}  val cells: {len(val_set)}")
    else:
        print(f"FINAL run on all train patients: {len(train_set)} cells, {args.epochs} epochs")

    counts = np.bincount(train_labels, minlength=n_classes)
    print("  train class counts: " + "  ".join(
        f"{class_names[i]}={counts[i]}" for i in range(n_classes) if counts[i]
    ))
    empty = [class_names[i] for i in range(n_classes) if counts[i] == 0]
    if empty:
        print(f"  WARNING: no training cells for {empty} — these can never be predicted.")

    class_weight = None
    if args.balance in ("loss", "both"):
        class_weight = compute_class_weights(
            manifest, "train", fold=args.fold, target=args.target,
            domain=args.train_domain
        ).to(device)
        print(f"  class_weight: {[round(v, 2) for v in class_weight.tolist()]}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=0.05)

    history: list[dict] = []
    for epoch in range(args.epochs):
        model.train()
        running, n_batches = 0.0, 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}", leave=False):
            x, y = batch[0], batch[1]
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.item())
            n_batches += 1
        sched.step()

        model.eval()
        metrics = evaluate(forward_fn, val_loader, class_names) if val_loader else {}
        row = {
            "epoch": epoch + 1,
            "train_loss": running / max(n_batches, 1),
            "mean_attr_f1": metrics.get("macro_f1", 0.0),  # key name kept for select_epochs.py
            "macro_f1": metrics.get("macro_f1", 0.0),
            "balanced_accuracy": metrics.get("balanced_accuracy", 0.0),
            "per_attr": {},
            "metrics": metrics,
        }
        history.append(row)
        if metrics:
            print(f"epoch {epoch+1}: loss={row['train_loss']:.4f}  "
                  f"macro_f1={metrics['macro_f1']:.4f}  bal_acc={metrics['balanced_accuracy']:.4f}")
        else:
            print(f"epoch {epoch+1}: loss={row['train_loss']:.4f}")

    if history and history[-1]["metrics"]:
        print("\nFinal per-class F1 (n = val support):")
        last = history[-1]["metrics"]
        for name, f1 in last["per_class_f1"].items():
            print(f"  {name:24s} {f1:.4f}   (n={last['support'][name]})")

    ckpt = {
        "model": model.state_dict(),
        "arch": args.backbone,
        "embed_dim": embed_dim,
        "n_classes": n_classes,
        "class_names": class_names,
        "target": args.target,
        "dinobloom_variant": args.dinobloom_variant if args.backbone == "dinobloom" else None,
        "dinobloom_weights": str(weights_path) if weights_path else None,
        "imgsz": args.imgsz,
        "balance": args.balance,
        "fold": args.fold,
        "val_patients": sorted(val_patients, key=int) if val_patients else [],
        "epochs_trained": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "metrics": history[-1]["metrics"] if history else {},
    }
    out_name = f"fold{args.fold}_celltype.pt" if is_cv else "best_celltype.pt"
    torch.save(ckpt, save_dir / out_name)
    hist_name = f"history_fold{args.fold}.json" if is_cv else "history.json"
    (save_dir / hist_name).write_text(json.dumps(history, indent=2))
    print(f"\nSaved {save_dir / out_name} and {save_dir / hist_name}")


if __name__ == "__main__":
    main()