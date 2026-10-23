#!/usr/bin/env python3
"""Compare C2-only vs C1+C2 DinoBloom attribute probes on identical held-out cells.

For each fold k, both fold-k probes are applied to the embeddings of fold k's
held-out patients (from the C1+C2 embedding cache), and predictions are pooled
over folds. Metrics are reported separately for C2 cells, C1 cells and both,
using the same metric functions as training (utils.metrics).

Run from wbc_unified/cv:
    python compare_attr_probes.py \
      --cache runs/attribute_dinobloom_c1c2/embeddings_l_224.pt \
      --manifest generated_C1C2/attr_manifest.csv \
      --probes C2=runs/attribute_dinobloom/folds C1C2=runs/attribute_dinobloom_c1c2/folds \
      --out runs/attribute_probe_comparison.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent.parent.parent))

from data.splits import load_folds  # noqa: E402
from utils.labels import BINARY_ATTRS  # noqa: E402
from utils.metrics import attribute_metrics, cell_size_metrics  # noqa: E402
from agentic_hematology.detection_agent_dinobloom import build_dinobloom_attribute_head  # noqa: E402


def domains_in_cache_order(manifest: Path, split: str) -> list[str]:
    """The cache embeds rows of one split in manifest order; recover each row's camera."""
    with manifest.open() as f:
        return [r.get("domain", "?") for r in csv.DictReader(f) if r["split"] == split]


def load_pool(cache_path: Path, manifest: Path) -> dict:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    pool = {"emb": [], "y": [], "size": [], "pid": [], "dom": []}
    for split in ("train", "test"):
        e = cache[split]
        doms = domains_in_cache_order(manifest, split)
        if len(doms) != len(e["patient_ids"]):
            raise SystemExit(f"{split}: {len(doms)} manifest rows vs {len(e['patient_ids'])} cached cells; "
                             "cache and manifest do not match")
        pool["emb"].append(e["emb"]); pool["y"].append(e["y"]); pool["size"].append(e["size"])
        pool["pid"] += [str(p) for p in e["patient_ids"]]; pool["dom"] += doms
    return {"emb": torch.cat(pool["emb"]), "y": torch.cat(pool["y"]).numpy(),
            "size": torch.cat(pool["size"]).numpy(),
            "pid": np.array(pool["pid"]), "dom": np.array(pool["dom"])}


@torch.no_grad()
def predict(ckpt_path: Path, emb: torch.Tensor, device) -> tuple[np.ndarray, np.ndarray]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head = build_dinobloom_attribute_head(ck["embed_dim"], ck["num_attrs"]).to(device).eval()
    head.load_state_dict(ck["model"])
    nb = len(ck["binary_attrs"])
    out = torch.cat([head(emb[i:i + 4096].to(device)).cpu() for i in range(0, len(emb), 4096)])
    return torch.sigmoid(out[:, :nb]).numpy(), out[:, nb:].argmax(1).numpy()


def summarize(y, p, s_true, s_pred) -> dict:
    m = attribute_metrics(y, p, BINARY_ATTRS)
    sm = cell_size_metrics(s_true, s_pred)
    if sm:
        m["Cell_Size"] = sm
    f1 = {k: float(v.get("f1", v.get("f1_macro", 0.0))) for k, v in m.items()}
    return {"per_attribute_f1": f1, "mean_f1": float(np.mean(list(f1.values()))), "n_cells": int(len(y))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True, help="C1+C2 attribute embedding cache")
    ap.add_argument("--manifest", type=Path, required=True, help="the manifest the cache was built from")
    ap.add_argument("--probes", nargs="+", required=True, help="NAME=dir_with_fold{k}_attr_dinobloom.pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() and torch.cuda.is_available() else "cpu")
    pool = load_pool(args.cache, args.manifest)
    folds = load_folds()
    probes = dict(p.split("=", 1) for p in args.probes)

    results = {}
    for name, pdir in probes.items():
        acc = {d: {"y": [], "p": [], "st": [], "sp": []} for d in ("C2", "C1", "all")}
        for k, test_pids in enumerate(folds):
            m = np.isin(pool["pid"], sorted(test_pids))
            p, sp = predict(Path(pdir) / f"fold{k}_attr_dinobloom.pt", pool["emb"][torch.from_numpy(m)], device)
            y, st, dom = pool["y"][m], pool["size"][m], pool["dom"][m]
            for d in ("C2", "C1", "all"):
                sel = np.ones(len(dom), bool) if d == "all" else dom == d
                acc[d]["y"].append(y[sel]); acc[d]["p"].append(p[sel])
                acc[d]["st"].append(st[sel]); acc[d]["sp"].append(sp[sel])
        results[name] = {d: summarize(np.concatenate(v["y"]), np.concatenate(v["p"]),
                                      np.concatenate(v["st"]), np.concatenate(v["sp"]))
                         for d, v in acc.items()}

    args.out.write_text(json.dumps(results, indent=2))
    attrs = list(next(iter(results.values()))["all"]["per_attribute_f1"])
    print(f"{'probe':6s} {'test':4s} {'n':>6s} {'mean':>6s} " + " ".join(f"{a[:10]:>10s}" for a in attrs))
    for name, r in results.items():
        for d, v in r.items():
            print(f"{name:6s} {d:4s} {v['n_cells']:6d} {v['mean_f1']:6.3f} " +
                  " ".join(f"{v['per_attribute_f1'][a]:10.3f}" for a in attrs))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()