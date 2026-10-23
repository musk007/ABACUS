#!/usr/bin/env python3
"""cAItomorph external test: crop-level HALO variant, 5-fold ensemble.

cAItomorph ships pre-cropped single cells, so YOLO cannot run. Per fold f:
  DinoBloom-L embedding (frozen, shared)
    -> cell-group head f   (train_cellType.py --target group, LLD C1+C2)
    -> attribute probe f   (train_dinobloom_attributes_torch.py)
    -> patient features    (group %, attribute rates, cell-size probs; no absolute counts)
    -> random forest f     (trained here on the fold's LLD training patients, one row
                            per patient x camera, features from the same heads on GT crops)
Ensemble: average the 5 forests' class probabilities; p(acute) = P(ALL)+P(AML)+P(APML).

Also reports the crop-level LLD cross-validation (each fold's held-out patients) so the
external numbers have an in-domain reference.

    python caitomorph_crop_ensemble.py --data-root /path/caitomorph \
      --lld-cache wbc_unified/cv/runs/celltype/dinobloom_C1C2_group/embeddings_l_224.pt \
      --celltype-dir wbc_unified/cv/runs/celltype/dinobloom_C1C2_group \
      --attr-dir wbc_unified/cv/runs/attribute_dinobloom_c1c2/folds \
      --folds-json wbc_unified/config/cv_folds.json \
      --dinobloom-weights /home/roba.majzoub/DinoBloom-L.pth \
      --cait-cache outputs/external/caitomorph_emb_l_224.pt --out outputs/external/crop_ensemble
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CV_ROOT = HERE / "wbc_unified" / "cv"
for p in (HERE.parent, HERE, CV_ROOT):
    sys.path.insert(0, str(p))

ACUTE = {"ALL", "AML", "APML"}
FINE_TO_LLD = {"ALL": "ALL", "AML": "AML", "CML": "CML"}          # cAItomorph labels with an LLD counterpart
GROUPS = ["blasts", "intermediate_myeloid", "mature_granulocytes", "lymphoid", "monocytic"]
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


# ------------------------------------------------------------------ features (numpy only)
def unit_features(group_prob: np.ndarray, group_names: list[str], attr_prob: np.ndarray,
                  size_prob: np.ndarray, attr_names: list[str]) -> dict[str, float]:
    """Patient features from one set of cells (percentages / means only, count-free)."""
    g = group_prob.argmax(1)
    names = np.array(group_names)[g]
    keep = names != "None"
    n = int(keep.sum())
    row = {"frac_artefact": float(1 - keep.mean()) if len(keep) else 0.0}
    for grp in GROUPS:
        row[f"group_{grp}_pct"] = float(100 * (names[keep] == grp).mean()) if n else 0.0
    for j, a in enumerate(attr_names):
        v = attr_prob[keep, j] if n else np.zeros(0)
        row[f"attr_{a.lower()}__positive_pct"] = float(100 * (v >= 0.5).mean()) if n else 0.0
        row[f"attr_{a.lower()}__mean_prob"] = float(v.mean()) if n else 0.0
    for j, s in enumerate(("small", "medium", "large")):
        row[f"attr_cell_size__{s}_mean_prob"] = float(size_prob[keep, j].mean()) if n else 0.0
    return row


def binary_metrics(y_acute: np.ndarray, p_acute: np.ndarray) -> dict:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    pred = p_acute >= 0.5
    out = {
        "n": int(len(y_acute)), "n_acute": int(y_acute.sum()), "n_chronic": int((~y_acute).sum()),
        "accuracy": float((pred == y_acute).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_acute, pred)),
        "sensitivity_acute": float(pred[y_acute].mean()) if y_acute.any() else None,
        "specificity_chronic": float((~pred[~y_acute]).mean()) if (~y_acute).any() else None,
    }
    out["auroc"] = float(roc_auc_score(y_acute, p_acute)) if 0 < y_acute.sum() < len(y_acute) else None
    return out


def align(proba: np.ndarray, classes, all_classes: list[str]) -> np.ndarray:
    out = np.zeros((len(proba), len(all_classes)))
    for j, c in enumerate(classes):
        out[:, all_classes.index(str(c))] = proba[:, j]
    return out


# ------------------------------------------------------------------ torch parts
def load_heads(fold: int, celltype_dir: Path, attr_dir: Path, embed_dim: int):
    import torch
    from train_cellType import build_probe_head
    from agentic_hematology.detection_agent_dinobloom import build_dinobloom_attribute_head

    ct_path = next(celltype_dir.glob(f"*_cv_fold{fold}/fold{fold}_celltype.pt"))
    ct = torch.load(ct_path, map_location="cpu", weights_only=False)
    ct_head = build_probe_head(embed_dim, ct["n_classes"])
    ct_head.load_state_dict(ct["model"])
    at = torch.load(attr_dir / f"fold{fold}_attr_dinobloom.pt", map_location="cpu", weights_only=False)
    at_head = build_dinobloom_attribute_head(embed_dim, at["num_attrs"])
    at_head.load_state_dict(at["model"])
    return ct_head.eval(), list(ct["class_names"]), at_head.eval(), list(at["binary_attrs"])


def apply_heads(emb, ct_head, at_head, nb: int, device, bs: int = 4096):
    import torch

    gp, ap, sp = [], [], []
    ct_head.to(device)
    at_head.to(device)
    with torch.no_grad():
        for i in range(0, len(emb), bs):
            x = emb[i:i + bs].to(device)
            gp.append(torch.softmax(ct_head(x), 1).cpu())
            a = at_head(x)
            ap.append(torch.sigmoid(a[:, :nb]).cpu())
            sp.append(torch.softmax(a[:, nb:], 1).cpu())
    return torch.cat(gp).numpy(), torch.cat(ap).numpy(), torch.cat(sp).numpy()


def embed_caitomorph(rows, patients_dir: Path, args) -> dict:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from agentic_hematology.detection_agent_dinobloom import DinoBloomEmbedder, resolve_dinobloom_weights
    from train_dinobloom_attributes_torch import embed_tensor_batch

    files, pids = [], []
    for r in rows:
        d = patients_dir / r["patient_id"]
        crops = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) if d.is_dir() else []
        if args.max_cells and len(crops) > args.max_cells:
            crops = [crops[i] for i in np.linspace(0, len(crops) - 1, args.max_cells).astype(int)]
        files += [str(p) for p in crops]
        pids += [r["patient_id"]] * len(crops)

    class Crops(Dataset):
        def __len__(self):
            return len(files)

        def __getitem__(self, i):
            im = Image.open(files[i]).convert("RGB").resize((args.imgsz, args.imgsz), Image.BILINEAR)
            arr = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD
            return torch.from_numpy(arr).permute(2, 0, 1).float()

    device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    embedder = DinoBloomEmbedder(weights_path=resolve_dinobloom_weights(str(args.dinobloom_weights), args.dinobloom_variant),
                                 variant=args.dinobloom_variant, device=device)
    embedder.model.eval()
    embs = []
    from tqdm import tqdm
    for x in tqdm(DataLoader(Crops(), batch_size=args.batch, num_workers=args.workers), desc="embedding cAItomorph"):
        embs.append(embed_tensor_batch(embedder, x).cpu())
    return {"emb": torch.cat(embs), "patient_ids": pids, "files": files, "imgsz": args.imgsz}


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True, help="metadata.csv + patients/<pid>/*")
    ap.add_argument("--lld-cache", type=Path, required=True, help="train_cellType.py embedding cache (C1+C2)")
    ap.add_argument("--celltype-dir", type=Path, required=True, help="group-head project dir (*_cv_fold{k}/)")
    ap.add_argument("--attr-dir", type=Path, required=True, help="dir with fold{k}_attr_dinobloom.pt")
    ap.add_argument("--folds-json", type=Path, required=True)
    ap.add_argument("--dinobloom-weights", type=Path, required=True)
    ap.add_argument("--dinobloom-variant", default="l")
    ap.add_argument("--cait-cache", type=Path, required=True, help="where to store/reuse cAItomorph embeddings")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exclude", nargs="*", default=["24"])
    ap.add_argument("--lld-domains", nargs="*", default=None,
                help="LLD cameras used to train the random forests (default: all)")
    ap.add_argument("--in-distribution-only", action="store_true", help="only ALL/AML/CML patients (faster)")
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    import torch
    from agentic_hematology.detection_agent_dinobloom import DINOBLOOM_VARIANTS
    from agentic_hematology.tabular_classifier import build_tabular_classifier, fit_tabular_classifier

    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    embed_dim = DINOBLOOM_VARIANTS[args.dinobloom_variant][1]
    cv = json.loads(args.folds_json.read_text())
    dx = cv["diagnosis"]
    all_classes = sorted({dx[p] for p in dx if p not in args.exclude})

    # --- LLD crop embeddings (C1+C2), grouped by patient x camera
    lc = torch.load(args.lld_cache, map_location="cpu", weights_only=False)
    l_emb = torch.cat([lc[s]["emb"] for s in ("train", "test")])
    l_pid = np.array([str(p) for s in ("train", "test") for p in lc[s]["patient_ids"]])
    l_dom = np.array([str(d) for s in ("train", "test") for d in lc[s]["domains"]])
    units = sorted({(p, d) for p, d in zip(l_pid, l_dom)
                if p not in args.exclude and (not args.lld_domains or d in args.lld_domains)})
    print(f"LLD: {len(l_emb)} crops, {len(units)} patient x camera units")

    # --- cAItomorph crops
    rows = list(csv.DictReader((args.data_root / "metadata.csv").open(newline="", encoding="utf-8")))
    if args.in_distribution_only:
        rows = [r for r in rows if r["diagnosis_fine"] in FINE_TO_LLD]
    if args.cait_cache.is_file():
        cc = torch.load(args.cait_cache, map_location="cpu", weights_only=False)
        print(f"Loaded cAItomorph embeddings from {args.cait_cache}")
    else:
        cc = embed_caitomorph(rows, args.data_root / "patients", args)
        args.cait_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cc, args.cait_cache)
    c_pid = np.array(cc["patient_ids"])
    meta = {r["patient_id"]: r for r in rows}
    c_patients = [p for p in dict.fromkeys(c_pid.tolist()) if p in meta]
    print(f"cAItomorph: {len(cc['emb'])} crops, {len(c_patients)} patients")

    lld_rows, cait_probs, cait_blast = [], [], []
    for f, test_pids in enumerate(cv["folds"]):
        ct_head, gnames, at_head, anames = load_heads(f, args.celltype_dir, args.attr_dir, embed_dim)
        nb = len(anames)
        lg, la, ls = apply_heads(l_emb, ct_head, at_head, nb, device)
        cg, ca, cs = apply_heads(cc["emb"], ct_head, at_head, nb, device)

        feats, keys = [], []
        for p, d in units:
            m = (l_pid == p) & (l_dom == d)
            feats.append(unit_features(lg[m], gnames, la[m], ls[m], anames))
            keys.append((p, d))
        names = sorted(feats[0])
        X = np.array([[r[k] for k in names] for r in feats])
        is_test = np.array([p in set(test_pids) for p, _ in keys])
        y_train = [dx[p] for (p, _), t in zip(keys, is_test) if not t]

        clf = build_tabular_classifier("random_forest", random_state=args.random_state)
        clf, _ = fit_tabular_classifier(clf, X[~is_test], y_train, backend="random_forest")

        # LLD held-out: average the two cameras per patient
        pt = align(clf.predict_proba(X[is_test]), clf.classes_, all_classes)
        tkeys = [k for k, t in zip(keys, is_test) if t]
        for p in sorted({k[0] for k in tkeys}, key=int):
            pr = pt[[i for i, k in enumerate(tkeys) if k[0] == p]].mean(0)
            lld_rows.append({"fold": f, "patient_id": p, "true": dx[p],
                             "pred": all_classes[int(pr.argmax())],
                             "p_acute": float(sum(pr[all_classes.index(c)] for c in ACUTE if c in all_classes))})

        cfeat = []
        for p in c_patients:
            m = c_pid == p
            cfeat.append(unit_features(cg[m], gnames, ca[m], cs[m], anames))
        Xc = np.array([[r[k] for k in names] for r in cfeat])
        cait_probs.append(align(clf.predict_proba(Xc), clf.classes_, all_classes))
        cait_blast.append([r["group_blasts_pct"] for r in cfeat])
        print(f"fold {f}: trained on {int((~is_test).sum())} units, "
              f"LLD held-out acc {np.mean([r['pred'] == r['true'] for r in lld_rows if r['fold'] == f]):.3f}")

    # --- ensemble on cAItomorph
    P = np.stack(cait_probs)                                   # folds x patients x classes
    acute_idx = [all_classes.index(c) for c in ACUTE if c in all_classes]
    p_acute_folds = P[:, :, acute_idx].sum(2)                  # folds x patients
    p_ens = P.mean(0)
    p_acute = p_acute_folds.mean(0)
    blast = np.mean(cait_blast, 0)

    out_rows = []
    for i, p in enumerate(c_patients):
        r = meta[p]
        out_rows.append({
            "patient_id": p, "diagnosis_coarse": r.get("diagnosis_coarse", ""), "diagnosis_fine": r["diagnosis_fine"],
            "lld_label": FINE_TO_LLD.get(r["diagnosis_fine"], ""), "n_crops": int((c_pid == p).sum()),
            "blast_pct": round(float(blast[i]), 2), "p_acute": round(float(p_acute[i]), 4),
            "pred_binary": "acute" if p_acute[i] >= 0.5 else "chronic",
            "pred_class": all_classes[int(p_ens[i].argmax())],
            **{f"p_acute_fold{f}": round(float(p_acute_folds[f, i]), 4) for f in range(len(P))},
        })

    ind = [i for i, p in enumerate(c_patients) if meta[p]["diagnosis_fine"] in FINE_TO_LLD]
    y_ind = np.array([FINE_TO_LLD[meta[c_patients[i]]["diagnosis_fine"]] in ACUTE for i in ind])
    per_fold = [binary_metrics(y_ind, p_acute_folds[f, ind]) for f in range(len(P))]
    lld_y = np.array([r["true"] in ACUTE for r in lld_rows])
    lld_p = np.array([r["p_acute"] for r in lld_rows])
    summary = {
        "classes": all_classes,
        "caitomorph_acute_vs_chronic": {
            "ensemble": binary_metrics(y_ind, p_acute[ind]),
            "per_fold": per_fold,
            "per_fold_mean_std": {k: [float(np.mean([m[k] for m in per_fold if m[k] is not None])),
                                      float(np.std([m[k] for m in per_fold if m[k] is not None]))]
                                  for k in ("accuracy", "balanced_accuracy", "auroc")
                                  if any(m[k] is not None for m in per_fold)},
            "three_class_accuracy_ALL_AML_CML": float(np.mean(
                [all_classes[int(p_ens[i].argmax())] == FINE_TO_LLD[meta[c_patients[i]]["diagnosis_fine"]] for i in ind])),
        },
        "lld_crop_cv": {
            "n": len(lld_rows),
            "five_class_accuracy": float(np.mean([r["pred"] == r["true"] for r in lld_rows])),
            "acute_vs_chronic": binary_metrics(lld_y, lld_p),
        },
        "caitomorph_by_diagnosis": {},
    }
    for key in ("diagnosis_coarse", "diagnosis_fine"):
        groups = {}
        for r in out_rows:
            groups.setdefault(r[key] or "?", []).append(r)
        summary["caitomorph_by_diagnosis"][key] = {
            g: {"n": len(rs), "mean_p_acute": round(float(np.mean([r["p_acute"] for r in rs])), 4),
                "frac_pred_acute": round(float(np.mean([r["pred_binary"] == "acute" for r in rs])), 4),
                "mean_blast_pct": round(float(np.mean([r["blast_pct"] for r in rs])), 2)}
            for g, rs in sorted(groups.items())}

    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in (("caitomorph_per_patient.csv", out_rows), ("lld_crop_cv_per_patient.csv", lld_rows)):
        with (args.out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    summary["provenance"] = {k: str(v) for k, v in vars(args).items()}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("caitomorph_acute_vs_chronic", "lld_crop_cv")}, indent=2))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()