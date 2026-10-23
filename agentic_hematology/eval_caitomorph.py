#!/usr/bin/env python3
"""External-cohort evaluation of the patient-level leukemia classifier.

This cohort ships PRE-CROPPED single cells (patients/<PID>/*.TIF), not whole PBS
tiles, so the pipeline is entered one stage later than usual:

    crop -> cell type (YOLO class head, or the trained cell-type classifier)
         -> attribute head (EfficientNet or DinoBloom + MLP)
         -> Detection objects -> aggregate() -> patient features -> classifier

Label caveat (read before interpreting anything):
    The LLD classifier emits {ALL, AML, APML, CLL, CML}. This cohort has 18 fine
    labels; only ALL (7 patients), AML (37) and CML (8) have a counterpart, and
    APML/CLL do not occur here at all. The remaining 357 patients (MDS, MPN, MM,
    lymphoma, healthy donors, ...) are OUT OF THE LABEL SET and cannot be
    predicted correctly by construction.

    So this script reports two separate things:
      [A] in-distribution: 3-class accuracy on the 52 mappable patients. This is
          the external-validation / domain-shift number.
      [B] full cohort: predicted-class distribution and blast% per diagnosis
          group. Not an accuracy — a check on whether blast burden separates
          acute leukemia from healthy donors under domain shift, which is
          label-set independent and arguably the more interesting result.

    python val_caitomorph.py --backend dinobloom \\
      --data-root /path/to/caitomorph_test_set \\
      --yolo-weights wbc_unified/cv/runs/detector/train/weights/best.pt \\
      --dinobloom-weights /path/DinoBloom-L.pth \\
      --dinobloom-attr-weights wbc_unified/cv/runs/attribute_dinobloom/train/best_attr_dinobloom.pt \\
      --classifier-model runs/classifier/dinobloom/random_forest/leukemia_random_forest.pkl \\
      --out outputs/external/dinobloom_rf.json --device 0
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from agentic_hematology.aggregator import aggregate  # noqa: E402
from agentic_hematology.detection_agent import LLD_CLASSES  # noqa: E402
from agentic_hematology.leukemia_features import build_feature_row_from_findings  # noqa: E402
from agentic_hematology.schemas import Detection, DetectionResult  # noqa: E402

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# Fine labels in this cohort that have a counterpart in the LLD label set.
# APML and CLL do not occur here; everything else is out of the label set.
FINE_TO_LLD: dict[str, str] = {
    "ALL": "ALL",
    "AML": "AML",
    "CML": "CML",
}


# ---------------------------------------------------------------------------
# cohort loading
# ---------------------------------------------------------------------------

def load_metadata(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def patient_crops(patients_dir: Path, pid: str, max_cells: int | None) -> list[Path]:
    d = patients_dir / pid
    if not d.is_dir():
        return []
    crops = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if max_cells is not None and len(crops) > max_cells:
        # deterministic even subsample so the differential stays representative
        idx = np.linspace(0, len(crops) - 1, max_cells).astype(int)
        crops = [crops[i] for i in idx]
    return crops


# ---------------------------------------------------------------------------
# cell typing
# ---------------------------------------------------------------------------

class YoloCropTyper:
    """Assign a cell type per crop using the YOLO class head.

    Matches how the patient classifier's training features were produced (cell
    type from YOLO), so this is the default. Each crop is a single cell, so we
    take the highest-confidence detection in it.
    """

    def __init__(self, weights: Path, device: str, conf: float, imgsz: int = 640):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.names = self.model.names
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def type_batch(self, paths: list[Path]) -> list[tuple[str, float]]:
        results = self.model.predict(
            source=[str(p) for p in paths],
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        out: list[tuple[str, float]] = []
        for r in results:
            if r.boxes is None or not len(r.boxes):
                out.append(("None", 0.0))
                continue
            confs = r.boxes.conf.cpu().numpy()
            best = int(confs.argmax())
            cid = int(r.boxes.cls.cpu().numpy()[best])
            name = str(self.names[cid]) if cid in self.names else "None"
            out.append((name, float(confs[best])))
        return out


class ClassifierCropTyper:
    """Assign a cell type per crop using a trained cell-type classifier."""

    def __init__(self, weights: Path, device_str: str):
        import torch

        from wbc_unified.cv.eval_cellType import load_model  # wbc_unified/cv/eval_celltype.py

        device = torch.device(f"cuda:{device_str}" if device_str.isdigit() else device_str)
        self.torch = torch
        self.model, self.embedder, self.class_names, self.imgsz, ckpt = load_model(weights, device)
        self.device = device
        if ckpt.get("target") != "celltype":
            raise SystemExit(
                f"Cell-type checkpoint has target={ckpt.get('target')!r}; need 'celltype'."
            )

    def type_batch(self, paths: list[Path]) -> list[tuple[str, float]]:
        from utils.e2e_attributes import crops_to_tensor
        from wbc_unified.cv.eval_cellType import embed_tensor_batch

        torch = self.torch
        crops = [Image.open(p).convert("RGB") for p in paths]
        x = crops_to_tensor(crops, self.imgsz)
        with torch.inference_mode():
            logits = self.model(embed_tensor_batch(self.embedder, x)) if self.embedder is not None \
                else self.model(x.to(self.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        idx = probs.argmax(axis=1)
        return [(self.class_names[i], float(probs[k, i])) for k, i in enumerate(idx)]


# ---------------------------------------------------------------------------
# attribute head
# ---------------------------------------------------------------------------

def build_attribute_head(args):
    if args.backend == "efficientnet":
        from agentic_hematology.detection_agent_v2 import EfficientNetAttributeClassifier

        return EfficientNetAttributeClassifier(
            weights_path=str(args.effnet_weights), device=args.device
        )
    from agentic_hematology.detection_agent_dinobloom import DinoBloomAttributeClassifier

    return DinoBloomAttributeClassifier(
        weights_path=str(args.dinobloom_weights),
        attr_probes_path=str(args.dinobloom_attr_weights),
        variant=args.dinobloom_variant,
        attr_mode="probes",
        device=args.device,
    )


# ---------------------------------------------------------------------------

def detections_for_patient(
    pid: str,
    crop_paths: list[Path],
    typer,
    attr_head,
    batch: int,
) -> DetectionResult:
    detections: list[Detection] = []
    for start in range(0, len(crop_paths), batch):
        chunk = crop_paths[start : start + batch]
        types = typer.type_batch(chunk)
        crops = [Image.open(p).convert("RGB") for p in chunk]
        attr_results = attr_head.classify_crops(crops)
        for k, (path, (cell_type, conf)) in enumerate(zip(chunk, types)):
            attrs, attr_probs, _, _ = attr_results[k]
            attrs = dict(attrs)
            attrs["class_id"] = None
            w, h = crops[k].size
            detections.append(
                Detection(
                    cell_id=f"{pid}_c{start + k:06d}",
                    image_id=path.name,
                    bbox_xyxy=(0.0, 0.0, float(w), float(h)),
                    cell_type=cell_type,
                    objectness=float(conf),
                    cell_type_prob=float(conf),
                    attributes=attrs,
                    attribute_probs=dict(attr_probs),
                )
            )
    return DetectionResult(case_id=pid, n_images=len(crop_paths), detections=detections)


def main() -> None:
    ap = argparse.ArgumentParser(description="External-cohort patient-level evaluation.")
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Directory containing metadata.csv and patients/")
    ap.add_argument("--backend", choices=("efficientnet", "dinobloom"), required=True)
    ap.add_argument("--celltype-source", choices=("yolo", "classifier"), default="yolo",
                    help="yolo matches how the patient classifier's features were built.")
    ap.add_argument("--yolo-weights", type=Path, default=None)
    ap.add_argument("--celltype-weights", type=Path, default=None)
    ap.add_argument("--effnet-weights", type=Path, default=None)
    ap.add_argument("--dinobloom-weights", type=Path, default=None)
    ap.add_argument("--dinobloom-attr-weights", type=Path, default=None)
    ap.add_argument("--dinobloom-variant", choices=["s", "b", "l", "g"], default="l")
    ap.add_argument("--classifier-model", type=Path, required=True)
    ap.add_argument("--classifier-meta", type=Path, default=None)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--conf-threshold", type=float, default=0.25,
                    help="Detection confidence; also the aggregate() threshold.")
    ap.add_argument("--max-cells", type=int, default=None,
                    help="Cap cells per patient (evenly subsampled) to bound runtime.")
    ap.add_argument("--in-distribution-only", action="store_true",
                    help="Skip the 357 patients whose diagnosis is outside the LLD label set.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    meta_path = args.data_root / "metadata.csv"
    patients_dir = args.data_root / "patients"
    if not meta_path.is_file():
        raise SystemExit(f"metadata.csv not found: {meta_path}")
    if not patients_dir.is_dir():
        raise SystemExit(f"patients/ not found: {patients_dir}")

    rows = load_metadata(meta_path)
    if args.in_distribution_only:
        rows = [r for r in rows if r["diagnosis_fine"] in FINE_TO_LLD]
    print(f"Cohort: {len(rows)} patients "
          f"({sum(1 for r in rows if r['diagnosis_fine'] in FINE_TO_LLD)} in the LLD label set)")

    # --- build the two stages ------------------------------------------
    if args.celltype_source == "yolo":
        if not args.yolo_weights:
            raise SystemExit("--yolo-weights is required for --celltype-source yolo")
        typer = YoloCropTyper(args.yolo_weights, args.device, args.conf_threshold)
    else:
        if not args.celltype_weights:
            raise SystemExit("--celltype-weights is required for --celltype-source classifier")
        typer = ClassifierCropTyper(args.celltype_weights, args.device)

    attr_head = build_attribute_head(args)

    # --- run the pipeline per patient ----------------------------------
    records: list[dict] = []
    for r in tqdm(rows, desc="patients"):
        pid = r["patient_id"]
        crops = patient_crops(patients_dir, pid, args.max_cells)
        if not crops:
            print(f"  no crops for {pid}, skipping", file=sys.stderr)
            continue
        det = detections_for_patient(pid, crops, typer, attr_head, args.batch)
        findings = aggregate(det, conf_threshold=args.conf_threshold)
        row = build_feature_row_from_findings(findings, detection_result=det)
        records.append({
            "patient_id": pid,
            "diagnosis_coarse": r["diagnosis_coarse"],
            "diagnosis_fine": r["diagnosis_fine"],
            "lld_label": FINE_TO_LLD.get(r["diagnosis_fine"]),
            "n_cells": len(crops),
            "n_informative": findings.n_cells_identified_wbc,
            "blast_pct": float(findings.report_ready.get("blast_pct", 0.0)),
            "differential": dict(findings.cell_percentages_clinical),
            "features": row,
        })

    if not records:
        raise SystemExit("No patients processed.")

    # --- patient classifier --------------------------------------------
    with open(args.classifier_model, "rb") as fh:
        clf = pickle.load(fh)
    meta_path_clf = args.classifier_meta or args.classifier_model.with_name(
        f"{args.classifier_model.stem}_meta.json"
    )
    feature_names, class_names = None, None
    if meta_path_clf.is_file():
        m = json.loads(meta_path_clf.read_text())
        feature_names = m.get("feature_names")
        class_names = m.get("classes")

    X = pd.DataFrame([rec["features"] for rec in records],
                     index=[rec["patient_id"] for rec in records])
    if feature_names:
        X = X.reindex(columns=feature_names, fill_value=0.0)

    raw = clf.predict(X)
    if len(raw) and np.issubdtype(np.asarray(raw).dtype, np.number) and class_names:
        preds = [class_names[int(i)] if int(i) < len(class_names) else str(i) for i in raw]
    else:
        preds = [str(v) for v in raw]
    for rec, p in zip(records, preds):
        rec["predicted_class"] = p

    # --- [A] in-distribution accuracy ----------------------------------
    ind = [rec for rec in records if rec["lld_label"]]
    report = {}
    if ind:
        y_true = [rec["lld_label"] for rec in ind]
        y_pred = [rec["predicted_class"] for rec in ind]
        labels = sorted(set(y_true) | set(y_pred))
        acc = accuracy_score(y_true, y_pred)
        print(f"\n[A] In-distribution: {len(ind)} patients "
              f"({dict(Counter(y_true))})")
        print(f"    accuracy = {acc:.3f}")
        print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
        print(pd.DataFrame(
            confusion_matrix(y_true, y_pred, labels=labels),
            index=[f"True_{c}" for c in labels],
            columns=[f"Pred_{c}" for c in labels],
        ))
        report["in_distribution"] = {
            "n_patients": len(ind),
            "accuracy": float(acc),
            "labels": labels,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, labels=labels, zero_division=0
            ),
        }

    # --- [B] full-cohort behaviour --------------------------------------
    by_group: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_group[rec["diagnosis_coarse"]].append(rec)

    print(f"\n[B] Full cohort ({len(records)} patients): blast% and predictions by group")
    print(f"{'group':24s} {'n':>4} {'blast% mean':>12} {'median':>8}   top predictions")
    group_rows = {}
    for grp, recs in sorted(by_group.items()):
        b = np.array([r["blast_pct"] for r in recs])
        top = Counter(r["predicted_class"] for r in recs).most_common(3)
        print(f"{grp:24s} {len(recs):4d} {b.mean():12.1f} {np.median(b):8.1f}   "
              + ", ".join(f"{c}={n}" for c, n in top))
        group_rows[grp] = {
            "n_patients": len(recs),
            "blast_pct_mean": float(b.mean()),
            "blast_pct_median": float(np.median(b)),
            "blast_pct_std": float(b.std()),
            "predicted_distribution": dict(Counter(r["predicted_class"] for r in recs)),
        }
    report["by_diagnosis_group"] = group_rows

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "backend": args.backend,
            "celltype_source": args.celltype_source,
            "classifier_model": str(args.classifier_model.resolve()),
            "conf_threshold": args.conf_threshold,
            "max_cells": args.max_cells,
            "n_patients": len(records),
            **report,
            "patients": [
                {k: v for k, v in rec.items() if k != "features"} for rec in records
            ],
        }, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()