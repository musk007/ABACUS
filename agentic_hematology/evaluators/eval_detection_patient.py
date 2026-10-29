"""
eval_detection_classification.py
----------------------------------
Proper IoU-based detection and cell classification evaluation.
Matches predicted boxes (case_<id>_detections.json) to GT boxes (YOLO .txt files).

Usage:
    python eval_detection_classification.py \
        --results_dir /path/to/results \
        --labels_dir  /path/to/gt_txt_labels \
        --iou_thresh  0.5
"""

import json, argparse, statistics
from pathlib import Path
from collections import Counter, defaultdict

IMG_W, IMG_H = 640, 640
ARTEFACT = {"none", "unknown"}

YOLO_CLASS_NAMES = {
    0: "none", 1: "myeloblast", 2: "lymphoblast", 3: "neutrophil",
    4: "atypical lymphocyte", 5: "promonocyte", 6: "monoblast",
    7: "lymphocyte", 8: "myelocyte", 9: "abnormal promyelocyte",
    10: "monocyte", 11: "metamyelocyte", 12: "eosinophil", 13: "basophil"
}

def load(path):
    with open(path) as f:
        return json.load(f)

def slug(name):
    return str(name).strip().lower()

def yolo_to_xyxy(cx, cy, w, h, img_w=IMG_W, img_h=IMG_H):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0

def load_gt_boxes(txt_path):
    """Returns list of (class_name, xyxy)."""
    boxes = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append((YOLO_CLASS_NAMES.get(cls_id, "none"), yolo_to_xyxy(cx, cy, w, h)))
    return boxes

def match_boxes(pred_boxes, gt_boxes, iou_thresh):
    """
    Greedy IoU matching. Returns (tp_pairs, fp_count, fn_count).
    tp_pairs: list of (pred_class, gt_class) for matched boxes.
    """
    matched_gt = set()
    tp_pairs = []

    for pred_cls, pred_box in pred_boxes:
        best_iou, best_j = 0.0, -1
        for j, (gt_cls, gt_box) in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            score = iou(pred_box, gt_box)
            if score > best_iou:
                best_iou, best_j = score, j
        if best_iou >= iou_thresh and best_j >= 0:
            matched_gt.add(best_j)
            tp_pairs.append((pred_cls, gt_boxes[best_j][0]))

    fp = len(pred_boxes) - len(tp_pairs)
    fn = len(gt_boxes) - len(matched_gt)
    return tp_pairs, fp, fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--labels_dir",  required=True, help="Directory with GT YOLO .txt files")
    ap.add_argument("--iou_thresh",  type=float, default=0.5)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    labels_dir  = Path(args.labels_dir)

    patient_ids = sorted(
        p.stem.replace("_detections", "").replace("case_", "")
        for p in results_dir.glob("case_*_detections.json")
    )
    assert patient_ids, "No detection files found."

    # Global accumulators
    det_tp = det_fp = det_fn = 0
    clf_correct = clf_total = 0
    clf_tp = defaultdict(int)
    clf_fp = defaultdict(int)
    clf_fn = defaultdict(int)
    all_classes = set()
    missing_labels = 0

    for pid in patient_ids:
        detections = load(results_dir / f"case_{pid}_detections.json")["detections"]

        # Group predictions by image
        by_image = defaultdict(list)
        for d in detections:
            img_id = d["image_id"].replace(".png", "").replace(".jpg", "")
            pred_cls = slug(d["class"])
            bbox = tuple(d["bbox_xyxy"])
            by_image[img_id].append((pred_cls, bbox))

        for img_id, pred_boxes in by_image.items():
            txt_path = labels_dir / f"{img_id}.txt"
            if not txt_path.exists():
                missing_labels += 1
                # Count all predictions as FP if no GT file
                det_fp += len(pred_boxes)
                continue

            gt_boxes = load_gt_boxes(txt_path)
            # Filter out artefact GT boxes for detection evaluation
            gt_wbc   = [(c, b) for c, b in gt_boxes if slug(c) not in ARTEFACT]
            pred_wbc = [(c, b) for c, b in pred_boxes if slug(c) not in ARTEFACT]

            tp_pairs, fp, fn = match_boxes(pred_wbc, gt_wbc, args.iou_thresh)
            det_tp += len(tp_pairs)
            det_fp += fp
            det_fn += fn

            # Cell classification on matched pairs only
            for pred_cls, gt_cls in tp_pairs:
                all_classes.update([pred_cls, gt_cls])
                clf_total += 1
                if pred_cls == gt_cls:
                    clf_correct += 1
                    clf_tp[gt_cls] += 1
                else:
                    clf_fp[pred_cls] += 1
                    clf_fn[gt_cls]   += 1

    # ── Metrics ───────────────────────────────────────────────────────────
    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2*p*r / (p+r) if (p+r) else 0.0
        return round(p, 4), round(r, 4), round(f, 4)

    det_p, det_r, det_f = prf(det_tp, det_fp, det_fn)
    clf_acc = round(clf_correct / clf_total, 4) if clf_total else 0.0

    per_class = {}
    for cls in sorted(all_classes):
        tp = clf_tp[cls]
        fp = clf_fp[cls]
        fn = clf_fn[cls]
        p, r, f = prf(tp, fp, fn)
        per_class[cls] = {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}

    macro_p = round(statistics.mean(v["precision"] for v in per_class.values()), 4) if per_class else 0.0
    macro_r = round(statistics.mean(v["recall"]    for v in per_class.values()), 4) if per_class else 0.0
    macro_f = round(statistics.mean(v["f1"]        for v in per_class.values()), 4) if per_class else 0.0

    # ── Print ─────────────────────────────────────────────────────────────
    sep = "─" * 58
    print(f"\n{'═'*58}")
    print(f"  DETECTION & CLASSIFICATION  ({len(patient_ids)} patients, IoU≥{args.iou_thresh})")
    print(f"{'═'*58}")

    print(f"\n[1] DETECTION  (IoU ≥ {args.iou_thresh})")
    print(sep)
    print(f"  Precision : {det_p:.4f}")
    print(f"  Recall    : {det_r:.4f}")
    print(f"  F1        : {det_f:.4f}")
    print(f"  TP / FP / FN : {det_tp} / {det_fp} / {det_fn}")
    if missing_labels:
        print(f"  WARNING: {missing_labels} images had no GT label file (counted as FP).")

    print(f"\n[2] CELL CLASSIFICATION  (matched pairs only)")
    print(sep)
    print(f"  Overall accuracy : {clf_acc:.4f}  ({clf_correct}/{clf_total} matched cells)")
    print(f"\n  {'Class':<25} {'Prec':>6} {'Rec':>6} {'F1':>6}  TP/FP/FN")
    print(sep)
    for cls, m in per_class.items():
        print(f"  {cls:<25} {m['precision']:>6.4f} {m['recall']:>6.4f} {m['f1']:>6.4f}"
              f"  {m['tp']}/{m['fp']}/{m['fn']}")
    print(sep)
    print(f"  {'Macro average':<25} {macro_p:>6.4f} {macro_r:>6.4f} {macro_f:>6.4f}")
    print(f"\n{'═'*58}\n")

    # ── Save ──────────────────────────────────────────────────────────────
    out = {
        "n_patients": len(patient_ids),
        "iou_threshold": args.iou_thresh,
        "detection": {"precision": det_p, "recall": det_r, "f1": det_f,
                      "tp": det_tp, "fp": det_fp, "fn": det_fn},
        "cell_classification": {
            "overall_accuracy": clf_acc,
            "macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f,
            "per_class": per_class,
        }
    }
    out_path = results_dir / "detection_classification_eval.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()