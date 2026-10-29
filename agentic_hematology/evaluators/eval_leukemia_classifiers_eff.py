#!/usr/bin/env python3
import argparse
import json
import pickle
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from agentic_hematology.aggregator import aggregate
from agentic_hematology.detection_agent_v2 import (
    EfficientNetAttributeClassifier,
    TwoStageDetectionAgent,
    YOLOv11Localizer,
)
from agentic_hematology.leukemia_features import (
    build_feature_row_from_findings,
    discover_lld_split_from_cv,
    discover_patient_labels_from_cv,
)


def group_patients(split, image_root: Path, cv_root: Path):
    derived = discover_lld_split_from_cv(cv_root)
    patient_ids = derived.get(split, [])

    image_dir = image_root / split
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    by_patient = {pid: [] for pid in patient_ids}
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() not in suffixes:
            continue
        pid = path.stem.split("_")[0]
        if pid in by_patient:
            by_patient[pid].append(str(path.resolve()))

    return [(pid, paths) for pid, paths in by_patient.items() if paths]


def build_detector(args):
    localizer = YOLOv11Localizer(
        weights_path=str(args.yolo_weights),
        device=args.device,
        conf_threshold=args.conf_threshold,
    )

    attr_clf = EfficientNetAttributeClassifier(
        weights_path=str(args.effnet_weights),
        device=args.device,
        predicts_cell_type=args.effnet_predicts_celltype,
    )

    return TwoStageDetectionAgent(
        localizer=localizer,
        attribute_classifier=attr_clf,
        attribute_head_name="EfficientNet",
    )


def rows_to_dataframe(patient_ids, rows):
    feature_names = sorted({k for row in rows for k in row})
    return pd.DataFrame(
        [[row.get(name, 0.0) for name in feature_names] for row in rows],
        columns=feature_names,
        index=patient_ids,
    )


def predict_labels(clf, X, class_names):
    raw = clf.predict(X)

    
    if len(raw) > 0 and np.issubdtype(np.asarray(raw).dtype, np.number):
        le = LabelEncoder()
        le.classes_ = pd.Index(class_names).to_numpy()
        return [str(x) for x in le.inverse_transform(raw)]

    return [str(x) for x in raw]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--cv-root", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "test"], default="test")

    ap.add_argument("--yolo-weights", type=Path, required=True)
    ap.add_argument("--effnet-weights", type=Path, required=True)
    ap.add_argument("--effnet-predicts-celltype", action="store_true")

    ap.add_argument("--classifier-model", type=Path, required=True)
    ap.add_argument("--classifier-meta", type=Path, default=None)

    ap.add_argument("--device", default="0")
    ap.add_argument("--conf-threshold", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=None)

    args = ap.parse_args()

    labels = discover_patient_labels_from_cv(args.cv_root, image_root=args.image_root)
    patients = group_patients(args.split, args.image_root, args.cv_root)

    detector = build_detector(args)

    patient_ids, rows, y_true = [], [], []

    for pid, image_paths in patients:
        if pid not in labels:
            continue

        print(f"Running patient {pid} ({len(image_paths)} images)...", flush=True)

        detection_result = detector.detect(pid, image_paths)
        findings = aggregate(detection_result)

        row = build_feature_row_from_findings(
            findings,
            detection_result=detection_result,
        )

        patient_ids.append(pid)
        rows.append(row)
        y_true.append(labels[pid])

    X = rows_to_dataframe(patient_ids, rows)

    with open(args.classifier_model, "rb") as f:
        clf = pickle.load(f)

    if args.classifier_meta and args.classifier_meta.is_file():
        meta = json.loads(args.classifier_meta.read_text())
        class_names = meta.get("classes", sorted(set(y_true)))
        train_features = meta.get("feature_names")

        if train_features:
            X = X.reindex(columns=train_features, fill_value=0.0)
    else:
        class_names = sorted(set(y_true))

    y_pred = predict_labels(clf, X, class_names)
    # ---------------------------
    # Confusion matrix
    # ---------------------------
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=class_names,
    )

    print("\nConfusion Matrix:")
    print(pd.DataFrame(
        cm,
        index=[f"True_{c}" for c in class_names],
        columns=[f"Pred_{c}" for c in class_names],
    ))

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=11,
            )

    fig.tight_layout()

    plt.savefig("confusion_matrix.pdf", bbox_inches="tight")
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

    acc = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    print("\n===== EfficientNet leukemia classifier inference metrics =====")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")
    print(f"Micro F1    : {micro_f1:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=class_names,
        digits=4,
        zero_division=0,
    ))

    results = {
        "accuracy": float(acc),
        "weighted_f1": float(weighted_f1),
        "micro_f1": float(micro_f1),
        "n_test": len(y_true),
        "patients": [
            {
                "patient_id": pid,
                "true_label": true,
                "predicted_label": pred,
            }
            for pid, true, pred in zip(patient_ids, y_true, y_pred)
        ],
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nSaved results to: {args.out}")


if __name__ == "__main__":
    main()