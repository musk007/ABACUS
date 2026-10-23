#!/usr/bin/env python3
"""Train patient-level classifiers from YOLO + DinoBloom + MLP pipeline features.

Runs detection (YOLO localizer + frozen DinoBloom embedder + trained attribute MLP)
on the LLD train/test image splits, aggregates per-patient tabular features, then
fits Random Forest, XGBoost, and/or LightGBM on the train split and evaluates on
the held-out test split (same 34/13 patients as wbc_unified det_dataset).
Labels are parsed from LLD image filenames ({patient}_*_{DIAGNOSIS}.png).

Example:
  cd /home/roba.majzoub/agentic_hematology
  python train_leukemia_from_dinobloom.py --backend all --device 0

  # Re-train classifiers only (skip GPU detection):
  python train_leukemia_from_dinobloom.py --backend all \
    --features-cache runs/classifier/dinobloom/features.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from agentic_hematology.aggregator import aggregate  # noqa: E402
from agentic_hematology.detection_agent_dinobloom import DinoBloomAttributeClassifier  # noqa: E402
from agentic_hematology.detection_agent_v2 import (  # noqa: E402
    TwoStageDetectionAgent,
    YOLOv11Localizer,
)
from agentic_hematology.leukemia_features import (  # noqa: E402
    discover_lld_split_from_cv,
    discover_patient_labels_from_cv,
)
from agentic_hematology.schemas import DetectionResult  # noqa: E402
from agentic_hematology.tabular_classifier import (  # noqa: E402
    Backend,
    build_tabular_classifier,
    fit_tabular_classifier,
    native_feature_importances,
)

WBC_CV = HERE / "wbc_unified" / "cv"
DEFAULT_YOLO = WBC_CV / "runs" / "detector" / "train" / "weights" / "best.pt"
DEFAULT_DINOBLOOM_WEIGHTS = Path("/home/roba.majzoub/DinoBloom-L.pth")
DEFAULT_DINOBLOOM_ATTR = WBC_CV / "runs" / "attribute_dinobloom" / "train" / "best_attr_dinobloom.pt"
DEFAULT_IMAGE_ROOT = WBC_CV / "generated" / "det_dataset" / "images"
DEFAULT_OUT_DIR = HERE / "runs" / "classifier" / "dinobloom"
ALL_BACKENDS: tuple[Backend, ...] = ("random_forest", "xgboost", "lightgbm")


def _group_patients(split: str, image_root: Path, cv_root: Path) -> list[tuple[str, list[str]]]:
    derived = discover_lld_split_from_cv(cv_root)
    patient_ids = derived.get(split, [])
    if not patient_ids:
        raise SystemExit(f"No patient IDs for split={split!r} under {cv_root}")

    image_dir = image_root / split
    if not image_dir.is_dir():
        raise SystemExit(f"Image directory not found: {image_dir}")

    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    by_patient: dict[str, list[str]] = {pid: [] for pid in patient_ids}
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() not in suffixes:
            continue
        pid = path.stem.split("_")[0]
        if pid in by_patient:
            by_patient[pid].append(str(path.resolve()))

    patients = [(pid, paths) for pid in patient_ids if (paths := by_patient.get(pid, []))]
    if not patients:
        raise SystemExit(f"No images matched split={split} in {image_dir}")
    return patients


def _build_detector(
    *,
    yolo_weights: Path,
    dinobloom_weights: str | Path,
    dinobloom_attr_weights: Path,
    dinobloom_variant: str,
    device: str,
    conf_threshold: float,
) -> TwoStageDetectionAgent:
    localizer = YOLOv11Localizer(
        weights_path=str(yolo_weights),
        device=device,
        conf_threshold=conf_threshold,
    )
    attr_clf = DinoBloomAttributeClassifier(
        weights_path=str(dinobloom_weights),
        attr_probes_path=str(dinobloom_attr_weights),
        variant=dinobloom_variant,
        attr_mode="probes",
        device=device,
    )
    return TwoStageDetectionAgent(
        localizer=localizer,
        attribute_classifier=attr_clf,
        attribute_head_name="DinoBloom MLP",
    )


def _patient_feature_row(detection_result: DetectionResult, findings) -> dict[str, float]:
    from agentic_hematology.leukemia_features import build_feature_row_from_findings

    return build_feature_row_from_findings(
        findings,
        detection_result=detection_result,
    )


def _extract_features(
    detector: TwoStageDetectionAgent,
    patients: list[tuple[str, list[str]]],
    labels: dict[str, str],
) -> tuple[list[str], list[dict[str, float]], list[str]]:
    rows: list[dict[str, float]] = []
    patient_ids: list[str] = []
    y_labels: list[str] = []

    for pid, image_paths in patients:
        label = labels.get(pid)
        if not label:
            print(f"  Skipping patient {pid}: no diagnosis label in image filenames", file=sys.stderr)
            continue
        print(f"  Detecting patient {pid} ({len(image_paths)} images)...", flush=True)
        detection_result = detector.detect(pid, image_paths)
        findings = aggregate(detection_result)
        rows.append(_patient_feature_row(detection_result, findings))
        patient_ids.append(pid)
        y_labels.append(label)

    return patient_ids, rows, y_labels


def _rows_to_dataframe(
    patient_ids: list[str],
    rows: list[dict[str, float]],
    y_labels: list[str],
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feature_names = sorted({key for row in rows for key in row})
    X = pd.DataFrame(
        [[row.get(name, 0.0) for name in feature_names] for row in rows],
        columns=feature_names,
        index=patient_ids,
    )
    y = pd.Series(y_labels, index=patient_ids, name="metadata_filename_diagnosis")
    return X, y, feature_names


def _save_feature_cache(
    path: Path,
    *,
    patient_ids: list[str],
    rows: list[dict[str, float]],
    y_labels: list[str],
    meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "patient_ids": patient_ids,
        "rows": rows,
        "labels": y_labels,
        "meta": meta,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_feature_cache(path: Path) -> tuple[list[str], list[dict[str, float]], list[str], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["patient_ids"], payload["rows"], payload["labels"], payload.get("meta", {})


def _predict_labels(clf, X: pd.DataFrame, le: LabelEncoder, backend: Backend) -> list[str]:
    raw = clf.predict(X)
    if backend == "xgboost":
        return [str(x) for x in le.inverse_transform(raw)]
    return [str(x) for x in raw]


def _predict_proba(clf, X: pd.DataFrame, le: LabelEncoder) -> list[list[float]]:
    if not hasattr(clf, "predict_proba"):
        return []
    probs = clf.predict_proba(X)
    return probs.tolist()


def _train_one_backend(
    backend: Backend,
    X: pd.DataFrame,
    y: pd.Series,
    train_ids: list[str],
    test_ids: list[str],
    out_dir: Path,
    *,
    random_state: int,
    meta_base: dict,
) -> dict:
    y_train = [str(y.loc[pid]) for pid in train_ids]
    y_test = [str(y.loc[pid]) for pid in test_ids]
    feature_names = list(X.columns)

    clf = build_tabular_classifier(backend, random_state=random_state)
    clf, le = fit_tabular_classifier(clf, X.loc[train_ids], y_train, backend=backend)

    test_preds = _predict_labels(clf, X.loc[test_ids], le, backend)
    test_acc = accuracy_score(y_test, test_preds)
    class_names = list(le.classes_)

    print(f"\n[{backend}] test accuracy = {test_acc:.1%}  (n={len(test_ids)})")
    print(classification_report(y_test, test_preds, labels=class_names, zero_division=0))

    model_stem = f"leukemia_{backend}"
    model_path = out_dir / f"{model_stem}.pkl"
    meta_path = out_dir / f"{model_stem}_meta.json"
    pred_path = out_dir / f"{model_stem}_predictions.json"

    with open(model_path, "wb") as handle:
        pickle.dump(clf, handle)

    predictions = {}
    prob_rows = _predict_proba(clf, X.loc[test_ids], le)
    for idx, pid in enumerate(test_ids):
        prob_map = (
            {cls: float(prob_rows[idx][j]) for j, cls in enumerate(class_names)}
            if prob_rows
            else {}
        )
        predictions[pid] = {
            "patient_id": pid,
            "true_label": y.loc[pid],
            "predicted_class": test_preds[idx],
            "confidence": float(max(prob_map.values()) if prob_map else 0.0),
            "class_probabilities": prob_map,
            "split": "test",
        }

    summary = {
        "backend": backend,
        "accuracy": float(test_acc),
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "classification_report": classification_report(
            y_test, test_preds, labels=class_names, zero_division=0
        ),
        "confusion_matrix_labels": class_names,
        "confusion_matrix": confusion_matrix(y_test, test_preds, labels=class_names).tolist(),
        "predictions": predictions,
        "feature_importances": native_feature_importances(clf, feature_names),
    }

    meta = {
        **meta_base,
        "classifier_backend": backend,
        "feature_source": "detection_pipeline",
        "feature_names": feature_names,
        "classes": class_names,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "test_evaluation": summary,
        "split": {"train": train_ids, "test": test_ids},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    pred_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    print(f"  Saved {model_path}")
    print(f"  Saved {meta_path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train RF/XGBoost/LightGBM on YOLO + DinoBloom + MLP patient features."
    )
    ap.add_argument("--cv-root", type=Path, default=WBC_CV)
    ap.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    ap.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO)
    ap.add_argument("--dinobloom-weights", type=Path, default=DEFAULT_DINOBLOOM_WEIGHTS)
    ap.add_argument("--dinobloom-attr-weights", type=Path, default=DEFAULT_DINOBLOOM_ATTR)
    ap.add_argument("--dinobloom-variant", choices=["s", "b", "l", "g"], default="l")
    ap.add_argument("--conf-threshold", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--backend", choices=(*ALL_BACKENDS, "all"), default="all")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--features-cache",
        type=Path,
        default=None,
        help="Load precomputed detection features (skip running YOLO + attribute head).",
    )
    ap.add_argument(
        "--write-features-cache",
        type=Path,
        default=None,
        help="Save detection-derived features to JSON after running the pipeline.",
    )
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--folds-json", type=Path, default=HERE / "wbc_unified" / "config" / "cv_folds.json")
    ap.add_argument("--exclude", nargs="*", default=["24"])
    args = ap.parse_args()

    if not args.yolo_weights.is_file():
        raise SystemExit(f"YOLO weights not found: {args.yolo_weights}")
    if not args.dinobloom_attr_weights.is_file():
        raise SystemExit(
            f"DinoBloom attribute MLP not found: {args.dinobloom_attr_weights}\n"
            "Train it first with wbc_unified/cv/train_dinobloom_attributes_torch.py"
        )
    if not args.dinobloom_weights.is_file():
        raise SystemExit(f"DinoBloom backbone weights not found: {args.dinobloom_weights}")

    labels = discover_patient_labels_from_cv(args.cv_root, image_root=args.image_root)
    if not labels:
        raise SystemExit(
            f"No diagnosis labels found under {args.image_root}/{{train,test}}. "
            "Expected filenames like {{patient}}_*_{{ALL|AML|...}}.png"
        )

    split = discover_lld_split_from_cv(args.cv_root)
    train_ids = [pid for pid in split.get("train", []) if pid in labels]
    test_ids = [pid for pid in split.get("test", []) if pid in labels]
    if args.fold is not None:
        test_set = set(json.loads(args.folds_json.read_text())["folds"][args.fold])
        everyone = [pid for pid in labels if pid not in args.exclude]
        train_ids = [p for p in everyone if p not in test_set]
        test_ids = [p for p in everyone if p in test_set]
    if not train_ids or not test_ids:
        raise SystemExit("Could not resolve LLD train/test patient IDs.")

    cache_meta = {
        "yolo_weights": str(args.yolo_weights),
        "dinobloom_weights": str(args.dinobloom_weights),
        "dinobloom_attr_weights": str(args.dinobloom_attr_weights),
        "dinobloom_variant": args.dinobloom_variant,
        "conf_threshold": args.conf_threshold,
        "feature_source": "detection_pipeline",
    }

    if args.features_cache and args.features_cache.is_file():
        print(f"Loading cached detection features from {args.features_cache}")
        patient_ids, rows, y_labels, cache_meta = _load_feature_cache(args.features_cache)
    else:
        train_patients = _group_patients("train", args.image_root, args.cv_root)
        test_patients = _group_patients("test", args.image_root, args.cv_root)
        all_patients = train_patients + test_patients

        print(
            f"Building detector: YOLO + DinoBloom-{args.dinobloom_variant.upper()} + MLP "
            f"({args.dinobloom_attr_weights.name})"
        )
        detector = _build_detector(
            yolo_weights=args.yolo_weights,
            dinobloom_weights=args.dinobloom_weights,
            dinobloom_attr_weights=args.dinobloom_attr_weights,
            dinobloom_variant=args.dinobloom_variant,
            device=args.device,
            conf_threshold=args.conf_threshold,
        )
        print(f"Extracting features via detection + attribute classification for {len(all_patients)} patients...")
        patient_ids, rows, y_labels = _extract_features(detector, all_patients, labels)

        write_cache = args.write_features_cache or args.features_cache
        if write_cache:
            _save_feature_cache(
                write_cache,
                patient_ids=patient_ids,
                rows=rows,
                y_labels=y_labels,
                meta=cache_meta,
            )
            print(f"Wrote feature cache: {write_cache}")

    X, y, feature_names = _rows_to_dataframe(patient_ids, rows, y_labels)
    train_ids = [pid for pid in train_ids if pid in X.index]
    test_ids = [pid for pid in test_ids if pid in X.index]

    print(f"Patients: {len(X.index)}  features: {len(feature_names)}")
    print(f"Split: train={len(train_ids)}  test={len(test_ids)}")
    print(f"Train labels: {y.loc[train_ids].value_counts().to_dict()}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_base = {
        **cache_meta,
        "label_source": "lld_image_filenames",
        "image_root": str(args.image_root),
        "attribute_head": "dinobloom_mlp",
    }

    backends = ALL_BACKENDS if args.backend == "all" else (args.backend,)
    results = {}
    for backend in backends:
        backend_out = args.out_dir / backend
        backend_out.mkdir(parents=True, exist_ok=True)
        results[backend] = _train_one_backend(
            backend,
            X,
            y,
            train_ids,
            test_ids,
            backend_out,
            random_state=args.random_state,
            meta_base=meta_base,
        )

    summary_path = args.out_dir / "training_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote summary: {summary_path}")


if __name__ == "__main__":
    main()
