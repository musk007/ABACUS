#!/usr/bin/env python3
"""Patient-level subtype classifier: training cameras x test camera.

Reuses train_leukemia_from_dinobloom.py's detection-based feature extraction
(fold YOLO + DinoBloom attribute probes -> aggregate -> patient features),
run separately on each camera's images. One feature row per patient x camera.
The random forest is trained on the fold's training patients from
--train-domains and evaluated on the fold's held-out patients for every camera.

Per fold (from agentic_hematology/):
  python train_patient_clf_cameras.py --fold 0 \
    --yolo-weights ... --dinobloom-attr-weights ... \
    --image-root C2=wbc_unified/cv/generated_C2_f/det_dataset/images \
                 C1=wbc_unified/cv/generated_C1_f/det_dataset/images \
    --cache-dir runs/classifier/camera_C1C2/fold0 --train-domains C2 C1 \
    --out runs/classifier/camera_C1C2/fold0/predictions.json
Feature caches (<cache-dir>/features_<CAM>.json) are reused if present;
--cache CAM=path reuses an existing cache (it must come from the same probes).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from train_leukemia_from_dinobloom import (  # noqa: E402
    DEFAULT_DINOBLOOM_WEIGHTS, WBC_CV,
    _build_detector, _extract_features, _load_feature_cache, _save_feature_cache,
)
from agentic_hematology.leukemia_features import discover_patient_labels_from_cv  # noqa: E402
from agentic_hematology.tabular_classifier import build_tabular_classifier, fit_tabular_classifier  # noqa: E402


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _all_patients(root: Path) -> list[tuple[str, list[str]]]:
    """All patients under root/{train,test}, whatever split folder they sit in.

    The pipeline's own grouping takes the patient split from the C2 label folders;
    a camera whose images are split differently would silently lose patients.
    """
    by_patient: dict[str, list[str]] = {}
    for split in ("train", "test"):
        d = root / split
        if d.is_dir():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in IMAGE_SUFFIXES:
                    by_patient.setdefault(p.stem.split("_")[0], []).append(str(p.resolve()))
    return sorted(by_patient.items(), key=lambda kv: (not kv[0].isdigit(), int(kv[0]) if kv[0].isdigit() else 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--yolo-weights", type=Path, required=True)
    ap.add_argument("--dinobloom-attr-weights", type=Path, required=True)
    ap.add_argument("--dinobloom-weights", type=Path, default=DEFAULT_DINOBLOOM_WEIGHTS)
    ap.add_argument("--dinobloom-variant", default="l")
    ap.add_argument("--image-root", nargs="+", required=True, help="CAM=path/to/det_dataset/images")
    ap.add_argument("--cache", nargs="*", default=[], help="CAM=existing features.json to reuse")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--train-domains", nargs="+", required=True)
    ap.add_argument("--folds-json", type=Path, default=HERE / "wbc_unified" / "config" / "cv_folds.json")
    ap.add_argument("--exclude", nargs="*", default=["24"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conf-threshold", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    roots = dict(s.split("=", 1) for s in args.image_root)
    given = dict(s.split("=", 1) for s in args.cache)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    detector = None
    feats: dict[str, dict[str, dict]] = {}
    for cam, root in roots.items():
        cpath = Path(given.get(cam, args.cache_dir / f"features_{cam}.json"))
        if cpath.is_file():
            pids, rows, _, _ = _load_feature_cache(cpath)
            print(f"[{cam}] loaded {len(pids)} patients from {cpath}")
        else:
            if detector is None:
                detector = _build_detector(
                    yolo_weights=args.yolo_weights, dinobloom_weights=args.dinobloom_weights,
                    dinobloom_attr_weights=args.dinobloom_attr_weights,
                    dinobloom_variant=args.dinobloom_variant, device=args.device,
                    conf_threshold=args.conf_threshold)
            root = Path(root)
            labels = discover_patient_labels_from_cv(WBC_CV, image_root=root)
            patients = _all_patients(root)
            if not patients:
                raise SystemExit(f"No images found under {root}/{{train,test}}")
            print(f"[{cam}] extracting features for {len(patients)} patients from {root}")
            pids, rows, ys = _extract_features(detector, patients, labels)
            _save_feature_cache(cpath, patient_ids=pids, rows=rows, y_labels=ys,
                                meta={"camera": cam, "image_root": str(root),
                                      "yolo_weights": str(args.yolo_weights),
                                      "dinobloom_attr_weights": str(args.dinobloom_attr_weights)})
        feats[cam] = {str(p): r for p, r in zip(pids, rows)}

    cv = json.loads(args.folds_json.read_text())
    dx, held_out = cv["diagnosis"], set(cv["folds"][args.fold])
    keys = [(p, c) for c, fc in feats.items() for p in fc if p in dx and p not in args.exclude]
    names = sorted({n for p, c in keys for n in feats[c][p]})
    X = pd.DataFrame([[feats[c][p].get(n, 0.0) for n in names] for p, c in keys], columns=names)
    tr = [i for i, (p, c) in enumerate(keys) if p not in held_out and c in args.train_domains]
    te = [i for i, (p, c) in enumerate(keys) if p in held_out]

    clf = build_tabular_classifier("random_forest", random_state=args.random_state)
    clf, _ = fit_tabular_classifier(clf, X.iloc[tr], [dx[keys[i][0]] for i in tr], backend="random_forest")
    proba = clf.predict_proba(X.iloc[te])
    classes = [str(c) for c in clf.classes_]

    preds: dict[str, dict] = {}
    for j, i in enumerate(te):
        p, c = keys[i]
        preds.setdefault(p, {"true": dx[p], "by_camera": {}})["by_camera"][c] = \
            {k: float(v) for k, v in zip(classes, proba[j])}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "fold": args.fold, "train_domains": args.train_domains, "n_train_rows": len(tr),
        "classes": classes, "dinobloom_attr_weights": str(args.dinobloom_attr_weights),
        "predictions": preds}, indent=2))
    for cam in roots:
        rs = [(v["true"], max(v["by_camera"][cam], key=v["by_camera"][cam].get))
              for v in preds.values() if cam in v["by_camera"]]
        print(f"fold {args.fold} | train {'+'.join(args.train_domains)} | test {cam}: "
              f"{sum(t == p for t, p in rs)}/{len(rs)} correct")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()