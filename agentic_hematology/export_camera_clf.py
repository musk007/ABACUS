#!/usr/bin/env python3
"""Train and export the per-fold patient classifier from cached camera features.

Reuses the feature caches written by train_patient_clf_cameras.py, so no GPU is
needed. Writes leukemia_random_forest.pkl + _meta.json per fold, in the layout
run_orchestrator.py expects.

    python export_camera_clf.py --cache-dir runs/classifier/camera_C1C2 \
      --train-domains C2 C1 --out-dir runs/classifier/camera_C1C2_models
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from train_leukemia_from_dinobloom import _load_feature_cache  # noqa: E402
from agentic_hematology.tabular_classifier import (  # noqa: E402
    build_tabular_classifier, fit_tabular_classifier, native_feature_importances,
)

ap = argparse.ArgumentParser()
ap.add_argument("--cache-dir", type=Path, required=True, help="dir with fold*/features_<CAM>.json")
ap.add_argument("--train-domains", nargs="+", required=True)
ap.add_argument("--out-dir", type=Path, required=True)
ap.add_argument("--folds-json", type=Path, default=HERE / "wbc_unified" / "config" / "cv_folds.json")
ap.add_argument("--exclude", nargs="*", default=["24"])
ap.add_argument("--random-state", type=int, default=42)
args = ap.parse_args()

cv = json.loads(args.folds_json.read_text())
dx = cv["diagnosis"]

for fold, held_out in enumerate(cv["folds"]):
    fold_dir = args.cache_dir / f"fold{fold}"
    feats: dict[str, dict[str, dict]] = {}
    for cache in sorted(fold_dir.glob("features_*.json")):
        cam = cache.stem.split("_", 1)[1]
        pids, rows, _, _ = _load_feature_cache(cache)
        feats[cam] = {str(p): r for p, r in zip(pids, rows)}
    if not feats:
        raise SystemExit(f"No feature caches in {fold_dir}")

    keys = [(p, c) for c, fc in feats.items() for p in fc if p in dx and p not in args.exclude]
    names = sorted({n for p, c in keys for n in feats[c][p]})
    X = pd.DataFrame([[feats[c][p].get(n, 0.0) for n in names] for p, c in keys], columns=names)
    tr = [i for i, (p, c) in enumerate(keys) if p not in set(held_out) and c in args.train_domains]

    clf = build_tabular_classifier("random_forest", random_state=args.random_state)
    clf, _ = fit_tabular_classifier(clf, X.iloc[tr], [dx[keys[i][0]] for i in tr], backend="random_forest")

    out = args.out_dir / f"fold{fold}" / "random_forest"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "leukemia_random_forest.pkl", "wb") as f:
        pickle.dump(clf, f)
    (out / "leukemia_random_forest_meta.json").write_text(json.dumps({
        "classifier_backend": "random_forest",
        "feature_source": "detection_pipeline",
        "feature_names": names,
        "classes": [str(c) for c in clf.classes_],
        "train_domains": args.train_domains,
        "n_train": len(tr),
        "cache_dir": str(fold_dir),
        "feature_importances": native_feature_importances(clf, names),
    }, indent=2))
    print(f"fold {fold}: trained on {len(tr)} rows ({'+'.join(args.train_domains)}), "
          f"{len(names)} features -> {out}")