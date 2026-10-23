#!/usr/bin/env python3
"""Pool camera_<COND>/fold*/predictions.json into accuracy / macro-F1 per test camera.

    python summarize_patient_clf_cameras.py runs/classifier/camera_C2 runs/classifier/camera_C1C2
Writes camera_comparison.json next to the first argument's parent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sklearn.metrics import f1_score


def pooled(cond_dir: Path) -> dict:
    by_cam: dict[str, tuple[list, list]] = {}
    for f in sorted(cond_dir.glob("fold*/predictions.json")):
        for v in json.loads(f.read_text())["predictions"].values():
            for cam, probs in v["by_camera"].items():
                t, p = by_cam.setdefault(cam, ([], []))
                t.append(v["true"])
                p.append(max(probs, key=probs.get))
    out = {}
    for cam, (t, p) in sorted(by_cam.items()):
        out[cam] = {"n": len(t), "correct": sum(a == b for a, b in zip(t, p)),
                    "accuracy": sum(a == b for a, b in zip(t, p)) / len(t),
                    "macro_f1": float(f1_score(t, p, average="macro", zero_division=0))}
    return out


if __name__ == "__main__":
    res = {Path(d).name: pooled(Path(d)) for d in sys.argv[1:]}
    for cond, cams in res.items():
        for cam, m in cams.items():
            print(f"train {cond:12s} test {cam}: {m['correct']}/{m['n']}  acc {100 * m['accuracy']:.1f}  "
                  f"macro-F1 {m['macro_f1']:.3f}")
    out = Path(sys.argv[1]).parent / "camera_comparison.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"Saved {out}")