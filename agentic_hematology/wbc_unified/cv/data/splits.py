"""Patient-grouped 5-fold CV over ALL LLD patients (train + test dirs).

Every patient is held out exactly once. Folds are stratified by diagnosis and
greedily balanced by annotated cell count. Built once and frozen to
configs/cv_folds.json so every module uses identical folds.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FOLDS_JSON = ROOT / "configs" / "cv_folds.json"
N_FOLDS = 5


def _scan_patients(data_root: Path) -> tuple[dict[str, str], Counter]:
    """patient -> diagnosis (last filename field), patient -> annotated cell count."""
    dx: dict[str, str] = {}
    cells: Counter = Counter()
    for split in ("train", "test"):
        for lb in (data_root / "attributes" / split).glob("*.txt"):
            if lb.name.startswith("."):
                continue
            parts = lb.stem.split("_")
            pid, d = parts[0], parts[-1]
            if dx.setdefault(pid, d) != d:
                raise ValueError(f"patient {pid} has two diagnoses: {dx[pid]} vs {d}")
            cells[pid] += sum(1 for line in lb.read_text().splitlines() if line.strip())
    return dx, cells


def build_folds(data_root: Path, n_folds: int = N_FOLDS) -> list[list[str]]:
    dx, cells = _scan_patients(Path(data_root))
    by_dx: dict[str, list[str]] = defaultdict(list)
    for pid, d in dx.items():
        by_dx[d].append(pid)

    folds: list[list[str]] = [[] for _ in range(n_folds)]
    total = [0] * n_folds
    # rarest diagnosis first, largest patients first -> rare classes spread across folds
    for d in sorted(by_dx, key=lambda d: len(by_dx[d])):
        per_dx = [0] * n_folds
        for pid in sorted(by_dx[d], key=lambda p: (-cells[p], int(p) if p.isdigit() else p)):
            k = min(range(n_folds), key=lambda i: (per_dx[i], total[i]))
            folds[k].append(pid)
            per_dx[k] += 1
            total[k] += cells[pid]
    return [sorted(f, key=lambda p: int(p) if p.isdigit() else p) for f in folds]


def write_folds(data_root: Path, path: Path = FOLDS_JSON) -> list[list[str]]:
    folds = build_folds(data_root)
    dx, cells = _scan_patients(Path(data_root))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "folds": folds,
        "diagnosis": dx,
        "cells": dict(cells),
    }, indent=2))
    for k, f in enumerate(folds):
        print(f"fold{k}: n={len(f)} cells={sum(cells[p] for p in f)} "
              f"dx={dict(Counter(dx[p] for p in f))}")
    return folds


def load_folds(path: Path = FOLDS_JSON) -> list[frozenset[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing. Run: python splits.py --data-root $DATA_ROOT")
    return [frozenset(f) for f in json.loads(path.read_text())["folds"]]


def test_patients_for_fold(fold: int) -> frozenset[str]:
    return load_folds()[fold]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    write_folds(ap.parse_args().data_root)