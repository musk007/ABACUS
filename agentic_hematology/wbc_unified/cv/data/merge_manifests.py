#!/usr/bin/env python3
"""Merge the C1 and C2 cell manifests into one, adding a `domain` column.

Merges both attr_manifest.csv (attribute probes) and celltype_manifest.csv
(cell-type head). Patient IDs are shared across cameras, so fold filtering by
patient keeps both cameras of a held-out patient together.

    python merge_manifests.py --c1 generated_C1_f --c2 generated_C2_f --out generated_C1C2
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

NAMES = ("attr_manifest.csv", "celltype_manifest.csv")


def read(path: Path, domain: str) -> tuple[list[str], list[dict]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = [{**r, "domain": domain} for r in reader]
        return list(reader.fieldnames or []), rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1", type=Path, required=True, help="Folder with the C1 manifests")
    ap.add_argument("--c2", type=Path, required=True, help="Folder with the C2 manifests")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for name in NAMES:
        cols1, rows1 = read(args.c1 / name, "C1")
        cols2, rows2 = read(args.c2 / name, "C2")
        if cols1 != cols2:
            raise SystemExit(f"{name}: column mismatch\n  C1: {cols1}\n  C2: {cols2}")

        images = {r["image"] for r in rows1} & {r["image"] for r in rows2}
        if images:
            raise SystemExit(f"{name}: {len(images)} image paths appear in both cameras, e.g. {next(iter(images))}")

        rows = rows2 + rows1
        with (args.out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols1 + ["domain"])
            w.writeheader()
            w.writerows(rows)

        p1 = {r["patient_id"] for r in rows1}
        p2 = {r["patient_id"] for r in rows2}
        print(f"{name}: C1={len(rows1)} cells / {len(p1)} patients, "
              f"C2={len(rows2)} cells / {len(p2)} patients, total={len(rows)}")
        if p1 ^ p2:
            print(f"  patients in only one camera: C1-only={sorted(p1 - p2)} C2-only={sorted(p2 - p1)}")
        print(f"  split counts: {dict(Counter((r['domain'], r['split']) for r in rows))}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()