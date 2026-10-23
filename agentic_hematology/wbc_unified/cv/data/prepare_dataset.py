#!/usr/bin/env python3
"""Build YOLO detection dataset + attribute cell manifest from LLD Organized."""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.labels import (  # noqa: E402
    ATTR_NAMES,
    BINARY_ATTRS,
    CELL_SIZE_IGNORE,
    IGNORE_ATTR,
    is_unannotated_row,
    parse_label_file,
    write_det_label,
)

CLASS_NAMES = [
    "None",
    "Myeloblast",
    "Lymphoblast",
    "Neutrophil",
    "Atypical lymphocyte",
    "Promonocyte",
    "Monoblast",
    "Lymphocyte",
    "Myelocyte",
    "Abnormal promyelocyte",
    "Monocyte",
    "Metamyelocyte",
    "Eosinophil",
    "Basophil",
]

MANIFEST_FIELDS = [
    "split",
    "image",
    "label_file",
    "patient_id",
    "cell_idx",
    "class_id",
    "x",
    "y",
    "w",
    "h",
] + ATTR_NAMES


def patient_id_from_stem(stem: str) -> str:
    """LLD filenames are ``{patient}_{...}_{DIAGNOSIS}``."""
    return stem.split("_")[0]

def resolve_image_path(src_img: Path, stem: str) -> Path:
    """Return expected image path even if not mounted on this node."""
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        p = src_img / f"{stem}{ext}"
        if p.is_file():
            return p
    return src_img / f"{stem}.png"


def remove_mac_resource_forks(img_dir: Path) -> int:
    """Remove AppleDouble resource-fork files (._*) that break YOLO label parsing."""
    if not img_dir.is_dir():
        return 0
    removed = 0
    for p in img_dir.glob("._*"):
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def install_image(src: Path, dst: Path, mode: str = "auto") -> str:
    """
    Place image under det_dataset/images.
    Prefer hardlink (DDP-safe, no duplicate bytes); fall back to copy; symlink last.
    """
    src = src.resolve()
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode in ("auto", "hardlink"):
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise

    if mode in ("auto", "copy"):
        shutil.copy2(src, dst)
        return "copy"

    dst.symlink_to(src)
    return "symlink"


def ensure_split_image_dir(dst_img: Path) -> None:
    """Remove legacy whole-directory symlinks (they make YOLO read NFS labels/)."""
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    if dst_img.is_symlink() or dst_img.is_file():
        dst_img.unlink()
    if not dst_img.is_dir():
        dst_img.mkdir(parents=True, exist_ok=True)


def link_split_images(data_root: Path, split: str, out_root: Path, image_mode: str = "auto") -> int:
    """
    Create per-file image entries under det_dataset/images/{split}.
    This is important: YOLO derives label paths from image paths.
    If we symlink the whole source images dir, labels are resolved to source labels/.
    """
    src_img = data_root / "images" / split
    src_attr = data_root / "attributes" / split
    dst_img = out_root / "images" / split
    ensure_split_image_dir(dst_img)

    # Clear old links/files to avoid stale mismatches across reruns.
    for p in dst_img.glob("*"):
        try:
            if p.is_symlink() or p.is_file():
                p.unlink()
        except OSError:
            pass

    if not src_img.is_dir() or not src_attr.is_dir():
        print(f"WARNING: missing split dirs for {split}: {src_img} / {src_attr}")
        return 0

    removed = remove_mac_resource_forks(src_img)
    if removed:
        print(f"  removed {removed} macOS ._* files from {src_img}")

    linked = 0
    modes: dict[str, int] = {}
    for lb_path in sorted(src_attr.glob("*.txt")):
        stem = lb_path.stem
        img_path = resolve_image_path(src_img, stem)
        if not img_path.exists():
            continue
        target = dst_img / img_path.name
        how = install_image(img_path, target, mode=image_mode)
        modes[how] = modes.get(how, 0) + 1
        linked += 1
    if modes:
        print(f"  images/{split}: {linked} files ({', '.join(f'{k}={v}' for k, v in sorted(modes.items()))})")
    return linked


def process_split(
    data_root: Path,
    split: str,
    out_root: Path,
    parse_stats: dict,
) -> tuple[list[dict], list[dict], set[str]]:
    """Write YOLO detection labels; return (attribute rows, celltype rows, patients).

    Single pass: label files are parsed once and used for the detection labels,
    the attribute manifest (annotated cells only) and the cell-type manifest
    (all cells, including class-0 'None' artefacts).
    """
    src_attr = data_root / "attributes" / split
    src_img = data_root / "images" / split
    out_lbl = out_root / "labels" / split
    rows_csv: list[dict] = []
    celltype_rows: list[dict] = []
    patients: set[str] = set()

    if not src_attr.is_dir():
        print(f"WARNING: no attributes/{split} at {src_attr}")
        return rows_csv, celltype_rows, patients

    n_unannotated = 0
    for lb_path in sorted(src_attr.glob("*.txt")):
        stem = lb_path.stem
        img_path = resolve_image_path(src_img, stem)
        lb = parse_label_file(lb_path, stats=parse_stats)
        write_det_label(out_lbl / f"{stem}.txt", lb)
        if lb.size == 0:
            continue
        patients.add(patient_id_from_stem(stem))

        for i in range(lb.shape[0]):
            attrs = lb[i, 5 : 5 + len(ATTR_NAMES)]
            row = {
                "split": split,
                "image": str(img_path),
                "label_file": str(lb_path),
                "patient_id": patient_id_from_stem(stem),
                "cell_idx": i,
                "class_id": int(lb[i, 0]),
                "x": float(lb[i, 1]),
                "y": float(lb[i, 2]),
                "w": float(lb[i, 3]),
                "h": float(lb[i, 4]),
                **{name: int(attrs[j]) for j, name in enumerate(ATTR_NAMES)},
            }
            # Every cell goes to the cell-type manifest, artefacts included —
            # 'None' is a class the classifier must be able to predict.
            celltype_rows.append(row)
            # Attribute training keeps only cells with binary annotations.
            # Cell_Size is excluded from that test because 2 = large, not ignore.
            if is_unannotated_row(attrs):
                n_unannotated += 1
                continue
            rows_csv.append(row)

    print(f"  {split}: {len(rows_csv)} annotated cells, "
          f"{len(celltype_rows)} total cells")
    return rows_csv, celltype_rows, patients


def export_cv_folds(det_root: Path, image_mode: str) -> None:
    """Fold k: its patients -> images/test, all others -> images/train (both source dirs)."""
    from splits import load_folds

    folds = load_folds()
    for k, test_pids in enumerate(folds):
        fold_root = det_root.parent / f"det_dataset_fold{k}"
        for sub in ("images/train", "images/test", "labels/train", "labels/test"):
            d = fold_root / sub
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
        n = {"train": 0, "test": 0}
        for src in ("train", "test"):
            for img in sorted((det_root / "images" / src).glob("*.png")):
                part = "test" if patient_id_from_stem(img.stem) in test_pids else "train"
                install_image(img, fold_root / "images" / part / img.name, mode=image_mode)
                lbl = det_root / "labels" / src / f"{img.stem}.txt"
                if lbl.is_file():
                    install_image(lbl, fold_root / "labels" / part / lbl.name, mode=image_mode)
                n[part] += 1
        y = ROOT / "configs" / f"dataset_fold{k}.yaml"
        y.write_text(yaml.safe_dump({
            "path": str(fold_root.resolve()), "train": "images/train",
            "val": "images/test", "test": "images/test",
            "nc": len(CLASS_NAMES), "names": CLASS_NAMES}, sort_keys=False))
        print(f"  fold{k}: train={n['train']} test={n['test']} images -> {y}")

def report_attribute_distribution(rows: list[dict], split: str) -> None:
    """Print per-attribute label distribution so shifts are caught immediately."""
    split_rows = [r for r in rows if r["split"] == split]
    if not split_rows:
        return
    print(f"\n  Attribute distribution [{split}] (n={len(split_rows)}):")
    for name in ATTR_NAMES:
        counts = Counter(int(r[name]) for r in split_rows)
        if name in BINARY_ATTRS:
            neg, pos = counts.get(0, 0), counts.get(1, 0)
            ign = counts.get(IGNORE_ATTR, 0)
            ratio = (neg / pos) if pos else float("inf")
            print(
                f"    {name:24s} 0={neg:6d}  1={pos:6d}  ignore={ign:6d}  neg/pos={ratio:.2f}"
            )
        else:
            ign = counts.get(CELL_SIZE_IGNORE, 0)
            body = "  ".join(f"{k}={counts.get(k, 0)}" for k in range(3))
            print(f"    {name:24s} {body}  ignore={ign}")


def check_patient_overlap(train_patients: set[str], test_patients: set[str]) -> None:
    overlap = sorted(train_patients & test_patients)
    if overlap:
        print(
            f"\n  WARNING: {len(overlap)} patient(s) appear in BOTH train and test: "
            f"{', '.join(overlap[:10])}{' ...' if len(overlap) > 10 else ''}\n"
            "  The detector and attribute head will train on images from test "
            "patients. Re-split by patient before reporting held-out results."
        )
    else:
        print(
            f"\n  Patient split is clean: train={len(train_patients)} "
            f"test={len(test_patients)}, no overlap."
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "/home/rao.anwer/datasets/LeukemiaDataset_Organized")),
    )
    p.add_argument("--out", type=Path, default=ROOT / "generated")
    p.add_argument(
        "--image-mode",
        type=str,
        default=os.environ.get("DET_IMAGE_MODE", "auto"),
        choices=("auto", "hardlink", "copy", "symlink"),
        help="How to populate det_dataset/images (auto=hardlink then copy; avoid symlink for DDP)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any label row is short/unparsable or patients overlap splits.",
    )
    p.add_argument("--export-folds", action="store_true", help="Write det_dataset_fold{k} for 5-fold CV")
    args = p.parse_args()

    det_root = args.out / "det_dataset"
    det_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        remove_mac_resource_forks(args.data_root / "images" / split)

    linked_train = link_split_images(args.data_root, "train", det_root, args.image_mode)
    linked_test = link_split_images(args.data_root, "test", det_root, args.image_mode)

    parse_stats: dict = defaultdict(int)

    rows_train, ct_train, patients_train = process_split(args.data_root, "train", det_root, parse_stats)
    rows_test, ct_test, patients_test = process_split(args.data_root, "test", det_root, parse_stats)
    all_rows = rows_train + rows_test
    all_celltype = ct_train + ct_test

    manifest = args.out / "attr_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    celltype_manifest = args.out / "celltype_manifest.csv"
    with celltype_manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(all_celltype)

    yaml_path = ROOT / "configs" / "dataset.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "path": str(det_root.resolve()),
        "train": "images/train",
        "val": "images/test",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
        "data_root": str(args.data_root.resolve()),
        "attr_manifest": str(manifest.resolve()),
        "celltype_manifest": str(celltype_manifest.resolve()),
        "attr_names": ATTR_NAMES,
        "binary_attrs": BINARY_ATTRS,
    }
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    

    # Count images
    n_img_train = len(list((det_root / "images" / "train").glob("*"))) if (det_root / "images" / "train").exists() else 0
    n_img_test = len(list((det_root / "images" / "test").glob("*"))) if (det_root / "images" / "test").exists() else 0
    print(f"\nPrepared detection dataset: {det_root}")
    print(f"  train cells (manifest): {len(rows_train)}  test cells: {len(rows_test)}")
    print(f"  linked images: train={linked_train}, test={linked_test}")
    print(f"  image entries: train={n_img_train}, test={n_img_test}")
    print(f"  config: {yaml_path}")
    print(f"  manifest: {manifest}")

    print(
        f"\n  Label parsing: {parse_stats['files']} files, {parse_stats['rows']} rows, "
        f"short_rows={parse_stats['short_rows']}, "
        f"unparsable_rows={parse_stats['unparsable_rows']}, "
        f"clamped_attr_values={parse_stats['bad_attr_values']}"
    )
    if parse_stats["short_rows"] or parse_stats["unparsable_rows"]:
        print(
            "  NOTE: short rows were padded with ignore codes, not zeros. "
            "Inspect the source files if this count is non-zero."
        )

    report_attribute_distribution(all_rows, "train")
    report_attribute_distribution(all_rows, "test")
    check_patient_overlap(patients_train, patients_test)

    if n_img_train == 0:
        raise SystemExit(
            "No images under det_dataset/images/train. Check DATA_ROOT and re-run prepare_dataset."
        )
    for split in ("train", "test"):
        img_dir = det_root / "images" / split
        if img_dir.is_symlink():
            raise SystemExit(
                f"{img_dir} is still a directory symlink. Remove it and re-run prepare_dataset."
            )
        sample = next(img_dir.glob("*.png"), None)
        if sample is None:
            continue
        lbl = det_root / "labels" / split / f"{sample.stem}.txt"
        if not lbl.is_file():
            raise SystemExit(f"Missing detection label for sample image: {lbl}")
        text = lbl.read_text().strip()
        ncol = len(text.splitlines()[0].split()) if text else 0
        if ncol != 5:
            raise SystemExit(f"Expected 5-column YOLO labels, got {ncol} in {lbl}")
        if sample.is_symlink():
            print(f"WARNING: {sample} is still a symlink; use --image-mode copy or hardlink for stable DDP training")

    if args.strict:
        if parse_stats["short_rows"] or parse_stats["unparsable_rows"]:
            raise SystemExit("--strict: malformed label rows found.")
        if patients_train & patients_test:
            raise SystemExit("--strict: patients overlap between train and test.")

    print(f"  manifest: {manifest}")
    print(f"  celltype manifest: {celltype_manifest}")
    if args.export_folds:
        export_cv_folds(det_root, args.image_mode)

if __name__ == "__main__":
    main()
