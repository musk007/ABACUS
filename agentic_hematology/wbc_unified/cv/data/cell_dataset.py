"""PyTorch dataset: cell crops + 6 binary attributes."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from data.splits import load_folds

from utils.labels import ATTR_NAMES, BINARY_ATTRS, IGNORE_ATTR, crop_with_padding, CELL_SIZE_N_CLASSES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


SIZE_IGNORE_INDEX = -100


def load_manifest(
    path: Path,
    split: str,
    *,
    val_patients: Iterable[str] | None = None,
    domain=None,
    fold: int | None = None,
) -> List[dict]:
    """Rows for one split.

    ``split="val"`` reads the manifest's *train* rows and keeps only
    ``val_patients``; ``split="train"`` reads them and drops those patients.
    ``split="test"`` is the untouched held-out split.
    """
    if fold is not None:
        

        test_pids = load_folds()[fold]
        rows = []
        with path.open() as f:
            for row in csv.DictReader(f):
                if domain and row.get("domain") != domain:
                    continue
                in_test = row["patient_id"] in test_pids
                if (split == "test") == in_test:
                    rows.append(row)
        return rows

    val_patients = set(val_patients or ())
    source_split = "train" if split == "val" else split

    rows: List[dict] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["split"] != source_split:
                continue
            if domain and row.get("domain") != domain:
                continue
            if val_patients and source_split == "train":
                in_val = row["patient_id"] in val_patients
                if split == "val" and not in_val:
                    continue
                if split == "train" and in_val:
                    continue
            rows.append(row)

    if split == "val" and val_patients and not rows:
        raise ValueError(
            f"No rows matched val_patients={sorted(val_patients)} in {path}. "
            "Check the patient_id column exists (regenerate the manifest)."
        )
    return rows


class CellAttributeDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path,
        split: str,
        imgsz: int = 224,
        pad: float = 0.15,
        augment: Optional[Callable] = None,
        val_patients: Iterable[str] | None = None,
        fold: int | None = None,
    ):
        self.rows = load_manifest(manifest_csv, split, val_patients=val_patients, fold=fold)
        self.split = split
        self.imgsz = imgsz
        self.pad = pad
        self.augment = augment
        self.attr_names = ATTR_NAMES

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        r = self.rows[idx]
        img_path = Path(r["image"])
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        xywh = np.array([float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])], dtype=np.float32)
        x1, y1, x2, y2 = crop_with_padding(w, h, xywh, pad=self.pad)
        crop = image.crop((x1, y1, x2, y2)).resize((self.imgsz, self.imgsz), Image.BILINEAR)

        if self.augment is not None:
            crop = self.augment(crop)

        arr = np.asarray(crop, dtype=np.float32) / 255.0
        arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float()

        bin_targets = [attr_target_value(int(r[name])) for name in BINARY_ATTRS]
        size_raw = int(r["Cell_Size"])
        size_target = size_raw if 0 <= size_raw < CELL_SIZE_N_CLASSES else SIZE_IGNORE_INDEX
        meta = {"image": str(img_path), "cell_idx": int(r["cell_idx"]), "class_id": int(r["class_id"]), "patient_id": str(r["patient_id"])}
        return (
            tensor,
            torch.tensor(bin_targets, dtype=torch.float32),
            torch.tensor(size_target, dtype=torch.long),
            meta,
        )


def attr_target_value(v: int) -> float:
    """Map raw label to BCE target; -1 = masked (ignore)."""
    if v == IGNORE_ATTR or v > 1 or v < 0:
        return -1.0
    return float(v)


def compute_pos_weights(manifest_csv: Path, split: str, *, val_patients: Iterable[str] | None = None, fold: int | None = None, floor: float = 0.1, ceil: float = 10.0,) -> torch.Tensor:
    """Inverse-frequency BCE weights for the 6 binary attributes only.

    Length and order must match BINARY_ATTRS, since the result is applied to
    bin_logits. Cell_Size is categorical and handled by the separate size head.

    ``floor`` below 1.0 lets attributes whose *negative* class is the minority
    (Nuclear_Shape, Cytoplasm, Cytoplasmic_Basophilia in LLD) be rebalanced by
    down-weighting the majority positives.
    """
    rows = load_manifest(manifest_csv, split, val_patients=val_patients, fold=fold)
    counts = np.zeros((len(BINARY_ATTRS), 2), dtype=np.float64)
    for r in rows:
        for j, name in enumerate(BINARY_ATTRS):
            v = int(r[name])
            if attr_target_value(v) < 0:
                continue
            counts[j, v] += 1
    weights = []
    for j in range(len(BINARY_ATTRS)):
        neg, pos = counts[j, 0], counts[j, 1]
        if pos < 1:
            weights.append(1.0)
        else:
            weights.append(min(ceil, max(floor, neg / pos)))
    return torch.tensor(weights, dtype=torch.float32)

def compute_size_class_weights(
    manifest_csv: Path,
    split: str,
    *,
    val_patients: Iterable[str] | None = None,
    fold: int | None = None,
    ceil: float = 10.0,
) -> torch.Tensor:
    """Inverse-frequency class weights for the 3-class Cell_Size head.

    'large' is ~3.6% of LLD cells, so unweighted cross-entropy collapses to
    predicting 'medium'.
    """
    rows = load_manifest(manifest_csv, split, val_patients=val_patients, fold=fold)
    counts = np.zeros(CELL_SIZE_N_CLASSES, dtype=np.float64)
    for r in rows:
        v = int(r["Cell_Size"])
        if 0 <= v < CELL_SIZE_N_CLASSES:
            counts[v] += 1
    total = counts.sum()
    if total == 0:
        return torch.ones(CELL_SIZE_N_CLASSES, dtype=torch.float32)
    weights = total / (CELL_SIZE_N_CLASSES * np.maximum(counts, 1.0))
    weights = np.minimum(weights, ceil)
    return torch.tensor(weights, dtype=torch.float32)