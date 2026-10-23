"""Cell-type classification dataset (crops -> 14 LLD classes, or 6 clinical groups).

Reads ``celltype_manifest.csv``, which unlike ``attr_manifest.csv`` contains
EVERY cell including the class-0 'None' artefacts. That matters: 'None' is a
class the classifier must be able to predict, since ``aggregate()`` relies on it
to exclude artefacts from the differential.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.cell_dataset import IMAGENET_MEAN, IMAGENET_STD, load_manifest
from utils.labels import crop_with_padding

# LLD detection classes, index == class_id in the manifest.
CLASS_NAMES = [
    "None",                    # 0
    "Myeloblast",              # 1
    "Lymphoblast",             # 2
    "Neutrophil",              # 3
    "Atypical lymphocyte",     # 4
    "Promonocyte",             # 5
    "Monoblast",               # 6
    "Lymphocyte",              # 7
    "Myelocyte",               # 8
    "Abnormal promyelocyte",   # 9
    "Monocyte",                # 10
    "Metamyelocyte",           # 11
    "Eosinophil",              # 12
    "Basophil",                # 13
]

# Coarse targets matching CLINICAL_GROUPS in leukemia_features.py. Useful
# because several fine classes have <120 training cells (Basophil has 16).
GROUP_NAMES = [
    "None",                    # 0
    "blasts",                  # 1
    "intermediate_myeloid",    # 2
    "mature_granulocytes",     # 3
    "lymphoid",                # 4
    "monocytic",               # 5
]

CLASS_ID_TO_GROUP = {
    0: 0,                                   # None
    1: 1, 2: 1, 6: 1, 9: 1,                 # blasts
    5: 2, 8: 2, 11: 2,                      # intermediate myeloid
    3: 3, 12: 3, 13: 3,                     # mature granulocytes
    7: 4, 4: 4,                             # lymphoid
    10: 5,                                  # monocytic
}


def target_names(target: str) -> list[str]:
    if target == "group":
        return list(GROUP_NAMES)
    return list(CLASS_NAMES)


def row_label(row: dict, target: str) -> int:
    class_id = int(row["class_id"])
    if target == "group":
        return CLASS_ID_TO_GROUP[class_id]
    return class_id


class CellTypeDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path,
        split: str,
        imgsz: int = 224,
        pad: float = 0.15,
        augment: Optional[Callable] = None,
        val_patients: Iterable[str] | None = None,
        target: str = "celltype",
        domain: str | None = None,
        fold: int | None = None,
    ):
        if target not in ("celltype", "group"):
            raise ValueError(f"target must be 'celltype' or 'group', got {target!r}")
        self.rows = load_manifest(manifest_csv, split, val_patients=val_patients,
                            domain=domain, fold=fold)
        self.split = split
        self.imgsz = imgsz
        self.pad = pad
        self.augment = augment
        self.target = target
        self.class_names = target_names(target)


    def __len__(self) -> int:
        return len(self.rows)

    @property
    def labels(self) -> np.ndarray:
        """All labels, without loading images — for samplers and class weights."""
        return np.array([row_label(r, self.target) for r in self.rows], dtype=np.int64)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        r = self.rows[idx]
        img_path = Path(r["image"])
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        xywh = np.array(
            [float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])],
            dtype=np.float32,
        )
        x1, y1, x2, y2 = crop_with_padding(w, h, xywh, pad=self.pad)
        crop = image.crop((x1, y1, x2, y2)).resize((self.imgsz, self.imgsz), Image.BILINEAR)

        if self.augment is not None:
            crop = self.augment(crop)

        arr = np.asarray(crop, dtype=np.float32) / 255.0
        arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float()

        meta = {
            "image": str(img_path),
            "cell_idx": int(r["cell_idx"]),
            "class_id": int(r["class_id"]),
            "patient_id": str(r["patient_id"]),
            "domain": str(r.get("domain", "?"))
        }
        return tensor, torch.tensor(row_label(r, self.target), dtype=torch.long), meta


def compute_class_counts(
    manifest_csv: Path,
    split: str,
    *,
    val_patients: Iterable[str] | None = None,
    target: str = "celltype",
    ceil: float = 20.0,
    domain: str | None = None,
    fold: int | None = None,
) -> np.ndarray:
    rows = load_manifest(manifest_csv, split, val_patients=val_patients, domain=domain, fold=fold)
    n = len(target_names(target))
    counts = np.zeros(n, dtype=np.int64)
    for r in rows:
        counts[row_label(r, target)] += 1
    return counts


def compute_class_weights(
    manifest_csv: Path,
    split: str,
    *,
    val_patients: Iterable[str] | None = None,
    target: str = "celltype",
    ceil: float = 20.0,
    domain: str | None = None,
    fold: int | None = None,
) -> torch.Tensor:
    """Inverse-frequency class weights, clipped.

    LLD is ~134:1 between the largest and smallest class, so an uncapped
    inverse-frequency weight would let a 16-cell class dominate the loss.
    """
    counts = compute_class_counts(
        manifest_csv, split, val_patients=val_patients, target=target,
        domain=domain, fold=fold,
    )
    total = counts.sum()
    n = len(counts)
    if total == 0:
        return torch.ones(n, dtype=torch.float32)
    weights = total / (n * np.maximum(counts, 1))
    weights = np.minimum(weights, ceil)
    weights[counts == 0] = 0.0  # absent classes contribute nothing
    return torch.tensor(weights, dtype=torch.float32)


def sample_weights_for_balancing(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """Per-sample weights for WeightedRandomSampler (inverse class frequency)."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    per_class = 1.0 / counts
    return torch.tensor(per_class[labels], dtype=torch.double)