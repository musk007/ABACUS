"""Detection agent interfaces and a JSON-backed stub implementation."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .schemas import Detection, DetectionResult


LLD_CLASSES = [
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

ATTRIBUTE_VOCAB = {
    "Cell_Size": [0, 1, 2],
    "Nuclear_Chromatin": [0, 1],
    "Nuclear_Shape": [0, 1],
    "Nucleolus": [0, 1],
    "Cytoplasm": [0, 1],
    "Cytoplasmic_Basophilia": [0, 1],
    "Cytoplasmic_Vacuoles": [0, 1],
}


class BaseDetectionAgent(ABC):
    @abstractmethod
    def detect(self, case_id: str, image_paths: list[str]) -> DetectionResult:
        raise NotImplementedError


class StubDetector(BaseDetectionAgent):
    """Replay detections from a wbc_unified-style prediction JSON file."""

    def __init__(self, source: str | Path):
        self.source = Path(source)

    def detect(self, case_id: str, image_paths: list[str]) -> DetectionResult:
        payload = json.loads(self.source.read_text())
        detections: list[Detection] = []
        wanted = {str(Path(p)) for p in image_paths}
        for img_idx, item in enumerate(payload):
            image = str(item.get("image", ""))
            if wanted and image not in wanted and Path(image).name not in {Path(p).name for p in image_paths}:
                continue
            image_id = Path(image).name
            for cell_idx, cell in enumerate(item.get("cells", [])):
                attrs: dict[str, Any] = dict(cell.get("attributes", {}))
                attrs["class_id"] = cell.get("class_id")
                detections.append(
                    Detection(
                        cell_id=str(cell.get("cell_id") or f"img{img_idx:03d}_c{cell_idx:03d}"),
                        image_id=image_id,
                        bbox_xyxy=tuple(float(v) for v in cell.get("xyxy", [0, 0, 1, 1])),
                        cell_type=str(cell.get("class_name", "Unknown")),
                        objectness=float(cell.get("conf", 0.0)),
                        cell_type_prob=float(cell.get("conf", 0.0)),
                        attributes=attrs,
                        attribute_probs={
                            k: float(v) for k, v in cell.get("attributes", {}).items()
                        },
                    )
                )
        return DetectionResult(case_id=case_id, n_images=len(image_paths), detections=detections)
