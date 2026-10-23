"""
detection_agent_v2.py
=====================
Two-model detection + attribute-classification agent.

Architecture (the design you specified):

    image ──▶ YOLOv11 localizer ──▶ bbox + cell-type per cell
                                         │
                                         ▼
                         crop each cell from the image
                                         │
                                         ▼
              EfficientNet attribute classifier (per crop)
                                         │
                                         ▼
              Detection objects with bbox + cell_type + 7 attributes

This separates the two learnable components cleanly:
- `YOLOv11Localizer`     — wraps an Ultralytics YOLOv11 model. Produces
                            bounding boxes + cell-type labels + objectness.
- `EfficientNetAttributeClassifier` — wraps a (multi-head) EfficientNet that
                            takes a cell crop and predicts the 7 morphologic
                            attributes. One classification head per attribute.
- `TwoStageDetectionAgent` — orchestrates the two: localize, crop, classify,
                            assemble `Detection` objects.

All three conform to the same `DetectionResult` contract the rest of the
pipeline already consumes, so nothing downstream changes.

Backends degrade gracefully: if torch / ultralytics are not installed, the
classes raise a clear ImportError only when instantiated, so the module
still imports for testing with the StubDetector.
"""

from __future__ import annotations

import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .detection_agent import BaseDetectionAgent, LLD_CLASSES
from .schemas import Detection, DetectionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attribute schema — the 6 binary LLD morphologic attributes used by
# wbc_unified/cv/models/attribute_net.py.
# ---------------------------------------------------------------------------

ATTRIBUTE_ORDER = [
    "Cell_Size",
    "Nuclear_Chromatin",
    "Nuclear_Shape",
    "Nucleolus",
    "Cytoplasm",
    "Cytoplasmic_Basophilia",
    "Cytoplasmic_Vacuoles",
]
CELL_SIZE_ATTR = "Cell_Size"
CELL_SIZE_STATES = ("small", "medium", "large")
BINARY_ATTRIBUTE_ORDER = [a for a in ATTRIBUTE_ORDER if a != CELL_SIZE_ATTR]

@runtime_checkable
class CropAttributeClassifier(Protocol):
    """Stage-2 head: always receives YOLO cell crops, never full PBS tiles."""

    def classify_crops(
        self, crops: list[Any], **kwargs: Any
    ) -> list[tuple[dict[str, float], dict[str, float], str | None, float | None]]: ...


def _resolve_device(device: str | None) -> tuple[str, str]:
    """Return (torch_device, ultralytics_device), falling back if CUDA is unusable."""
    try:
        import torch  # type: ignore
    except ImportError:
        return "cpu", "cpu"

    requested = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    requested = str(requested)
    wants_cuda = requested.isdigit() or requested.startswith("cuda")
    if wants_cuda and not torch.cuda.is_available():
        print(
            "WARNING: CUDA was requested but torch.cuda.is_available() is false; "
            "falling back to CPU. Check the node driver vs installed Torch CUDA build.",
            file=sys.stderr,
        )
        return "cpu", "cpu"
    if requested.isdigit():
        return f"cuda:{requested}", requested
    return requested, requested


# ---------------------------------------------------------------------------
# Stage 1 — YOLOv11 localizer
# ---------------------------------------------------------------------------

class YOLOv11Localizer:
    """
    Wraps an Ultralytics YOLOv11 model trained on the LLD 14-class schema.

    Produces, per image, a list of (bbox_xyxy, cell_type, objectness).
    Cell-type comes from the YOLO classification head; if you trained YOLO
    only for localization (single 'cell' class), set `class_agnostic=True`
    and let EfficientNet handle typing too (not the default).
    """

    def __init__(
        self,
        weights_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        image_size: int = 640,
        batch_size: int = 1,
        half_precision: bool = True,
        device: str | None = None,
        class_agnostic: bool = False,
    ):
        weights = Path(weights_path)
        if not weights.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")

        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Install ultralytics for YOLOv11: `pip install ultralytics`"
            ) from e
        logger.info(
            "Loading YOLOv11 localizer from %s (conf=%.2f, iou=%.2f, imgsz=%d)",
            weights,
            conf_threshold,
            iou_threshold,
            image_size,
        )
        self.weights_path = str(weights.resolve())
        self.model = YOLO(self.weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.batch_size = batch_size
        self.half_precision = half_precision
        _, self.device = _resolve_device(device)
        self.class_agnostic = class_agnostic

    def localize(self, image_paths: list[str]) -> dict[str, list[dict[str, Any]]]:
        """
        Returns {image_path: [{bbox_xyxy, cell_type, objectness}, ...]}.
        """
        out: dict[str, list[dict[str, Any]]] = {}
        use_half = self.half_precision and str(self.device).startswith(("0", "1", "2", "3", "cuda"))
        for img_path in image_paths:
            # Run one image at a time so a large patient folder does not keep
            # all YOLO activations/results resident on the GPU.
            results = self.model.predict(
                source=str(img_path),
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.image_size,
                batch=self.batch_size,
                device=self.device,
                half=use_half,
                verbose=False,
            )
            r = results[0]
            cells: list[dict[str, Any]] = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                for i in range(len(xyxy)):
                    if self.class_agnostic:
                        cell_type = "Unknown"
                    else:
                        cid = int(cls_ids[i])
                        cell_type = LLD_CLASSES[cid] if cid < len(LLD_CLASSES) else "None"
                    cells.append({
                        "bbox_xyxy": tuple(float(v) for v in xyxy[i]),
                        "cell_type": cell_type,
                        "class_id": int(cls_ids[i]),
                        "objectness": float(confs[i]),
                    })
            out[img_path] = cells
            del results, r
            try:
                import torch  # type: ignore

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        return out


# ---------------------------------------------------------------------------
# Stage 2 — EfficientNet attribute classifier
# ---------------------------------------------------------------------------

class EfficientNetAttributeClassifier:
    """
    Wraps the wbc_unified EfficientNet binary attribute classifier.

    The expected checkpoint is `cv/runs/attribute/train/best_attr.pt`, which
    stores a state dict under `model` and metadata such as `backbone`/`imgsz`.
    """

    def __init__(
        self,
        weights_path: str,
        device: str | None = None,
        image_size: int = 224,
        predicts_cell_type: bool = False,
    ):
        try:
            import torch  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Install torch + torchvision for EfficientNet: "
                "`pip install torch torchvision`"
            ) from e
        self._torch = torch
        self.device, _ = _resolve_device(device)
        self.predicts_cell_type = predicts_cell_type

        wbc_cv = Path(__file__).resolve().parent / "wbc_unified" / "cv"
        if str(wbc_cv) not in sys.path:
            sys.path.insert(0, str(wbc_cv))
        from models.attribute_net import build_attribute_model  # type: ignore

        ckpt = torch.load(weights_path, map_location=self.device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            self.model = build_attribute_model(
                num_binary=len(BINARY_ATTRIBUTE_ORDER),
                num_size=3,
                backbone=ckpt.get("backbone", "efficientnet_b0"),
                pretrained=False,
            )
            self.model.load_state_dict(ckpt["model"])
            image_size = int(ckpt.get("imgsz", image_size))
        else:
            self.model = torch.jit.load(weights_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

        self.image_size = image_size
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def classify_crops(
        self, crops: list["Any"]
    ) -> list[tuple[dict[str, float], dict[str, float], str | None, float | None]]:
        """
        :param crops: list of PIL.Image cell crops.
        :return: list of (attributes, attribute_probs, cell_type, cell_type_prob)
                 aligned with `crops`. cell_type is None unless the model
                 predicts it.
        """
        torch = self._torch
        if not crops:
            return []

        batch = torch.stack([self._preprocess(c) for c in crops]).to(self.device)
        with torch.no_grad():
            bin_logits, size_logits = self.model(batch)

        bin_probs = torch.sigmoid(bin_logits)
        size_probs = torch.softmax(size_logits, dim=1)

        results = []
        for i in range(batch.shape[0]):
            attrs: dict[str, float] = {}
            attr_probs: dict[str, float] = {}
            for j, attr in enumerate(BINARY_ATTRIBUTE_ORDER):
                prob = float(bin_probs[i, j].item())
                attrs[attr] = prob
                attr_probs[attr] = prob
            for k, state in enumerate(CELL_SIZE_STATES):
                key = f"{CELL_SIZE_ATTR}_{state}"
                prob = float(size_probs[i, k].item())
                attrs[key] = prob
                attr_probs[key] = prob
            attrs[CELL_SIZE_ATTR] = float(size_probs[i].argmax().item())
            results.append((attrs, attr_probs, None, None))
        return results

    def _preprocess(self, crop: Any):
        import numpy as np
        from PIL import Image

        torch = self._torch
        crop = crop.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(crop, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).float()
        return (x - self._mean) / self._std


# ---------------------------------------------------------------------------
# Two-stage agent
# ---------------------------------------------------------------------------

class TwoStageDetectionAgent(BaseDetectionAgent):
    """
    Two-stage detector used by both wbc-unified and dinobloom backends.

    Flow per case (always YOLO first):
    1. YOLOv11 localizes cells in every PBS tile → bboxes + cell types.
    2. Each detection is cropped from its source tile (with padding).
    3. Stage-2 head (EfficientNet or DinoBloom) runs on those crops only.
    4. Assemble Detection objects with stable cell_ids.
    """

    def __init__(
        self,
        localizer: YOLOv11Localizer,
        attribute_classifier: CropAttributeClassifier,
        crop_padding: float = 0.15,
        prefer_efficientnet_celltype: bool = False,
        attribute_head_name: str = "attribute",
    ):
        self.localizer = localizer
        self.attribute_classifier = attribute_classifier
        self.crop_padding = crop_padding
        self.prefer_efficientnet_celltype = prefer_efficientnet_celltype
        self.attribute_head_name = attribute_head_name

    def detect(self, case_id: str, image_paths: list[str]) -> DetectionResult:
        try:
            from PIL import Image  # type: ignore
        except ImportError as e:
            raise ImportError("Install Pillow: `pip install Pillow`") from e

        # Stage 1: YOLO on full PBS tiles (required before any attribute head).
        per_image_cells = self.localizer.localize(image_paths)
        total_cells = sum(len(cells) for cells in per_image_cells.values())
        logger.info(
            "[%s] YOLO localized %d cells in %d PBS image(s); passing crops to %s head",
            case_id,
            total_cells,
            len(image_paths),
            self.attribute_head_name,
        )

        all_detections: list[Detection] = []
        for img_idx, img_path in enumerate(image_paths):
            cells = per_image_cells.get(img_path, [])
            if not cells:
                continue

            image_id = os.path.basename(img_path)
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                W, H = im.size

                # Stage 1b: crop each YOLO box from the tile.
                crops = []
                for c in cells:
                    x1, y1, x2, y2 = c["bbox_xyxy"]
                    px = self.crop_padding * (x2 - x1)
                    py = self.crop_padding * (y2 - y1)
                    x1 = max(0, int(x1 - px))
                    y1 = max(0, int(y1 - py))
                    x2 = min(W, int(x2 + px))
                    y2 = min(H, int(y2 + py))
                    crops.append(im.crop((x1, y1, x2, y2)))

                if not crops:
                    continue

                # Stage 2: attribute head sees YOLO crops only (never full tiles).
                classify_kwargs: dict[str, Any] = {}
                sig = inspect.signature(self.attribute_classifier.classify_crops)
                if "yolo_cell_types" in sig.parameters:
                    classify_kwargs["yolo_cell_types"] = [c["cell_type"] for c in cells]
                attr_results = self.attribute_classifier.classify_crops(crops, **classify_kwargs)

            for cell_idx, (c, attr_res) in enumerate(zip(cells, attr_results)):
                attrs, attr_probs, en_cell_type, en_cell_type_prob = attr_res

                # Decide final cell type.
                if self.prefer_efficientnet_celltype and en_cell_type is not None:
                    cell_type = en_cell_type
                    cell_type_prob = en_cell_type_prob or c["objectness"]
                else:
                    cell_type = c["cell_type"]
                    cell_type_prob = c["objectness"]
                attrs["class_id"] = c.get("class_id")

                all_detections.append(Detection(
                    cell_id=f"img{img_idx:03d}_c{cell_idx:03d}",
                    image_id=image_id,
                    bbox_xyxy=c["bbox_xyxy"],
                    cell_type=cell_type,
                    objectness=c["objectness"],
                    cell_type_prob=cell_type_prob,
                    attributes=attrs,
                    attribute_probs=attr_probs,
                ))

        return DetectionResult(
            case_id=case_id,
            n_images=len(image_paths),
            detections=all_detections,
        )
