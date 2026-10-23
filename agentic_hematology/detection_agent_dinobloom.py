"""DinoBloom integration for the two-stage detection pipeline.

DinoBloom (https://github.com/marrlab/DinoBloom) is a DINOv2-based foundation model
for hematology cell embeddings. Pair it with YOLO for bounding boxes.

Attribute inference without training custom probes uses k-NN over the train manifest
(DinoBloom paper-style retrieval on frozen embeddings).
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from .detection_agent import LLD_CLASSES
from .detection_agent_v2 import (
    ATTRIBUTE_ORDER,
    BINARY_ATTRIBUTE_ORDER,
    CELL_SIZE_ATTR,
    CELL_SIZE_STATES,
    EfficientNetAttributeClassifier,
    _resolve_device,
)

LLD_CLASS_TO_IDX = {name: idx for idx, name in enumerate(LLD_CLASSES)}

logger = logging.getLogger(__name__)

DINOBLOOM_HF_REPO = "MarrLab/DinoBloom"
DINOBLOOM_VARIANTS: Dict[str, Tuple[str, int]] = {
    "s": ("dinov2_vits14", 384),
    "b": ("dinov2_vitb14", 768),
    "l": ("dinov2_vitl14", 1024),
    "g": ("dinov2_vitg14", 1536),
}

DINOBLOOM_INPUT_SIZE = 224
_PATCH_SIZE = 14


def _resolve_variant(variant: str) -> Tuple[str, int]:
    key = variant.strip().lower()
    if key not in DINOBLOOM_VARIANTS:
        supported = ", ".join(sorted(DINOBLOOM_VARIANTS))
        raise ValueError(f"Unknown DinoBloom variant {variant!r}; choose one of: {supported}")
    return DINOBLOOM_VARIANTS[key]


def resolve_dinobloom_weights(weights_spec: Optional[str], variant: str = "l") -> str:
    """Resolve a local checkpoint path or download pretrained weights from HuggingFace."""
    if weights_spec and str(weights_spec).lower() not in {"auto", "hf", "download"}:
        path = Path(weights_spec)
        if path.is_file():
            return str(path.resolve())
        raise FileNotFoundError(f"DinoBloom weights not found: {path}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Install huggingface_hub (`pip install huggingface_hub`) or pass "
            "--dinobloom-weights /path/to/DinoBloom-B.pth"
        ) from exc

    key = variant.strip().lower()
    filename = f"pytorch_model_{key}.bin"
    logger.info("Fetching DinoBloom-%s from HuggingFace (%s/%s)...", key.upper(), DINOBLOOM_HF_REPO, filename)
    return hf_hub_download(repo_id=DINOBLOOM_HF_REPO, filename=filename)


def _load_manifest_rows(manifest_csv: Path, split: str) -> List[dict]:
    rows: List[dict] = []
    with manifest_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("split") == split:
                rows.append(row)
    return rows


def _crop_from_manifest_row(row: dict, imgsz: int, pad: float) -> Image.Image:
    import sys

    cv_root = Path(__file__).resolve().parent / "wbc_unified" / "cv"
    if str(cv_root) not in sys.path:
        sys.path.insert(0, str(cv_root))
    from utils.labels import crop_with_padding  # noqa: WPS433

    image = Image.open(row["image"]).convert("RGB")
    w, h = image.size
    xywh = np.array([float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"])], dtype=np.float32)
    x1, y1, x2, y2 = crop_with_padding(w, h, xywh, pad=pad)
    return image.crop((x1, y1, x2, y2)).resize((imgsz, imgsz), Image.BILINEAR)


class DinoBloomKNNAttributePredictor:
    """k-NN attribute lookup bank built from labeled train crops (reference only).

    This runs once at startup to embed the train manifest. At inference time,
    ``DinoBloomAttributeClassifier.classify_crops`` still receives YOLO crops
    from ``TwoStageDetectionAgent``, not manifest/GT boxes.
    """

    def __init__(
        self,
        embedder: "DinoBloomEmbedder",
        manifest_path: str,
        cache_path: Optional[str] = None,
        train_split: str = "train",
        knn_k: int = 5,
        embed_batch: int = 32,
        pad: float = 0.15,
    ) -> None:
        self.embedder = embedder
        self.k = max(1, int(knn_k))
        manifest = Path(manifest_path)
        if not manifest.is_file():
            raise FileNotFoundError(
                f"Attribute manifest not found for DinoBloom k-NN: {manifest}. "
                "Run wbc_unified/cv/data/prepare_dataset.py or pass --dinobloom-knn-manifest."
            )

        cache = (
            Path(cache_path)
            if cache_path
            else Path(__file__).resolve().parent
            / "wbc_unified"
            / "cv"
            / "runs"
            / "attribute_dinobloom"
            / "knn_train_embeddings.npz"
        )
        self.cache_path = cache
        if cache.is_file():
            logger.info("Loading DinoBloom k-NN bank from cache: %s", cache)
            data = np.load(cache, allow_pickle=False)
            self._bank_emb = data["embeddings"].astype(np.float32)
            self._bank_y = data["labels"].astype(np.int8)
            self._bank_valid = data["valid"].astype(bool)
            self._bank_size = data["size_labels"].astype(np.int8)
        else:
            logger.info("Building DinoBloom k-NN bank from %s split=%s (one-time cache)...", manifest, train_split)
            rows = _load_manifest_rows(manifest, train_split)
            if not rows:
                raise ValueError(f"No train rows in manifest {manifest}")
            (
                self._bank_emb,
                self._bank_y,
                self._bank_valid,
                self._bank_size,
            ) = self._embed_manifest_rows(rows, embed_batch, pad)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache,
                embeddings=self._bank_emb,
                labels=self._bank_y,
                valid=self._bank_valid,
            )
            logger.info("Cached k-NN bank: %s (%d cells)", cache, len(self._bank_emb))

        norms = np.linalg.norm(self._bank_emb, axis=1, keepdims=True)
        self._bank_norm = self._bank_emb / np.clip(norms, 1e-8, None)

    def _embed_manifest_rows(
        self,
        rows: List[dict],
        batch: int,
        pad: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(rows)
        d = self.embedder.embed_dim
        a = len(BINARY_ATTRIBUTE_ORDER)
        embeddings = np.zeros((n, d), dtype=np.float32)
        labels = np.full((n, a), -1, dtype=np.int8)
        valid = np.zeros((n, a), dtype=bool)
        size_labels = np.full(n, -1, dtype=np.int8)


        batch_crops: List[Image.Image] = []
        batch_idx: List[int] = []

        def flush() -> None:
            if not batch_crops:
                return
            embs = self.embedder.embed_pil_batch(batch_crops)
            for bi, row_i in enumerate(batch_idx):
                embeddings[row_i] = embs[bi]
            batch_crops.clear()
            batch_idx.clear()

        for i, row in enumerate(rows):
            batch_crops.append(_crop_from_manifest_row(row, DINOBLOOM_INPUT_SIZE, pad))
            batch_idx.append(i)
            for j, name in enumerate(BINARY_ATTRIBUTE_ORDER):
                v = int(row[name])
                if v in (0, 1):
                    labels[i, j] = v
                    valid[i, j] = True
            sv = int(row[CELL_SIZE_ATTR])
            if 0 <= sv < len(CELL_SIZE_STATES):
                size_labels[i] = sv
            if len(batch_crops) >= batch:
                flush()
        flush()
        return embeddings, labels, valid, size_labels

    def predict(self, embeddings: np.ndarray) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
        if len(embeddings) == 0:
            return [], []

        q_norm = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8, None)
        sims = q_norm @ self._bank_norm.T

        attrs: List[Dict[str, float]] = []
        attr_probs: List[Dict[str, float]] = []
        for i in range(len(embeddings)):
            row_attrs: Dict[str, float] = {}
            row_probs: Dict[str, float] = {}
            order = np.argsort(-sims[i])
            for j, name in enumerate(BINARY_ATTRIBUTE_ORDER):
                size_picked: List[int] = []
                for idx in order:
                    if self._bank_size[idx] >= 0:
                        size_picked.append(int(self._bank_size[idx]))
                    if len(size_picked) >= self.k:
                        break
                counts = np.bincount(size_picked, minlength=len(CELL_SIZE_STATES)) if size_picked else None
                for k_i, state in enumerate(CELL_SIZE_STATES):
                    p = float(counts[k_i] / len(size_picked)) if size_picked else 0.0
                    row_attrs[f"{CELL_SIZE_ATTR}_{state}"] = p
                    row_probs[f"{CELL_SIZE_ATTR}_{state}"] = p
                row_attrs[CELL_SIZE_ATTR] = float(int(np.argmax(counts)) if size_picked else 0)
                row_probs[CELL_SIZE_ATTR] = row_attrs[CELL_SIZE_ATTR]
            attrs.append(row_attrs)
            attr_probs.append(row_probs)
        return attrs, attr_probs


def build_dinobloom_attribute_head(in_dim: int, num_attrs: int) -> nn.Sequential:
    """Small MLP head (same shape as EfficientNet AttributeNet head)."""
    return nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_attrs),
    )


def _parse_dinobloom_checkpoint(raw: object) -> dict[str, torch.Tensor]:
    """Normalize MarrLab / cAItomorph / HF DinoBloom checkpoints to DINOv2 keys."""
    if not isinstance(raw, dict):
        raise ValueError(f"Expected checkpoint dict, got {type(raw)!r}")

    if "teacher" in raw:
        state: dict[str, torch.Tensor] = {}
        for key, value in raw["teacher"].items():
            if "dino_head" in key or "ibot_head" in key:
                continue
            state[key.replace("backbone.", "", 1)] = value
        if not state:
            raise ValueError("DinoBloom checkpoint 'teacher' block contained no backbone weights")
        return state

    if "state_dict" in raw:
        return dict(raw["state_dict"])
    if "model" in raw and isinstance(raw["model"], dict):
        return dict(raw["model"])
    if any(k.startswith(("cls_token", "patch_embed", "blocks.")) for k in raw):
        return dict(raw)

    raise ValueError(
        "Unrecognized DinoBloom checkpoint format. Expected keys like "
        "'teacher', 'state_dict', 'model', or bare DINOv2 backbone tensors."
    )


class DinoBloomEmbedder:
    """Load DinoBloom weights on top of a DINOv2 ViT backbone."""

    def __init__(
        self,
        weights_path: str,
        variant: str = "l",
        device: Optional[str] = None,
        hub_dir: Optional[str] = None,
    ) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"DinoBloom weights not found: {self.weights_path}")

        dinov2_name, self.embed_dim = _resolve_variant(variant)
        torch_device, _ = _resolve_device(device)
        self.device = torch.device(torch_device)

        hub_kwargs = {"source": "github", "trust_repo": True}
        if hub_dir:
            hub_kwargs["model_dir"] = hub_dir

        logger.info("Loading DINOv2 backbone %s for DinoBloom-%s", dinov2_name, variant.upper())
        self.model = torch.hub.load("facebookresearch/dinov2", dinov2_name, **hub_kwargs)
        self.model.eval()

        num_tokens = int(1 + (DINOBLOOM_INPUT_SIZE / _PATCH_SIZE) ** 2)
        self.model.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, self.embed_dim))

        ckpt = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        state_dict = _parse_dinobloom_checkpoint(ckpt)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(
                f"DinoBloom checkpoint missing {len(missing)} backbone keys "
                f"(variant={variant!r}, file={self.weights_path.name}). "
                f"First missing: {missing[:3]}. "
                "Check that --dinobloom-variant matches the weights file "
                "(e.g. l for DinoBloom-L.pth, b for DinoBloom-B.pth)."
            )
        if unexpected:
            logger.warning(
                "Ignored %d unexpected DinoBloom checkpoint keys: %s",
                len(unexpected),
                ", ".join(unexpected[:5]),
            )
        self.model.to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.Resize((DINOBLOOM_INPUT_SIZE, DINOBLOOM_INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.inference_mode()
    def embed_pil_batch(self, crops: Sequence[Image.Image]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        tensors = torch.stack([self.transform(img.convert("RGB")) for img in crops]).to(self.device)
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(tensors)
            if isinstance(feats, dict):
                if "x_norm_clstoken" in feats:
                    emb = feats["x_norm_clstoken"]
                elif "cls_token" in feats:
                    emb = feats["cls_token"]
                else:
                    emb = next(iter(feats.values()))
            else:
                emb = feats
        else:
            emb = self.model(tensors)

        if emb.ndim > 2:
            emb = emb[:, 0] if emb.shape[1] > 1 else emb.mean(dim=1)
        return emb.detach().cpu().numpy().astype(np.float32)


class _TorchLinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class DinoBloomAttributeClassifier:
    """Stage-2 attribute head: DinoBloom embeddings on YOLO cell crops only.

    Must be wired through ``TwoStageDetectionAgent`` so PBS tiles are localized
    and cropped by YOLO before any DinoBloom forward pass at inference.
    """

    def __init__(
        self,
        weights_path: Optional[str] = "auto",
        attr_probes_path: Optional[str] = None,
        variant: str = "l",
        attr_mode: str = "auto",
        knn_manifest_path: Optional[str] = None,
        knn_cache_path: Optional[str] = None,
        knn_k: int = 5,
        device: Optional[str] = None,
        hub_dir: Optional[str] = None,
    ) -> None:
        probes_path = Path(attr_probes_path) if attr_probes_path else None
        if probes_path and probes_path.suffix.lower() in {".pt", ".pth"} and probes_path.is_file():
            ckpt = torch.load(probes_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and ckpt.get("dinobloom_variant") in {"s", "b", "l", "g"}:
                variant = str(ckpt["dinobloom_variant"])

        resolved_weights = resolve_dinobloom_weights(weights_path, variant)
        self.embedder = DinoBloomEmbedder(
            weights_path=resolved_weights,
            variant=variant,
            device=device,
            hub_dir=hub_dir,
        )
        self.device = self.embedder.device
        self._probes: Dict[str, object] = {}
        self._torch_head: Optional[nn.Module] = None
        self._knn: Optional[DinoBloomKNNAttributePredictor] = None

        mode = (attr_mode or "auto").lower()
        if mode == "auto":
            mode = "probes" if probes_path and probes_path.is_file() else "knn"

        if mode == "probes":
            if not probes_path or not probes_path.is_file():
                raise FileNotFoundError(
                    f"DinoBloom attribute probes not found: {probes_path}. "
                    "Use --dinobloom-attr-mode knn for inference without training, "
                    "or train probes with wbc_unified/cv/train_dinobloom_attributes.py."
                )
            self._load_attr_probes(probes_path)
            logger.info("DinoBloom attributes: linear probes from %s", probes_path)
        elif mode == "knn":
            if not knn_manifest_path:
                raise ValueError("knn_manifest_path is required for DinoBloom k-NN attributes")
            self._knn = DinoBloomKNNAttributePredictor(
                embedder=self.embedder,
                manifest_path=knn_manifest_path,
                cache_path=knn_cache_path,
                knn_k=knn_k,
            )
            logger.info(
                "DinoBloom attributes: k-NN (k=%d) over train manifest %s",
                knn_k,
                knn_manifest_path,
            )
        else:
            raise ValueError(f"Unknown DinoBloom attr_mode {attr_mode!r}; use auto, probes, or knn")

    def _load_attr_probes(self, path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix in {".joblib", ".pkl", ".pickle"}:
            try:
                import joblib
            except ImportError as exc:
                raise ImportError("joblib is required for DinoBloom attribute probes") from exc
            payload = joblib.load(path)
            if isinstance(payload, dict) and "probes" in payload:
                self._probes = dict(payload["probes"])
            elif isinstance(payload, dict):
                self._probes = payload
            else:
                raise ValueError(f"Unexpected attribute probe payload in {path}")
            missing = [name for name in ATTRIBUTE_ORDER if name not in self._probes]
            if missing:
                logger.warning("DinoBloom attribute probes missing heads: %s", ", ".join(missing))
        elif suffix in {".pt", ".pth"}:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, nn.Module):
                self._torch_head = ckpt.eval().to(self.device)
            elif isinstance(ckpt, dict) and "model" in ckpt:
                in_dim = ckpt.get("embed_dim", self.embedder.embed_dim)
                out_dim = ckpt.get("num_attrs", len(BINARY_ATTRIBUTE_ORDER) + 3)
                head = build_dinobloom_attribute_head(in_dim, out_dim)
                head.load_state_dict(ckpt["model"])
                self._torch_head = head.eval().to(self.device)
            else:
                raise ValueError(f"Unsupported torch attribute probe checkpoint: {path}")
        else:
            raise ValueError(f"Unsupported attribute probe format: {path}")

    def _predict_attribute_probs(self, embeddings: np.ndarray) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
        if self._knn is not None:
            return self._knn.predict(embeddings)

        n = len(embeddings)
        attrs: List[Dict[str, float]] = []
        attr_probs: List[Dict[str, float]] = []

        if self._torch_head is not None:
            x = torch.from_numpy(embeddings).to(self.device)
            with torch.inference_mode():
                logits = self._torch_head(x)
                nb = len(BINARY_ATTRIBUTE_ORDER)
                bin_probs = torch.sigmoid(logits[:, :nb]).cpu().numpy()
                size_probs = torch.softmax(logits[:, nb:], dim=1).cpu().numpy()
            for i in range(n):
                row = {name: float(bin_probs[i, j]) for j, name in enumerate(BINARY_ATTRIBUTE_ORDER)}
                for k, state in enumerate(CELL_SIZE_STATES):
                    row[f"{CELL_SIZE_ATTR}_{state}"] = float(size_probs[i, k])
                row[CELL_SIZE_ATTR] = float(size_probs[i].argmax())
                attrs.append(dict(row))
                attr_probs.append(dict(row))
            return attrs, attr_probs

        for i in range(n):
            row_attrs: Dict[str, float] = {}
            row_probs: Dict[str, float] = {}
            emb = embeddings[i : i + 1]
            for name in BINARY_ATTRIBUTE_ORDER:
                clf = self._probes.get(name)
                if clf is None:
                    row_attrs[name] = 0.0
                    row_probs[name] = 0.0
                    continue
                if hasattr(clf, "predict_proba"):
                    prob_pos = float(clf.predict_proba(emb)[0][1])
                else:
                    pred = float(clf.predict(emb)[0])
                    prob_pos = pred
                row_attrs[name] = prob_pos
                row_probs[name] = prob_pos
            for state in CELL_SIZE_STATES:
                row_attrs[f"{CELL_SIZE_ATTR}_{state}"] = 0.0
                row_probs[f"{CELL_SIZE_ATTR}_{state}"] = 0.0
            row_attrs[CELL_SIZE_ATTR] = 0.0
            row_probs[CELL_SIZE_ATTR] = 0.0
            
            attrs.append(row_attrs)
            attr_probs.append(row_probs)
        return attrs, attr_probs

    def classify_crops(
        self,
        crops: Sequence[Image.Image],
        yolo_cell_types: Optional[Sequence[str]] = None,
    ) -> List[Tuple[Dict[str, float], Dict[str, float], None, None]]:
        del yolo_cell_types  # cell type comes from YOLO stage-1 upstream
        if not crops:
            return []
        logger.info("DinoBloom attribute inference on %d YOLO crop(s)", len(crops))
        embeddings = self.embedder.embed_pil_batch(crops)
        attrs, attr_probs = self._predict_attribute_probs(embeddings)
        return [(a, p, None, None) for a, p in zip(attrs, attr_probs)]


class DinoBloomCellClassifier:
    """Classify YOLO crops with DinoBloom embeddings + optional attribute model."""

    def __init__(
        self,
        weights_path: str,
        variant: str = "l",
        classifier_path: Optional[str] = None,
        class_names_path: Optional[str] = None,
        device: Optional[str] = None,
        hub_dir: Optional[str] = None,
        attribute_classifier: Optional[EfficientNetAttributeClassifier] = None,
        fallback_to_yolo_type: bool = True,
    ) -> None:
        resolved_weights = resolve_dinobloom_weights(weights_path, variant)
        self.embedder = DinoBloomEmbedder(
            weights_path=resolved_weights,
            variant=variant,
            device=device,
            hub_dir=hub_dir,
        )
        self.device = self.embedder.device
        self.attribute_classifier = attribute_classifier
        self.fallback_to_yolo_type = fallback_to_yolo_type

        self._sklearn_clf = None
        self._torch_head: Optional[_TorchLinearHead] = None
        self._class_names: List[str] = []

        if classifier_path:
            self._load_classifier(classifier_path, class_names_path)
        elif not fallback_to_yolo_type:
            logger.warning(
                "No --dinobloom-classifier provided; cell types will remain Unclassified "
                "unless --dinobloom-fallback-yolo is set."
            )

    def _load_classifier(self, classifier_path: str, class_names_path: Optional[str]) -> None:
        path = Path(classifier_path)
        if not path.is_file():
            raise FileNotFoundError(f"DinoBloom classifier not found: {path}")

        suffix = path.suffix.lower()
        if suffix in {".joblib", ".pkl", ".pickle"}:
            try:
                import joblib
            except ImportError as exc:
                raise ImportError(
                    "joblib is required to load sklearn DinoBloom classifiers "
                    "(pip install joblib)."
                ) from exc
            self._sklearn_clf = joblib.load(path)
            if hasattr(self._sklearn_clf, "classes_"):
                self._class_names = [str(c) for c in self._sklearn_clf.classes_]
        elif suffix in {".pt", ".pth"}:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(payload, nn.Module):
                self._torch_head = payload
                self._torch_head.eval().to(self.device)
            elif isinstance(payload, dict):
                classes = payload.get("classes")
                if not classes:
                    classes = payload.get("class_names") or []
                self._class_names = [str(c) for c in classes]
                weight = payload.get("weight")
                if weight is None:
                    weight = payload.get("W")
                bias = payload.get("bias")
                if bias is None:
                    bias = payload.get("b")
                if weight is None:
                    raise ValueError(
                        f"Torch classifier checkpoint must contain 'weight' and 'bias': {path}"
                    )
                weight_t = torch.as_tensor(weight, dtype=torch.float32)
                bias_t = torch.as_tensor(bias, dtype=torch.float32) if bias is not None else None
                out_dim = weight_t.shape[0]
                in_dim = weight_t.shape[1]
                head = _TorchLinearHead(in_dim, out_dim)
                head.linear.weight.data.copy_(weight_t)
                if bias_t is not None:
                    head.linear.bias.data.copy_(bias_t)
                head.eval().to(self.device)
                self._torch_head = head
                if not self._class_names:
                    self._class_names = [str(i) for i in range(out_dim)]
            else:
                raise ValueError(f"Unsupported torch classifier payload in {path}")
        else:
            raise ValueError(
                f"Unsupported DinoBloom classifier format {suffix!r}; "
                "use .joblib/.pkl (sklearn) or .pt/.pth (linear head)."
            )

        if class_names_path:
            names_path = Path(class_names_path)
            if not names_path.is_file():
                raise FileNotFoundError(f"DinoBloom class names file not found: {names_path}")
            with names_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict) and "classes" in payload:
                payload = payload["classes"]
            self._class_names = [str(c) for c in payload]

        if self._class_names:
            unknown = [c for c in self._class_names if c not in LLD_CLASS_TO_IDX]
            if unknown:
                logger.warning(
                    "DinoBloom classifier includes labels outside LLD taxonomy: %s",
                    ", ".join(unknown[:5]),
                )

    def _predict_cell_types(
        self,
        embeddings: np.ndarray,
        yolo_types: Sequence[str],
    ) -> Tuple[List[str], List[float]]:
        n = len(embeddings)
        if n == 0:
            return [], []

        if self._sklearn_clf is not None:
            pred_labels = self._sklearn_clf.predict(embeddings)
            if hasattr(self._sklearn_clf, "predict_proba"):
                probs = self._sklearn_clf.predict_proba(embeddings)
                confidences = probs.max(axis=1).tolist()
            else:
                confidences = [0.5] * n
            labels = [str(label) for label in pred_labels]
            return self._map_to_lld(labels, confidences, yolo_types)

        if self._torch_head is not None:
            x = torch.from_numpy(embeddings).to(self.device)
            with torch.inference_mode():
                logits = self._torch_head(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            idx = probs.argmax(axis=1)
            labels = [
                self._class_names[i] if i < len(self._class_names) else str(i)
                for i in idx
            ]
            confidences = probs.max(axis=1).tolist()
            return self._map_to_lld(labels, confidences, yolo_types)

        if self.fallback_to_yolo_type:
            return list(yolo_types), [0.5] * n

        return ["Unclassified"] * n, [0.0] * n

    def _map_to_lld(
        self,
        labels: Sequence[str],
        confidences: Sequence[float],
        yolo_types: Sequence[str],
    ) -> Tuple[List[str], List[float]]:
        mapped: List[str] = []
        mapped_conf: List[float] = []
        for label, conf, yolo in zip(labels, confidences, yolo_types):
            if label in LLD_CLASS_TO_IDX:
                mapped.append(label)
                mapped_conf.append(float(conf))
            elif self.fallback_to_yolo_type and yolo in LLD_CLASS_TO_IDX:
                mapped.append(yolo)
                mapped_conf.append(float(conf) * 0.5)
            else:
                mapped.append("Unclassified")
                mapped_conf.append(float(conf) * 0.25)
        return mapped, mapped_conf

    def classify_crops(
        self,
        crops: Sequence[Image.Image],
        yolo_cell_types: Optional[Sequence[str]] = None,
    ) -> List[Tuple[Dict[str, float], Dict[str, float], str, float]]:
        """Same return contract as ``EfficientNetAttributeClassifier.classify_crops``."""
        n = len(crops)
        yolo_types = list(yolo_cell_types or ["Unclassified"] * n)
        if len(yolo_types) != n:
            raise ValueError("yolo_cell_types length must match number of crops")

        embeddings = self.embedder.embed_pil_batch(crops)
        cell_types, cell_type_probs = self._predict_cell_types(embeddings, yolo_types)

        attr_tuples: List[Tuple[Dict[str, float], Dict[str, float], str | None, float | None]] = []
        if self.attribute_classifier is not None:
            attr_tuples = self.attribute_classifier.classify_crops(list(crops))

        results: List[Tuple[Dict[str, float], Dict[str, float], str, float]] = []
        for i in range(n):
            if attr_tuples:
                attrs, attr_probs, _, _ = attr_tuples[i]
            else:
                attrs = {name: 0.0 for name in ATTRIBUTE_ORDER}
                attr_probs = dict(attrs)
            results.append((attrs, attr_probs, cell_types[i], cell_type_probs[i]))
        return results
