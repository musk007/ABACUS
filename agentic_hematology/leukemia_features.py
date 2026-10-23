"""Patient-level feature helpers and LLD split/label discovery."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 13 informative WBC types (exclude "none")
CLINICAL_CELL_TYPES: tuple[str, ...] = (
    "myeloblast",
    "lymphoblast",
    "neutrophil",
    "atypical lymphocyte",
    "promonocyte",
    "monoblast",
    "lymphocyte",
    "myelocyte",
    "abnormal promyelocyte",
    "monocyte",
    "metamyelocyte",
    "eosinophil",
    "basophil",
)

ATTRIBUTE_KEYS: tuple[str, ...] = (
    "cell_size",
    "nuclear_chromatio",
    "nuclear_shape",
    "nucleolus",
    "cytoplasm",
    "cytoplasmic_basophilia",
    "cytoplasmic_vacuoles",
)

CLINICAL_GROUPS: dict[str, tuple[str, ...]] = {
    "blasts": ("myeloblast", "lymphoblast", "monoblast", "abnormal promyelocyte"),
    "intermediate_myeloid": ("promonocyte", "myelocyte", "metamyelocyte"),
    "mature_granulocytes": ("neutrophil", "eosinophil", "basophil"),
    "lymphoid": ("lymphocyte", "atypical lymphocyte"),
    "monocytic": ("monocyte",),
}

FEATURES_EXCLUDED = {
    "qc__global_canvas_stitching_active",
    "qc__overlap_deduplication_active",
}

LLD_DIAGNOSIS_LABELS: frozenset[str] = frozenset({"ALL", "AML", "APML", "CLL", "CML"})


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def load_patient_stats(path: Path | str) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict patient_id -> record in {path}")
    return payload


def discover_lld_split_from_cv(cv_root: Path) -> dict[str, list[str]]:
    """Read-only split from wbc_unified det_dataset label filenames."""
    split: dict[str, set[str]] = {"train": set(), "test": set()}
    for part in ("train", "test"):
        label_dir = cv_root / "generated" / "det_dataset" / "labels" / part
        if not label_dir.is_dir():
            continue
        for label_path in label_dir.glob("*.txt"):
            split[part].add(label_path.name.split("_")[0])
    # If a patient appears in both, keep them in test only.
    overlap = split["train"] & split["test"]
    split["train"] -= overlap
    return {k: sorted(v, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)) for k, v in split.items()}


def discover_patient_labels_from_cv(
    cv_root: Path,
    *,
    image_root: Path | None = None,
) -> dict[str, str]:
    """Diagnosis labels from LLD tile filenames: ``{patient}_{...}_{DIAGNOSIS}.png``."""
    if image_root is None:
        image_root = cv_root / "generated" / "det_dataset" / "images"
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    labels: dict[str, str] = {}
    for split in ("train", "test"):
        img_dir = image_root / split
        if not img_dir.is_dir():
            continue
        for path in sorted(img_dir.iterdir()):
            if path.suffix.lower() not in suffixes:
                continue
            parts = path.stem.split("_")
            if len(parts) < 2:
                continue
            patient_id = parts[0]
            diagnosis = parts[-1].strip().upper()
            if diagnosis not in LLD_DIAGNOSIS_LABELS:
                continue
            existing = labels.get(patient_id)
            if existing and existing != diagnosis:
                raise ValueError(
                    f"Conflicting diagnosis labels for patient {patient_id}: "
                    f"{existing} vs {diagnosis} (from {path.name})"
                )
            labels[patient_id] = diagnosis
    return labels


def load_or_create_split(
    stats: dict[str, dict[str, Any]] | None = None,
    *,
    split_path: Path | None = None,
    cv_root: Path | None = None,
    patient_ids: set[str] | None = None,
) -> dict[str, list[str]]:
    if split_path and split_path.is_file():
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        return {
            "train": [str(x) for x in payload.get("train", [])],
            "test": [str(x) for x in payload.get("test", [])],
        }
    if cv_root is None:
        raise ValueError("Provide --split-json or --cv-root to derive train/test patient IDs.")
    derived = discover_lld_split_from_cv(cv_root)
    allowed = set(stats.keys()) if stats else patient_ids
    if allowed is not None:
        derived["train"] = [pid for pid in derived["train"] if pid in allowed]
        derived["test"] = [pid for pid in derived["test"] if pid in allowed]
    return derived


def _attribute_state_columns(stats: dict[str, dict[str, Any]]) -> list[str]:
    states: dict[str, set[str]] = {k: set() for k in ATTRIBUTE_KEYS}
    for rec in stats.values():
        attrs = rec.get("attributes") or {}
        for attr_key in ATTRIBUTE_KEYS:
            percentages = (attrs.get(attr_key) or {}).get("percentages") or {}
            for state in percentages:
                if str(state).lower() in {"n_a", "na"}:
                    continue
                states[attr_key].add(_slug(state))
    columns: list[str] = []
    for attr_key in ATTRIBUTE_KEYS:
        for state in sorted(states[attr_key]):
            columns.append(f"attr_{_slug(attr_key)}__{_slug(state)}__pct")
    return columns


def build_feature_matrix(
    stats: dict[str, dict[str, Any]],
    *,
    include_qc_features: bool = True,
    exclude_low_cell_count: bool = False,
) -> tuple[pd.DataFrame, pd.Series, list[str], dict[str, bool]]:
    attr_cols = _attribute_state_columns(stats)
    feature_names: list[str] = []
    for cell_type in CLINICAL_CELL_TYPES:
        feature_names.append(f"pct_{_slug(cell_type)}")
    feature_names.append("blast_pool_percentage_of_wbc")
    feature_names.extend(attr_cols)
    for group in CLINICAL_GROUPS:
        feature_names.append(f"group_{group}_pct")
    if include_qc_features:
        feature_names.append("n_cells_identified_wbc")

    rows: list[dict[str, float]] = []
    labels: list[str] = []
    patient_ids: list[str] = []
    flags: dict[str, bool] = {}

    for patient_id, rec in sorted(stats.items(), key=lambda kv: kv[0]):
        qc = (rec.get("report_ready") or {}).get("qc") or {}
        low_cell = bool(qc.get("low_cell_count_warning", False))
        flags[patient_id] = low_cell
        if exclude_low_cell_count and low_cell:
            continue

        pct_clinical = rec.get("cell_percentages_clinical") or {}
        row = {name: 0.0 for name in feature_names}
        for cell_type in CLINICAL_CELL_TYPES:
            row[f"pct_{_slug(cell_type)}"] = float(pct_clinical.get(cell_type, 0.0))

        report_ready = rec.get("report_ready") or {}
        row["blast_pool_percentage_of_wbc"] = float(
            report_ready.get("blast_pool_percentage_of_wbc", 0.0)
        )

        attrs = rec.get("attributes") or {}
        for attr_key in ATTRIBUTE_KEYS:
            percentages = (attrs.get(attr_key) or {}).get("percentages") or {}
            for state, value in percentages.items():
                if str(state).lower() in {"n_a", "na"}:
                    continue
                col = f"attr_{_slug(attr_key)}__{_slug(state)}__pct"
                if col in row:
                    row[col] = float(value)

        group_den = float(rec.get("n_cells_identified_wbc") or 0.0)
        counts = rec.get("cell_counts") or {}
        if group_den <= 0:
            group_den = float(sum(v for k, v in counts.items() if k != "none") or 1.0)
        for group, members in CLINICAL_GROUPS.items():
            group_count = sum(float(counts.get(ct, 0)) for ct in members)
            row[f"group_{group}_pct"] = round(100.0 * group_count / group_den, 4)

        if include_qc_features:
            row["n_cells_identified_wbc"] = float(rec.get("n_cells_identified_wbc") or group_den)

        rows.append(row)
        labels.append(str(rec.get("metadata_filename_diagnosis", "Unknown")))
        patient_ids.append(str(patient_id))

    X = pd.DataFrame(rows, columns=feature_names).fillna(0.0)
    X.index = patient_ids
    y = pd.Series(labels, index=patient_ids, name="metadata_filename_diagnosis")
    kept_names = [c for c in feature_names if c not in FEATURES_EXCLUDED]
    return X[kept_names], y, kept_names, flags


def humanize_feature_name(name: str) -> str:
    if name.startswith("pct_"):
        return f"{name.removeprefix('pct_').replace('_', ' ')} differential %"
    if name.startswith("group_") and name.endswith("_pct"):
        group = name.removeprefix("group_").removesuffix("_pct").replace("_", " ")
        return f"{group} lineage group %"
    if name.startswith("attr_") and name.endswith("__pct"):
        body = name.removeprefix("attr_").removesuffix("__pct")
        attr, state = body.split("__", 1)
        return f"{state.replace('_', ' ')} {attr.replace('_', ' ')} attribute %"
    if name == "blast_pool_percentage_of_wbc":
        return "blast pool % of WBC"
    if name == "n_cells_identified_wbc":
        return "identified WBC cell count"
    return name.replace("_", " ")


def build_simple_features_from_infer_json(
    pred_json: Path | str,
) -> tuple[list[dict[str, float]], list[str], list[str]]:
    """Legacy simple pct_* features from wbc_unified infer JSON (infer feature source)."""
    import json
    from pathlib import Path as _Path

    from agentic_hematology.aggregator import aggregate
    from agentic_hematology.schemas import Detection, DetectionResult

    pred_json = _Path(pred_json)
    payload = json.loads(pred_json.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "predictions" in payload:
        payload = payload["predictions"]

    patients: dict[str, dict] = {}
    for img_rec in payload:
        image_path = _Path(str(img_rec.get("image", "")))
        stem_parts = image_path.stem.split("_")
        if len(stem_parts) < 5:
            continue
        pid = stem_parts[0]
        gt_label = stem_parts[-1].strip()
        if not gt_label:
            continue
        rec = patients.setdefault(pid, {"patient_id": pid, "label": gt_label, "images": []})
        rec["images"].append(img_rec)

    X: list[dict[str, float]] = []
    y: list[str] = []
    patient_ids: list[str] = []
    for patient in patients.values():
        pid = str(patient.get("patient_id", "unknown"))
        gt_label = str(patient.get("label", "")).strip()
        detections = []
        for img_idx, img_rec in enumerate(patient.get("images", [])):
            image_id = _Path(str(img_rec.get("image", ""))).name
            for cell_idx, cell in enumerate(img_rec.get("cells", [])):
                attrs = dict(cell.get("attributes", {}))
                attrs["class_id"] = cell.get("class_id")
                detections.append(
                    Detection(
                        cell_id=f"img{img_idx:03d}_c{cell_idx:03d}",
                        image_id=image_id,
                        bbox_xyxy=tuple(float(v) for v in cell.get("xyxy", [0, 0, 1, 1])),
                        cell_type=str(cell.get("class_name", "Unknown")),
                        objectness=float(cell.get("conf", 0.0)),
                        attributes=attrs,
                        attribute_probs={},
                    )
                )
        if not detections:
            continue
        detection_result = DetectionResult(
            case_id=pid,
            n_images=len(patient.get("images", [])),
            detections=detections,
        )
        findings = aggregate(detection_result)
        X.append(build_feature_row_from_findings(findings, detection_result=detection_result))
        y.append(gt_label)
        patient_ids.append(pid)
    return X, y, patient_ids


def build_feature_row_from_findings(
    findings,
    feature_names: list[str] | None = None,
    *,
    detection_result=None,
) -> dict[str, float]:
    """Build patient-level tabular features from live detection + aggregation only."""
    from agentic_hematology.detection_agent_v2 import (
        BINARY_ATTRIBUTE_ORDER,
        CELL_SIZE_ATTR,
        CELL_SIZE_STATES,
    )
    ATTRIBUTE_ORDER = BINARY_ATTRIBUTE_ORDER

    row = {f"pct_{_slug(cell_type)}": 0.0 for cell_type in CLINICAL_CELL_TYPES}
    for name, value in (findings.cell_percentages_clinical or {}).items():
        key = f"pct_{_slug(name)}"
        if key in row:
            row[key] = float(value)

    blast_pct = float((findings.report_ready or {}).get("blast_pct", 0.0))
    row["blast_pool_percentage_of_wbc"] = blast_pct
    row["blast_pct"] = blast_pct
    row["n_cells_identified_wbc"] = float(findings.n_cells_identified_wbc)
    row["n_cells_informative"] = float(findings.n_cells_identified_wbc)

    counts = {_slug(k): v for k, v in (findings.cell_counts or {}).items()}
    denom = max(float(findings.n_cells_identified_wbc), 1.0)
    for group, members in CLINICAL_GROUPS.items():
        group_count = sum(float(counts.get(_slug(m), 0)) for m in members)
        row[f"group_{group}_pct"] = round(100.0 * group_count / denom, 4)

    attr_values: dict[str, list[float]] = {name: [] for name in ATTRIBUTE_ORDER}
    cells = []
    if detection_result is not None:
        excluded = {"None", "Unknown"}
        grounded_ids = set((findings.grounding_index or {}).keys())
        if grounded_ids:
            cells = [
                det
                for det in detection_result.detections
                if det.cell_id in grounded_ids and det.cell_type not in excluded
            ]
        else:
            cells = [
                det
                for det in detection_result.detections
                if det.cell_type not in excluded
            ]
    if not cells:
        for rec in (findings.grounding_index or {}).values():
            attrs = rec.get("attributes") or {}
            for attr in ATTRIBUTE_ORDER:
                value = attrs.get(attr)
                if isinstance(value, (int, float)):
                    attr_values[attr].append(float(value))
    else:
        for det in cells:
            for attr in ATTRIBUTE_ORDER:
                value = det.attributes.get(attr, det.attribute_probs.get(attr))
                if isinstance(value, (int, float)):
                    attr_values[attr].append(float(value))

    for attr in ATTRIBUTE_ORDER:
        vals = attr_values[attr]
        slug = _slug(attr)
        row[f"attr_{slug}__positive_pct"] = (
            round(100.0 * sum(v >= 0.5 for v in vals) / len(vals), 4) if vals else 0.0
        )
        row[f"attr_{slug}__mean_prob"] = round(sum(vals) / len(vals), 4) if vals else 0.0

    size_vals: dict[str, list[float]] = {s: [] for s in CELL_SIZE_STATES}
    source = cells if cells else list((findings.grounding_index or {}).values())
    for item in source:
        attrs = item.attributes if hasattr(item, "attributes") else (item.get("attributes") or {})
        for state in CELL_SIZE_STATES:
            v = attrs.get(f"{CELL_SIZE_ATTR}_{state}")
            if isinstance(v, (int, float)):
                size_vals[state].append(float(v))
    for state in CELL_SIZE_STATES:
        vals = size_vals[state]
        row[f"attr_cell_size__{state}_mean_prob"] = round(sum(vals) / len(vals), 4) if vals else 0.0

    if feature_names is None:
        return row
    return {name: float(row.get(name, 0.0)) for name in feature_names}


def build_feature_row_from_stats(
    stats: dict[str, dict[str, Any]],
    patient_id: str,
    feature_names: list[str],
) -> dict[str, float]:
    """Single patient feature row aligned with build_feature_matrix columns."""
    X, _, names, _ = build_feature_matrix(stats, include_qc_features=True)
    if str(patient_id) not in X.index:
        return {name: 0.0 for name in feature_names}
    row = X.loc[str(patient_id)]
    return {name: float(row.get(name, 0.0)) for name in feature_names}
