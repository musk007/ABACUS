"""Patient-level aggregation for detector and attribute outputs."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .schemas import AggregatedFindings, DetectionResult
from data_preprocessing.report_protocol import (
    add_cell,
    build_case_record,
    labels_from_prediction,
    new_store,
    normalise_cell_type,
)

BLAST_CLASSES = {"Myeloblast", "Lymphoblast", "Monoblast", "Abnormal promyelocyte"}
EXCLUDED_CLASSES = {"None", "Unknown"}

ATTR_VALUE_MAPS: dict[str, tuple[str, str]] = {
    # attribute name -> (label when prob < 0.5, label when prob >= 0.5)
    "Nuclear_Chromatin":      ("open", "coarse"),
    "Nuclear_Shape":          ("regular", "irregular"),
    "Nucleolus":              ("inconspicuous", "prominent"),
    "Cytoplasm":              ("scanty", "abundant"),
    "Cytoplasmic_Basophilia": ("slight", "moderate"),
    "Cytoplasmic_Vacuoles":   ("absent", "prominent"),
}
CELL_SIZE_STATES = ("small", "medium", "large")

# GT key name -> live attribute name
GT_ATTR_KEY = {
    "cell_size": "Cell_Size",
    "nuclear_chromatio": "Nuclear_Chromatin",
    "nuclear_shape": "Nuclear_Shape",
    "nucleolus": "Nucleolus",
    "cytoplasm": "Cytoplasm",
    "cytoplasmic_basophilia": "Cytoplasmic_Basophilia",
    "cytoplasmic_vacuoles": "Cytoplasmic_Vacuoles",
}
CELL_TYPE_PRIORITY: dict[str, int] = {
    "Abnormal promyelocyte": 1,
    "Myeloblast":            2,
    "Lymphoblast":           3,
    "Monoblast":             4,
    "Promonocyte":           5,
    "Atypical lymphocyte":   6,
    "Lymphocyte":            7,
}
COHORT_ELIGIBLE = set(CELL_TYPE_PRIORITY)
DEFAULT_PRIORITY = 50


def _select_cohort_type(counts: dict[str, int]) -> str | None:
    candidates = {ct: counts.get(ct, 0) for ct in COHORT_ELIGIBLE if counts.get(ct, 0) > 0}
    if not candidates:
        return None
    max_count = max(candidates.values())
    tied = [ct for ct, v in candidates.items() if v == max_count]
    return min(tied, key=lambda ct: CELL_TYPE_PRIORITY.get(ct, DEFAULT_PRIORITY))

def _blast_morphology(cells) -> tuple[dict[str, dict[str, Any]], int]:
    """Per-attribute {dominant, dominance_pct} over the dominant blast-like cohort,
    matching the ground-truth `report_ready.blast_morphology` schema."""
    if not cells:
        return {}, 0

    out: dict[str, dict[str, Any]] = {}
    n = len(cells)

    # Cell_Size: 3-class, from the per-state probabilities
    size_votes = Counter()
    for c in cells:
        probs = [c.attributes.get(f"Cell_Size_{s}") for s in CELL_SIZE_STATES]
        if all(isinstance(p, (int, float)) for p in probs):
            size_votes[CELL_SIZE_STATES[int(max(range(3), key=lambda i: probs[i]))]] += 1
    if size_votes:
        label, k = size_votes.most_common(1)[0]
        out["cell_size"] = {"dominant": label, "dominance_pct": round(100.0 * k / n, 2)}

    # Binary attributes
    for gt_key, attr in GT_ATTR_KEY.items():
        if gt_key == "cell_size":
            continue
        votes = Counter()
        for c in cells:
            v = c.attributes.get(attr)
            if isinstance(v, (int, float)):
                votes[ATTR_VALUE_MAPS[attr][int(v >= 0.5)]] += 1
        if votes:
            label, k = votes.most_common(1)[0]
            out[gt_key] = {"dominant": label, "dominance_pct": round(100.0 * k / sum(votes.values()), 2)}

    return out, n


def aggregate(result: DetectionResult, conf_threshold: float = 0.25) -> AggregatedFindings:
    cells = [d for d in result.detections if d.objectness >= conf_threshold]
    informative = [d for d in cells if d.cell_type not in EXCLUDED_CLASSES]

    counts = Counter(d.cell_type for d in informative)
    all_counts = Counter(d.cell_type for d in cells)

    total_inf = sum(counts.values())
    total_all = len(cells)

    clinical_pct = {
        name: round(100.0 * count / total_inf, 1)
        for name, count in counts.most_common()
    } if total_inf else {}

    all_pct = {
        name: round(100.0 * count / max(total_all, 1), 1)
        for name, count in all_counts.most_common()
    }

    morphology = _morphology_cohort(informative)
    blast_n = sum(counts.get(name, 0) for name in BLAST_CLASSES)
    blast_pct = round(100.0 * blast_n / total_inf, 1) if total_inf else 0.0
    grounding_index = _grounding_index(informative)

    dominant_cell_type = counts.most_common(1)[0][0] if counts else "none"
    dominant_cell_pct = clinical_pct.get(dominant_cell_type, 0.0)
    cohort_type = _select_cohort_type(counts)
    cohort = [d for d in informative if d.cell_type == cohort_type] if cohort_type else []
    blast_morphology, n_cohort = _blast_morphology(cohort)

    report_ready: dict[str, Any] = {
        "patient_id": result.case_id,
        "source": "agentic_orchestrator",
        "n_images": result.n_images,
        "image_stems": sorted({d.image_id.rsplit(".", 1)[0] for d in result.detections}),
        "n_cells_total": total_all,
        "n_cells_informative": total_inf,
        "n_cells_artifact": max(0, total_all - total_inf),
        "class_counts": dict(counts),
        "all_class_counts": dict(all_counts),
        "differential_pct": clinical_pct,
        "all_detection_pct": all_pct,
        "blast_pct": blast_pct,
        "flags": {
            "blasts_present": blast_n > 0,
            "blast_threshold_met": blast_pct >= 20.0,
        },
        "morphology_cohort": morphology,
        "grounding_index": grounding_index,
        "qc": {
            "mean_det_conf": round(
                sum(float(d.objectness) for d in informative) / total_inf, 3
            ) if total_inf else 0.0,
            "pct_class_none": round(
                100.0 * (total_all - total_inf) / max(total_all, 1), 1
            ),
            "n_fields_of_view": result.n_images,
            "n_annotated_cells": total_all,
            "n_identified_wbc": total_inf,
            "n_artifacts": max(0, total_all - total_inf),
            "n_cells_in_cohort": n_cohort,
            "low_cell_count_warning": total_inf < 30,
            "sparse_annotation_skew_warning": blast_pct < 20.0 and total_inf < 30,
        },
        "blast_pool_percentage_of_wbc": blast_pct,
        "dominant_cell_type": dominant_cell_type.lower(),
        "cohort_cell_type": cohort_type.lower() if cohort_type else None,
        "dominant_cell_pct": dominant_cell_pct,
        "blast_morphology": blast_morphology,
        "diagnostic_flags": {
            "blasts_present": blast_n > 0,
            "blast_threshold_met": blast_pct >= 20.0,
            "abnormal_promyelocytes_present": counts.get("Abnormal promyelocyte", 0) > 0,
            "atypical_lymphocytes_present": counts.get("Atypical lymphocyte", 0) > 0,
            "basophilia_present": clinical_pct.get("Basophil", 0.0) >= 2.0,
            "eosinophilia_present": clinical_pct.get("Eosinophil", 0.0) >= 5.0,
            "monocytosis_present": clinical_pct.get("Monocyte", 0.0) >= 10.0,
            "left_shifted_myeloid": (
                100.0 * sum(counts.get(c, 0) for c in
                            ("Promonocyte", "Myelocyte", "Metamyelocyte")) / max(total_inf, 1)
            ) >= 10.0,
        },
    }
    # Same record construction as the reference reports, applied to predicted cells.
    store = new_store()
    for d in cells:
        add_cell(store, normalise_cell_type(d.cell_type), labels_from_prediction(d.attributes))
    case_record = build_case_record(store, n_images=result.n_images, diagnosis=None)

    # Flags, cohort, and alerts seen by the reflection agent come from the same record.
    rec_rr = case_record["report_ready"]
    report_ready["diagnostic_flags"] = rec_rr["diagnostic_flags"]
    report_ready["blast_morphology"] = rec_rr["blast_morphology"]
    report_ready["cohort_cell_type"] = rec_rr["cohort_cell_type"]
    report_ready["qc"].update({
        "n_cells_in_cohort": rec_rr["qc"]["n_cells_in_cohort"],
        "sparse_annotation_skew_warning": rec_rr["qc"]["sparse_annotation_skew_warning"],
        "cohort_selection_ambiguous": rec_rr["qc"]["cohort_selection_ambiguous"],
        "differential_alerts": rec_rr["qc"]["differential_alerts"],
        "requires_review": rec_rr["qc"]["requires_review"],
    })

    return AggregatedFindings(
        case_id=result.case_id,
        n_images=result.n_images,
        n_cells_total=total_all,
        n_cells_identified_wbc=total_inf,
        cell_counts=dict(counts),
        cell_percentages_all=all_pct,
        cell_percentages_clinical=clinical_pct,
        attributes=morphology,
        report_ready=report_ready,
        grounding_index=grounding_index,
        case_record=case_record,
    )


def _morphology_cohort(cells) -> dict[str, dict[str, Any]]:
    by_class: dict[str, list] = defaultdict(list)
    for cell in cells:
        by_class[cell.cell_type].append(cell)

    out: dict[str, dict[str, Any]] = {}
    for cell_type, group in by_class.items():
        attr_values: dict[str, list[float]] = defaultdict(list)
        for cell in group:
            for name, value in cell.attributes.items():
                if name == "class_id" or not isinstance(value, (int, float)):
                    continue
                attr_values[name].append(float(value))
        out[cell_type] = {
            "n": len(group),
            "attr_pos_rate": {
                name: round(sum(values) / len(values), 4)
                for name, values in sorted(attr_values.items())
                if values
            },
        }
    return out


def _grounding_index(cells) -> dict[str, Any]:
    return {
        cell.cell_id: {
            "image_id": cell.image_id,
            "bbox_xyxy": [round(float(v), 2) for v in cell.bbox_xyxy],
            "cell_type": cell.cell_type,
            "confidence": round(float(cell.objectness), 4),
            "attributes": {
                k: v for k, v in cell.attributes.items() if k != "class_id"
            },
        }
        for cell in cells
    }
