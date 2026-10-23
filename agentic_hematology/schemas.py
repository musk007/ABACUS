"""Shared data contracts for the agentic hematology pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Detection:
    cell_id: str
    image_id: str
    bbox_xyxy: tuple[float, float, float, float]
    cell_type: str
    objectness: float
    cell_type_prob: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    attribute_probs: dict[str, float] = field(default_factory=dict)

    def to_wbc_cell(self) -> dict[str, Any]:
        class_id = self.attributes.get("class_id")
        return {
            "cell_id": self.cell_id,
            "image_id": self.image_id,
            "xyxy": [float(v) for v in self.bbox_xyxy],
            "conf": float(self.objectness),
            "class_id": int(class_id) if class_id is not None else None,
            "class_name": self.cell_type,
            "attributes": {
                k: float(v)
                for k, v in self.attributes.items()
                if k != "class_id" and isinstance(v, (int, float))
            },
            "attributes_bin": {
                k: int(v >= 0.5) if isinstance(v, (int, float)) else int(bool(v))
                for k, v in self.attributes.items()
                if k != "class_id"
            },
            "attribute_probs": self.attribute_probs,
        }


@dataclass
class DetectionResult:
    case_id: str
    n_images: int
    detections: list[Detection] = field(default_factory=list)


@dataclass
class AggregatedFindings:
    case_id: str
    n_images: int
    n_cells_total: int
    n_cells_identified_wbc: int
    cell_counts: dict[str, int]
    cell_percentages_all: dict[str, float]
    cell_percentages_clinical: dict[str, float]
    attributes: dict[str, Any]
    report_ready: dict[str, Any]
    grounding_index: dict[str, Any] = field(default_factory=dict)
    case_record: dict[str, Any] = field(default_factory=dict)


@dataclass
class LeukemiaClassification:
    predicted_class: str
    confidence: float
    rationale: str
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class GroundedReport:
    markdown: str
    grounding_index: dict[str, Any] = field(default_factory=dict)
    backend: str = "template"
    protocol_markdown: str = ""                                   # report body before appendices
    protocol_info: dict[str, Any] = field(default_factory=dict)   # enrichment + validation log


@dataclass
class PipelineState:
    case_id: str
    image_paths: list[str] = field(default_factory=list)
    text_input: str | None = None
    detection_result: DetectionResult | None = None
    findings: AggregatedFindings | None = None
    classification: LeukemiaClassification | None = None
    report: GroundedReport | None = None
    consistency_passed: bool | None = None
    llm_output_passed: bool | None = None
    template_json_passed: bool | None = None
    numerical_hallucination_passed: bool | None = None
    validation_passed: bool | None = None
    report_delivery_allowed: bool | None = None
    validation_details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    # --- agentic control trace (populated by the reflection loop) ---
    conf_threshold: float = 0.25            # current aggregation threshold
    agent_actions: list[dict] = field(default_factory=list)  # decision trace
    n_reflect_iterations: int = 0
    flagged_for_review: bool = False
    review_reasons: list[str] = field(default_factory=list)
    pre_reflection_snapshot: dict[str, Any] = field(default_factory=dict)
    dataset_source: str = "lld"
    routing_notes: list[str] = field(default_factory=list)