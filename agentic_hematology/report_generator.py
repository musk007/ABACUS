"""Report generation backends for the agentic hematology orchestrator."""
from __future__ import annotations

import json
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from data_preprocessing.report_protocol import run_report_protocol, labels_from_prediction
from .schemas import AggregatedFindings, GroundedReport, LeukemiaClassification


ROOT = Path(__file__).resolve().parent
WBC_UNIFIED = ROOT / "wbc_unified"
if str(WBC_UNIFIED) not in sys.path:
    sys.path.insert(0, str(WBC_UNIFIED))


def _load_default_cfg() -> dict[str, Any]:
    cfg_path = WBC_UNIFIED / "config" / "default.yaml"
    if cfg_path.is_file():
        import yaml

        return yaml.safe_load(cfg_path.read_text()) or {}
    return {}


class BaseReportGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        findings: AggregatedFindings,
        classification: LeukemiaClassification | None,
        instruction: str | None = None,
    ) -> GroundedReport:
        raise NotImplementedError


class TemplateReportGenerator(BaseReportGenerator):
    """Deterministic report backend using the wbc_unified template."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg or _load_default_cfg()

    def generate(
        self,
        findings: AggregatedFindings,
        classification: LeukemiaClassification | None,
        instruction: str | None = None,
    ) -> GroundedReport:
        return _protocol_report(findings, classification, None, backend="template")


class LocalLLMReportGenerator(BaseReportGenerator):
    """Local HuggingFace/Qwen backend with optional LoRA adapter."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        adapter_path: str | Path | None = None,
        max_new_tokens: int = 1000,
        temperature: float = 0.0,
    ):
        cfg = _load_default_cfg()
        inf = cfg.get("inference", {})
        self.model_path = Path(model_path or os.environ.get("MODEL_PATH") or inf.get("base_model"))
        adapter = adapter_path or os.environ.get("LORA_ADAPTER") or inf.get("lora_adapter")
        self.adapter_path = Path(adapter) if adapter else None
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None

    def attach(self, model, tokenizer) -> "LocalLLMReportGenerator":
        """Reuse an already-loaded model/tokenizer (e.g. from QwenLLMClient)
        so only one copy of the weights is resident on the GPU."""
        self._model = model
        self._tokenizer = tokenizer
        return self

    def generate(
        self,
        findings: AggregatedFindings,
        classification: LeukemiaClassification | None,
        instruction: str | None = None,
    ) -> GroundedReport:
        from report.src.llm_infer import generate_from_messages, load_model_and_tokenizer

        if self._model is None or self._tokenizer is None:
            self._model, self._tokenizer = load_model_and_tokenizer(self.model_path, self.adapter_path)

        def complete(system: str, user: str) -> str:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            return generate_from_messages(
                self._model, self._tokenizer, messages,
                max_new_tokens=self.max_new_tokens, temperature=self.temperature,
            ).strip()

        return _protocol_report(findings, classification, complete, backend="local_llm")


class ClaudeReportGenerator(TemplateReportGenerator):
    """Placeholder kept for CLI compatibility; uses template unless extended."""


class OpenAIReportGenerator(TemplateReportGenerator):
    """Placeholder kept for CLI compatibility; uses template unless extended."""

def _protocol_report(findings, classification, complete, *, backend: str) -> GroundedReport:
    """Template -> LLM enrichment -> validation, identical to reference construction.
    The LLM sees only findings.case_record; appendices are added afterwards."""
    if not findings.case_record:
        raise ValueError("findings.case_record is empty; aggregate() must build it")
    body, info = run_report_protocol(findings.case_id, findings.case_record, complete)
    markdown = _append_quantitative_summary(body, findings)
    markdown = _append_grounding(markdown, findings, classification)
    return GroundedReport(
        markdown=markdown,
        grounding_index=findings.grounding_index,
        backend=backend,
        protocol_markdown=body,
        protocol_info=info,
    )


def _append_quantitative_summary(markdown: str, findings: AggregatedFindings) -> str:
    summary = findings.report_ready
    lines = [markdown.rstrip(), "", "## Quantitative Cell Summary", ""]
    lines.append(
        f"- Fields of view: {summary.get('n_images', findings.n_images)}"
    )
    lines.append(
        f"- Detected cells: {summary.get('n_cells_total', findings.n_cells_total)}"
    )
    lines.append(
        f"- Informative WBCs: {summary.get('n_cells_informative', findings.n_cells_identified_wbc)}"
    )
    lines.append(
        f"- Artefacts/non-WBC detections: {summary.get('n_cells_artifact', 0)}"
    )
    lines.append(
        f"- Blast-equivalent burden is {summary.get('blast_pct', 0.0)}% of informative WBCs"
    )
    lines.append("")
    lines.append("| Cell type | Count | % informative WBCs |")
    lines.append("|---|---:|---:|")
    pct = findings.case_record.get("cell_percentages_clinical") or findings.cell_percentages_clinical
    for cell_type, count in sorted(
        findings.cell_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        v = pct.get(cell_type.lower(), pct.get(cell_type, 0.0))
        lines.append(f"| {cell_type} | {count} | {v:.1f}% |")

    return "\n".join(lines)

_ATTR_DISPLAY = {
    "cell_size": "size", "nuclear_chromatio": "chromatin", "nuclear_shape": "nuclear shape",
    "nucleolus": "nucleoli", "cytoplasm": "cytoplasm",
    "cytoplasmic_basophilia": "basophilia", "cytoplasmic_vacuoles": "vacuoles",
}

def _append_grounding(
    markdown: str,
    findings: AggregatedFindings,
    classification: LeukemiaClassification | None,
    limit: int = 5,
) -> str:
    lines = [markdown.rstrip(), "", "## Agentic Diagnosis"]

    if classification is not None:
        lines.append(
            f"Predicted diagnosis: **{classification.predicted_class}** "
            f"(confidence {classification.confidence:.2f}). "
            f"Rationale: {classification.rationale}."
        )
    else:
        lines.append(
            "No leukemia classification was available for this report."
        )

    lines.extend(["", "## Cell Grounding", ""])

    for cell_id, rec in list(findings.grounding_index.items())[:limit]:
        labels = labels_from_prediction(rec.get("attributes", {}))
        morph = "; ".join(f"{v} {_ATTR_DISPLAY.get(k, k)}" for k, v in labels.items()) \
                or "no attribute predictions"

        lines.append(
            f"- `{cell_id}` in `{rec['image_id']}` bbox={rec['bbox_xyxy']}: "
            f"{rec['cell_type']} ({rec['confidence']:.2f}); {morph}."
        )

    return "\n".join(lines)