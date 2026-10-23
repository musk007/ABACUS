"""Lightweight validation for generated reports."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any




@dataclass
class ValidationResult:
    passed: bool
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class ReportConsistencyValidator:
    """The predicted class appears in the Agentic Diagnosis appendix, not in the
    impression, which is deliberately lineage-level."""
    def validate(self, state) -> ValidationResult:
        if state.report is None:
            return ValidationResult(False, "missing report")
        if state.classification and state.classification.predicted_class not in state.report.markdown:
            return ValidationResult(
                False,
                "predicted class is not mentioned in report markdown",
            )
        return ValidationResult(True, "ok")

class ProtocolReportValidator:
    """Same validation as reference construction: template numbers against the record
    (0.11 pp tolerance) and numeric containment of the LLM section. Runs on the report
    body, before the pipeline appendices."""

    def validate(self, state) -> ValidationResult:
        if state.report is None or state.findings is None or not state.findings.case_record:
            return ValidationResult(False, "missing report or protocol record")
        info = state.report.protocol_info.get("validation")
        if info is None:
            from data_preprocessing.report_protocol import validate
            r = validate(state.case_id, state.findings.case_record, state.report.protocol_markdown)
            info = {
                "passed": r.passed,
                "parse_errors": r.parse_errors,
                "consistency_issues": r.consistency_issues,
                "hallucination_issues": r.hallucination_issues,
            }
        if info["passed"]:
            return ValidationResult(True, "ok", details=info)
        issues = info["parse_errors"] + info["consistency_issues"] + info["hallucination_issues"]
        return ValidationResult(False, "; ".join(issues[:4]), details=info)



class LLMOutputValidator:
    _BANNED = ("as an ai language model", "i cannot diagnose")

    def validate(self, markdown: str) -> ValidationResult:
        text = markdown.strip()
        if not text:
            return ValidationResult(False, "empty report")
        lowered = text.lower()
        if any(term in lowered for term in self._BANNED):
            return ValidationResult(False, "contains generic refusal text")
        return ValidationResult(True, "ok")




