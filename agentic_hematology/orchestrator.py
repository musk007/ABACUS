"""
orchestrator.py
===============
The agentic orchestrator. This is the entry point of the pipeline and the component the user (or an upstream
service) talks to. 

It is responsible for coordinating the workflow while deligating all domain-specific reasoning to specialized 
agents. Its responsibilities are:

1. Receives a request containing images, structured findings and /or free 
text instructions.
2. Determine the user's intent and generate an execution plan using either
a rule-based router or an LLM-based orchestration agent
3. Execute the selected sequence of pipeline tools while maintaining a shared 
"PipelineState"
4. Collects outputs, performs optional validation and return a structured
response describing both the execution plan and its results.

Design notes
------------
The orchestrator is deliberately lightweight, with no hematologic or diagnostic
logic itself.

Routing can operate in two modes:

- LLMRouter (default)
    A Qwen-based orchestration agent that analyzes the user's request,
    determines the intent, plans the sequence of pipeline tools, validates
    the execution plan, and dispatches the workflow. If the generated plan
    is invalid or the LLM is unavailable, routing automatically falls back
    to the rule-based router.

- RuleBasedRouter (fallback)
    A deterministic keyword-based router used as a lightweight baseline,
    for debugging, or whenever agentic routing is disabled.

Supported intents
-----------------
FULL_REPORT
    Execute the complete diagnostic workflow, including detection,
    aggregation, leukemia classification, agentic reflection, report
    generation, and report validation.

DETECT_ONLY
    Detect and aggregate white blood cells without downstream diagnosis or
    report generation.

CLASSIFY_ONLY
    Detect, aggregate, and classify the leukemia subtype without producing
    a narrative report.

REPORT_FROM_JSON
    Skip image analysis and generate a diagnostic report from previously
    computed aggregated findings.

EXPLAIN
    Answer natural-language questions about an existing case using the
    structured findings and generated report as grounded context.

Outputs
-------
The orchestrator returns an `OrchestratorResponse` containing:

- the selected intent,
- the execution plan (tool sequence),
- routing rationale,
- the final `PipelineState`,
- optional explanation text (for EXPLAIN),
- validation and reflection metadata for auditability.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .aggregator import aggregate
from .detection_agent import BaseDetectionAgent
from .leukemia_classifier import HybridClassifier
from .pipeline import (
    aggregate_node,
    classify_node,
    detect_node,
    reflect_node,
    report_node,
    validate_node,
)
from .report_generator import BaseReportGenerator, TemplateReportGenerator
from .schemas import AggregatedFindings, DetectionResult, PipelineState
from .validators import (
    LLMOutputValidator,
    ProtocolReportValidator,
    ReportConsistencyValidator,
)

VALID_TOOLS = {
                "detect",
                "aggregate",
                "classify",
                "reflect",
                "report",
                "validate",
                "explain",
            }

# ---------------------------------------------------------------------------
# Lineage guard: deterministic post-report consistency check
# ---------------------------------------------------------------------------
MYELOID_SUBTYPES = {"AML", "APML", "CML"}
LYMPHOID_SUBTYPES = {"ALL", "CLL"}
_COHORT_RE = re.compile(r"\*\*Cohort morphology \(n = \d+ ([^)]+?)\):\*\*")


def lineage_conflict(state: PipelineState) -> str | None:
    """Reason string if the predicted subtype's lineage contradicts the report's cohort, else None."""
    if state.classification is None or state.report is None:
        return None
    pred = state.classification.predicted_class
    if pred not in MYELOID_SUBTYPES | LYMPHOID_SUBTYPES:
        return None
    m = _COHORT_RE.search(state.report.protocol_markdown or state.report.markdown)
    if not m:
        return None
    cohort = m.group(1).strip().lower()
    cohort_lymphoid = "lympho" in cohort
    if cohort_lymphoid == (pred in MYELOID_SUBTYPES):
        pred_lineage = "myeloid" if pred in MYELOID_SUBTYPES else "lymphoid"
        cohort_lineage = "lymphoid" if cohort_lymphoid else "non-lymphoid"
        return (f"lineage conflict: predicted {pred} ({pred_lineage}) but the dominant "
                f"cohort in the report is {cohort} ({cohort_lineage})")
    return None

# ---------------------------------------------------------------------------
# Intents
# ---------------------------------------------------------------------------

class Intent(str, Enum):
    FULL_REPORT = "FULL_REPORT"
    DETECT_ONLY = "DETECT_ONLY"
    CLASSIFY_ONLY = "CLASSIFY_ONLY"
    REPORT_FROM_JSON = "REPORT_FROM_JSON"
    EXPLAIN = "EXPLAIN"

@dataclass
class RouteDecision:
    intent: Intent
    tool_sequence: list[str]
    rationale: str
    precondition_violation_caught: bool = False

@dataclass
class OrchestratorRequest:
    """What the orchestrator receives."""
    case_id: str
    image_paths: list[str] = field(default_factory=list)
    instruction: str | None = None          # free-text from the user
    precomputed_findings: dict | None = None  # for REPORT_FROM_JSON
    forced_intent: Intent | None = None      # bypass routing if set
    dataset_source: str = "lld"


@dataclass
class OrchestratorResponse:
    """What the orchestrator returns."""
    case_id: str
    intent: Intent
    routing_rationale: str
    tool_sequence: list[str]
    state: PipelineState
    answer: str | None = None                # for EXPLAIN intent

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "intent": self.intent.value,
            "routing_rationale": self.routing_rationale,
            "tool_sequence": self.tool_sequence,
            "report_markdown": (self.state.report.markdown if self.state.report else None),
            "leukemia_class": (
                self.state.classification.predicted_class
                if self.state.classification else None
            ),
            "consistency_passed": self.state.consistency_passed,
            "llm_output_passed": self.state.llm_output_passed,
            "template_json_passed": self.state.template_json_passed,
            "numerical_hallucination_passed": self.state.numerical_hallucination_passed,
            "validation_passed": self.state.validation_passed,
            "report_delivery_allowed": self.state.report_delivery_allowed,
            "validation_details": self.state.validation_details,
            "agent_actions": self.state.agent_actions,
            "n_reflect_iterations": self.state.n_reflect_iterations,
            "flagged_for_review": self.state.flagged_for_review,
            "review_reasons": self.state.review_reasons,
            "answer": self.answer,
            "errors": self.state.errors,
        }


# ---------------------------------------------------------------------------
# Intent routers
# ---------------------------------------------------------------------------

def tools_for_intent(intent: Intent) -> list[str]:
    return {
        Intent.FULL_REPORT: ["detect", "aggregate", "classify", "reflect", "report", "validate"],
        Intent.DETECT_ONLY: ["detect", "aggregate"],
        Intent.CLASSIFY_ONLY: ["detect", "aggregate", "classify"],
        Intent.REPORT_FROM_JSON: ["classify", "reflect", "report", "validate"],
        Intent.EXPLAIN: ["explain"],
    }[intent]




class RuleBasedRouter:
    """Keyword-driven intent routing. Deterministic, no API cost."""

    # Keyword → intent. Checked in order; first match wins.
    _PATTERNS: list[tuple[re.Pattern, Intent]] = [
        # CLASSIFY_ONLY — "only classify", "just tell me the subtype", etc.
        # Checked before DETECT_ONLY so "classify" wins over a stray "cells".
        (re.compile(r"\b(only|just)\b.*\b(classif|diagnos|subtype|leukemia type|which leukemia)\w*", re.I), Intent.CLASSIFY_ONLY),
        (re.compile(r"\b(classif|diagnos|subtype)\w*\b.*\b(only|just)\b", re.I), Intent.CLASSIFY_ONLY),
        # DETECT_ONLY — "only detect", "just count the cells", etc.
        (re.compile(r"\b(only|just)\b.*\b(detect|count|localiz|localis|find cells?)\w*", re.I), Intent.DETECT_ONLY),
        (re.compile(r"\b(detect|count|localiz|localis)\w*\b.*\b(only|just)\b", re.I), Intent.DETECT_ONLY),
        # EXPLAIN — questions / justification requests.
        (re.compile(r"\b(explain|why|how come|justify|rationale)\b", re.I), Intent.EXPLAIN),
        (re.compile(r"\bwhat does\b.*\bmean\b", re.I), Intent.EXPLAIN),
        # FULL_REPORT — anything mentioning a report.
        (re.compile(r"\breport\b", re.I), Intent.FULL_REPORT),
    ]

    def _safe_decision(self, intent: Intent, request: OrchestratorRequest, rationale: str):
        tools = tools_for_intent(intent)

        if "detect" in tools and not request.image_paths:
            return RouteDecision(
                intent=Intent.EXPLAIN,
                tool_sequence=["explain"],
                rationale=rationale + "; blocked because detection requires image inputs",
                precondition_violation_caught=True,
            )
            

        if (
            "detect" not in tools
            and request.precomputed_findings is None
            and intent != Intent.EXPLAIN
        ):
            return RouteDecision(
                intent=Intent.EXPLAIN,
                tool_sequence=["explain"],
                rationale=rationale + "; blocked because precomputed findings are missing",
                precondition_violation_caught=True,
            )

        return RouteDecision(
            intent=intent,
            tool_sequence=tools,
            rationale=rationale,
        )
    

    def route(self, request: OrchestratorRequest) -> RouteDecision:
        if request.forced_intent:
            intent = request.forced_intent
            return RouteDecision(
                intent=intent,
                tool_sequence=tools_for_intent(intent),
                rationale="forced by caller",
            )

        if request.precomputed_findings is not None and not request.image_paths:
            intent = Intent.REPORT_FROM_JSON
            return RouteDecision(
                intent=intent,
                tool_sequence=tools_for_intent(intent),
                rationale="precomputed findings supplied, no images",
            )

        text = (request.instruction or "").strip()
        if not text:
            if request.image_paths:
                intent = Intent.FULL_REPORT
                rationale = "no instruction; images present → full report"
            else:
                intent = Intent.EXPLAIN
                rationale = "no instruction and no images"

            return RouteDecision(
                intent=intent,
                tool_sequence=tools_for_intent(intent),
                rationale=rationale,
            )

        for pattern, intent in self._PATTERNS:
            if pattern.search(text):
                return self._safe_decision(
                    intent,
                    request,
                    rationale=f"matched rule {pattern.pattern!r}",
                )

        intent = Intent.FULL_REPORT
        return self._safe_decision(
                    intent,
                    request,
                    rationale="no specific rule matched; defaulting to full report",
                )



class LLMRouter:
    """
    LLM-backed intent classifier for ambiguous instructions. Falls back to
    the rule router if the LLM is unavailable or returns garbage.
    """

    def __init__(self, llm_complete, fallback: RuleBasedRouter | None = None, canonical_tools: bool = True):
        """
        :param llm_complete: a callable (system:str, user:str) -> str.
        """
        self.llm_complete = llm_complete
        self.fallback = fallback or RuleBasedRouter()
        self.canonical_tools = canonical_tools
        

    _SYSTEM = """
        You are the orchestration agent for an agentic hematology system.

        Your job is to decide:

        1. The user's intent.
        2. The sequence of tools that should be executed.

        Available tools:
        - detect
        - aggregate
        - classify
        - reflect
        - report
        - validate
        - explain

        Rules:
        - The "intent" field MUST be exactly one of: FULL_REPORT, DETECT_ONLY,
          CLASSIFY_ONLY, REPORT_FROM_JSON, EXPLAIN.
        - Only use the listed tools.
        - Preserve logical ordering.
        - Use explain only for question answering.
        - If precomputed findings are supplied, do not use detect or aggregate.
        - Return ONLY valid JSON.

        Example:

        {
        "intent": "FULL_REPORT",
        "tool_sequence": [
            "detect",
            "aggregate",
            "classify",
            "reflect",
            "report",
            "validate"
        ],
        "rationale": "The user requested a complete diagnosis."
        }
        """
    

    def route(self, request: OrchestratorRequest) -> RouteDecision:
        if request.forced_intent:
            return RouteDecision(
                intent=request.forced_intent,
                tool_sequence=tools_for_intent(request.forced_intent),
                rationale="forced by caller",
            )
        if request.precomputed_findings is not None and not request.image_paths:
            intent = Intent.REPORT_FROM_JSON
            return RouteDecision(
                intent=intent,
                tool_sequence=tools_for_intent(intent),
                rationale="precomputed findings supplied, no images",
            )

        text = (request.instruction or "").strip()
        if not text:
            return self.fallback.route(request)

        try:
            print("Running LLM tool router...", flush=True)
            raw = self.llm_complete(self._SYSTEM, text).strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON object found in router output: {raw}")
            payload = json.loads(match.group(0))

            intent = Intent(payload["intent"])
            tools = tools_for_intent(intent) if self.canonical_tools else payload["tool_sequence"]
            rationale = payload.get("rationale", "LLM router")

        
            # ---------- Validate tools ----------

            if not isinstance(tools, list):
                raise ValueError("tool_sequence must be a list.")

            if not all(tool in VALID_TOOLS for tool in tools):
                raise ValueError(f"Unknown tool in plan: {tools}")

            if not tools:
                raise ValueError("Empty tool_sequence.")
            if len(set(tools)) != len(tools):
                raise ValueError(f"Duplicate tools in plan: {tools}")
            if (intent == Intent.EXPLAIN) != (tools == ["explain"]):
                raise ValueError(f"Intent {intent.value} inconsistent with tools {tools}")
            order = ["detect", "aggregate", "classify", "reflect", "report", "validate"]
            idx = [order.index(t) for t in tools if t in order]
            if idx != sorted(idx):
                raise ValueError(f"Tools out of order: {tools}")

            if "explain" in tools:
                if len(tools) > 1:
                    raise ValueError("EXPLAIN should not be combined with other tools.")
            else:
                if "detect" in tools and not request.image_paths:
                    raise ValueError("Plan requires detection but no image paths were supplied.")
                if "detect" not in tools and request.precomputed_findings is None:
                    raise ValueError("Plan skips detection but no precomputed findings were supplied.")
            
            # ---------- Validate ordering ----------

            if (
                "aggregate" in tools
                and "detect" not in tools
                and request.precomputed_findings is None
            ):
                raise ValueError("Aggregation requires detection or precomputed findings.")

            if (
                "classify" in tools
                and "aggregate" not in tools
                and request.precomputed_findings is None
            ):
                raise ValueError(
                    "Classification requires aggregation unless precomputed findings are supplied."
                )

            if "report" in tools and "classify" not in tools:
                raise ValueError("Report generation requires classification.")

            if "validate" in tools and "report" not in tools:
                raise ValueError("Validation requires report generation.")

            if "explain" in tools and len(tools) > 1:
                raise ValueError("EXPLAIN should not be combined with other tools.")

            # ------------------------------------------------

            return RouteDecision(
                intent=intent,
                tool_sequence=tools,
                rationale=rationale,
            )

        except Exception as e:
            fallback_decision = self.fallback.route(request)
            fallback_decision.precondition_violation_caught = True
            fallback_decision.rationale = (
                f"LLM plan rejected due to precondition/order violation: {e}; "
                f"fallback: {fallback_decision.rationale}"
            )
            return fallback_decision

def build_pre_reflection_snapshot(state: PipelineState) -> dict[str, Any]:
    snapshot = {
        "conf_threshold": state.conf_threshold,
        "detection_and_attribute_classification": None,
        "aggregation": None,
        "leukemia_classification": None,
    }

    if state.detection_result is not None:
        det = state.detection_result
        snapshot["detection_and_attribute_classification"] = {
            "case_id": det.case_id,
            "n_images": det.n_images,
            "n_detections": len(det.detections),
            "detections": [
                {
                    "cell_id": d.cell_id,
                    "image_id": d.image_id,
                    "bbox_xyxy": [float(v) for v in d.bbox_xyxy],
                    "cell_type": d.cell_type,
                    "objectness": float(d.objectness),
                    "cell_type_prob": float(d.cell_type_prob) if d.cell_type_prob is not None else None,
                    "attributes": d.attributes,
                    "attribute_probs": {k: float(v) for k, v in d.attribute_probs.items()},
                }
                for d in det.detections
            ],
        }

    if state.findings is not None:
        f = state.findings
        snapshot["aggregation"] = {
            "case_id": f.case_id,
            "n_images": f.n_images,
            "n_cells_total": f.n_cells_total,
            "n_cells_identified_wbc": f.n_cells_identified_wbc,
            "cell_counts": f.cell_counts,
            "cell_percentages_all": f.cell_percentages_all,
            "cell_percentages_clinical": f.cell_percentages_clinical,
            "attributes": f.attributes,
            "report_ready": f.report_ready,
            "grounding_index": f.grounding_index,
        }

    if state.classification is not None:
        clf = state.classification
        snapshot["leukemia_classification"] = {
            "predicted_class": clf.predicted_class,
            "confidence": float(clf.confidence),
            "rationale": clf.rationale,
            "scores": {k: float(v) for k, v in clf.scores.items()},
        }

    return snapshot

# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Receives requests, routes by intent, dispatches to pipeline stages.

    Wire it up with the agents you want:
        orch = Orchestrator(
            detector=TwoStageDetectionAgent(...),
            classifier=HybridClassifier(...),
            report_generator=TemplateReportGenerator(),  # or Claude/OpenAI
            router=RuleBasedRouter(),
        )
        resp = orch.handle(OrchestratorRequest(
            case_id="12",
            image_paths=[...],
            instruction="Generate a full diagnostic report",
        ))
    """

    def __init__(
        self,
        detector: BaseDetectionAgent,
        classifier: HybridClassifier | None = None,
        report_generator: BaseReportGenerator | None = None,
        router: Any | None = None,
        enable_reflect: bool = True,
        enable_validate: bool = True,
        validate_failure_policy: str = "strip",
        llm_explain=None,  # optional callable (system, user) -> str for EXPLAIN
        reflection_agent=None,           # ReflectionAgent | None — enables agency
        max_reflect_iterations: int = 2,
        enable_lineage_guard: bool = True,
    ):
        # from agentic_hematology.test_reflection_agent import FakeReflectionAgent

        # self.reflection_agent = FakeReflectionAgent()
        self.detector = detector
        self.classifier = classifier or HybridClassifier()
        self.report_generator = report_generator or TemplateReportGenerator()
        self.router = router or RuleBasedRouter()
        self.enable_reflect = enable_reflect
        self.enable_validate = enable_validate
        self.validate_failure_policy = validate_failure_policy
        self.llm_explain = llm_explain
        self.reflection_agent = reflection_agent
        self.max_reflect_iterations = max_reflect_iterations
        self.enable_lineage_guard = enable_lineage_guard
        self._consistency = ReportConsistencyValidator()
        self._llm_guard = LLMOutputValidator()
        # The protocol validator covers both consistency and numeric containment.
        self._template_json = ProtocolReportValidator()
        self._numerical_hallucination = None

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def handle(self, request: OrchestratorRequest) -> OrchestratorResponse:
        decision = self.router.route(request)
        state = PipelineState(
            case_id=request.case_id,
            image_paths=request.image_paths,
            text_input=request.instruction,
            dataset_source=request.dataset_source,
        )

        # If caller supplied precomputed findings, inject them.
        if request.precomputed_findings is not None:
            state.findings = self._findings_from_dict(
                request.case_id, request.precomputed_findings
            )

        answer = None
        if decision.intent == Intent.EXPLAIN:
            answer = self._run_explain(state, request)
        else:
            state = self._run_tools(state, decision.tool_sequence, request)

        return OrchestratorResponse(
            case_id=request.case_id,
            intent=decision.intent,
            routing_rationale=decision.rationale,
            tool_sequence=decision.tool_sequence,
            state=state,
            answer=answer,
        )

    # ------------------------------------------------------------------
    # Intent handlers
    # ------------------------------------------------------------------



    def _run_tools(self, state, tools, request):
        FINDINGS_REQUIRED = {"classify", "reflect", "report", "validate"}

        for tool in tools:
            if tool in FINDINGS_REQUIRED and state.findings is None:
                state.errors.append(
                    f"Tool '{tool}' requires aggregated findings, but none are available; "
                    "halting tool sequence."
                )
                break
            if tool == "detect":
                state = detect_node(state, self.detector)

            elif tool == "aggregate":
                state = aggregate_node(state)

            elif tool == "classify":
                state = classify_node(state, self.classifier)

            elif tool == "reflect":
                state.pre_reflection_snapshot = build_pre_reflection_snapshot(state)

                
                if self.enable_reflect:
                    state = reflect_node(
                        state,
                        agent=self.reflection_agent,
                        classifier=self.classifier,
                        max_iterations=self.max_reflect_iterations,
                    )

            elif tool == "report":
                state = report_node(state, self.report_generator)

            elif tool == "validate":
                if self.enable_validate:
                    state = validate_node(
                        state,
                        consistency_validator=self._consistency,
                        llm_output_validator=self._llm_guard,
                        template_json_validator=self._template_json,
                        numerical_hallucination_validator=self._numerical_hallucination,
                        failure_policy=self.validate_failure_policy,
                        report_generator=self.report_generator,
                    )

        if self.enable_lineage_guard and "report" in tools:
            reason = lineage_conflict(state)
            if reason:
                state.flagged_for_review = True
                state.review_reasons.append(reason)
                state.agent_actions.append({
                    "iteration": len(state.agent_actions) + 1, "action": "lineage_guard_flag",
                    "reason": reason, "conf_threshold": None, "error": False,
                })
                state.report.markdown += (
                    "\n\n> **⚠ Flagged for mandatory human review by the lineage guard.** "
                    f"Reason: {reason}"
                )
        return state

    def _run_explain(
        self, state: PipelineState, request: OrchestratorRequest
    ) -> str | None:
        """
        Answer a free-text question. If findings are available, run the full
        pipeline first so the LLM has grounded context; then answer.
        """
        if state.findings is None and state.image_paths:
            state = self._run_tools(
                state,
                ["detect", "aggregate", "classify", "reflect", "report", "validate"],
                request,
            )

        if self.llm_explain is None:
            return (
                "EXPLAIN intent requires an LLM. Provide `llm_explain` to the "
                "Orchestrator. Structured findings are available in the state."
            )

        context = ""
        if state.findings is not None:
            context = json.dumps(state.findings.report_ready, indent=2)
        if state.report is not None:
            context += "\n\nREPORT:\n" + state.report.markdown

        system = (
            "You are a hematopathology assistant. Answer the user's question "
            "using ONLY the structured findings and report provided. Do not "
            "invent numbers or findings."
        )
        user = f"QUESTION: {request.instruction}\n\nCONTEXT:\n{context}"
        try:
            return self.llm_explain(system, user)
        except Exception as e:
            return f"EXPLAIN failed: {e}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _findings_from_dict(case_id: str, payload: dict) -> AggregatedFindings:
        """Build an AggregatedFindings from a precomputed JSON payload."""
        return AggregatedFindings(
            case_id=case_id,
            n_images=payload.get("n_images", 0),
            n_cells_total=payload.get("n_cells_total", 0),
            n_cells_identified_wbc=payload.get("n_cells_identified_wbc", 0),
            cell_counts=payload.get("cell_counts", {}),
            cell_percentages_all=payload.get("cell_percentages_all", {}),
            cell_percentages_clinical=payload.get("cell_percentages_clinical", {}),
            attributes=payload.get("attributes", {}),
            report_ready=payload.get("report_ready", {}),
            grounding_index=payload.get("grounding_index", {}),
            case_record=payload if "report_ready" in payload and "cell_percentages_clinical" in payload else {},
        )