"""Composable pipeline nodes used by the orchestrator."""
from __future__ import annotations

from .aggregator import aggregate
from .agent_controller import (
    AgentAction,
    ReflectionAgent,
    build_case_state,
)
from .detection_agent import BaseDetectionAgent
from .leukemia_classifier import HybridClassifier
from .report_generator import BaseReportGenerator
from .schemas import PipelineState


def detect_node(state: PipelineState, detector: BaseDetectionAgent) -> PipelineState:
    if not state.image_paths:
        state.errors.append("Detection requested but no image paths were supplied")
        return state
    state.detection_result = detector.detect(state.case_id, state.image_paths)
    return state


def aggregate_node(state: PipelineState) -> PipelineState:
    if state.detection_result is None:
        state.errors.append("Aggregation requested before detection")
        return state
    # Honour the (possibly agent-adjusted) confidence threshold.
    state.findings = aggregate(state.detection_result, conf_threshold=state.conf_threshold)
    return state


def classify_node(state: PipelineState, classifier: HybridClassifier) -> PipelineState:
    if state.findings is None:
        state.errors.append("Classification requested before aggregation")
        return state
    state.classification = classifier.classify(
        state.findings,
        detection_result=state.detection_result,
    )
    return state


def reflect_node(
    state: PipelineState,
    *,
    agent: ReflectionAgent | None = None,
    classifier: HybridClassifier | None = None,
    max_iterations: int = 4,
) -> PipelineState:
    """
    Agentic reflection loop.

    The reflection agent inspects the intermediate findings + classification
    and decides one process action:

      - proceed          → exit the loop
      - re_aggregate     → re-run aggregation at the threshold already
                           validated by agent_controller, then re-classify
      - flag_for_review  → mark the case for mandatory human review and exit

    The reflection agent does not change the diagnosis. The deterministic
    classifier remains authoritative.
    """
    if agent is None or state.findings is None:
        return state

    re_aggregate_count = 0
    max_reaggregations = max(0, max_iterations - 1)

    for iteration in range(1, max_iterations + 1):
        state.n_reflect_iterations = iteration

        case_state = build_case_state(
            state.findings,
            state.classification,
            dataset_source=state.dataset_source,
        )
        # breakpoint()

        decision = agent.decide(
            case_state,
            re_aggregate_used=(re_aggregate_count >= max_reaggregations), # True if we have re-aggregated even once
            current_conf_threshold=state.conf_threshold,
            iteration=iteration,
        )
        # breakpoint()

        state.agent_actions.append({"iteration": iteration, **decision.to_dict()})

        if decision.action == AgentAction.PROCEED:
            break

        if decision.action == AgentAction.FLAG_FOR_REVIEW:
            state.flagged_for_review = True
            state.review_reasons.append(decision.reason)
            break

        if decision.action == AgentAction.RE_AGGREGATE:
            if decision.conf_threshold is None:
                state.flagged_for_review = True
                state.review_reasons.append(
                    "Re-aggregation requested without a valid confidence threshold"
                )
                break

            state.conf_threshold = decision.conf_threshold
            re_aggregate_count += 1

            state = aggregate_node(state)

            if classifier is not None:
                state = classify_node(state, classifier)

            continue

    else:
        state.flagged_for_review = True
        state.agent_actions.append({
            "iteration": max_iterations,
            "action": AgentAction.FLAG_FOR_REVIEW.value,
            "reason": f"reflection loop hit max_iterations={max_iterations} without converging",
            "conf_threshold": None,
        })
        state.review_reasons.append(
            f"reflection loop hit max_iterations={max_iterations} without converging"
        )

    return state


def report_node(state: PipelineState, report_generator: BaseReportGenerator) -> PipelineState:
    if state.findings is None:
        state.errors.append("Report requested before aggregation")
        return state
    state.report = report_generator.generate(
        state.findings,
        state.classification,
        instruction=state.text_input,
    )
    # Surface the agent's review decision in the report markdown so it is
    # never lost downstream.
    if state.flagged_for_review and state.report is not None:
        reasons = "; ".join(state.review_reasons) or "agent flagged for review"
        banner = (
            "\n\n> **⚠ Flagged for mandatory human review by the reflection "
            f"agent.** Reason(s): {reasons}\n"
        )
        state.report.markdown = state.report.markdown.rstrip() + banner
    return state


def validate_node(
    state: PipelineState,
    *,
    consistency_validator,
    llm_output_validator,
    template_json_validator=None,
    numerical_hallucination_validator=None,
    failure_policy: str = "strip",
    report_generator: BaseReportGenerator | None = None,
) -> PipelineState:
    if state.report is None:
        state.errors.append("Validation requested before report generation")
        return state

    checks: list[tuple[str, Any]] = [
        ("consistency", consistency_validator.validate(state)),
        ("llm_output", llm_output_validator.validate(state.report.markdown)),
    ]

    if template_json_validator is not None:
        checks.append(("template_json", template_json_validator.validate(state)))

    if numerical_hallucination_validator is not None:
        checks.append(
            ("numerical_hallucination", numerical_hallucination_validator.validate(state))
        )

    state.consistency_passed = checks[0][1].passed
    state.llm_output_passed = checks[1][1].passed

    state.template_json_passed = next(
        (r.passed for name, r in checks if name == "template_json"), None
    )

    state.numerical_hallucination_passed = next(
        (r.passed for name, r in checks if name == "numerical_hallucination"), None
    )

    state.validation_details = {
        name: {
            "passed": result.passed,
            "message": result.message,
            "details": result.details,
        }
        for name, result in checks
    }

    all_passed = all(result.passed for _, result in checks)

    state.validation_passed = all_passed
    state.report_delivery_allowed = all_passed

    if not all_passed:
        for name, result in checks:
            if not result.passed:
                state.errors.append(f"Report validation failed — {name}: {result.message}")

        state.report_delivery_allowed = False

        if failure_policy == "flag":
            state.flagged_for_review = True
            state.review_reasons.append("Report failed pre-save validation")

        if failure_policy == "strip":
            state.review_reasons.append(
                "Report failed validation and should not be delivered automatically"
            )

    return state


class LeukemiaPipeline:
    """Small convenience wrapper for direct full-flow execution."""

    def __init__(
        self,
        detector: BaseDetectionAgent,
        classifier: HybridClassifier,
        report_generator: BaseReportGenerator,
        reflection_agent: ReflectionAgent | None = None,
    ):
        self.detector = detector
        self.classifier = classifier
        self.report_generator = report_generator
        self.reflection_agent = reflection_agent

    def run(self, case_id: str, image_paths: list[str], instruction: str | None = None) -> PipelineState:
        state = PipelineState(case_id=case_id, image_paths=image_paths, text_input=instruction)
        state = detect_node(state, self.detector)
        state = aggregate_node(state)
        state = classify_node(state, self.classifier)
        state = reflect_node(
            state, agent=self.reflection_agent, classifier=self.classifier
        )
        state = report_node(state, self.report_generator)
        return state