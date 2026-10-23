"""
agent_controller.py
====================
The components that make the pipeline *agentic* rather than merely automated:
a model (Qwen3) makes control-flow decisions at runtime, based on the
intermediate results of the deterministic stages.

Two pieces:

1. `QwenLLMClient` — a thin, reusable wrapper around the same local Qwen3
   model used for report generation. Exposes a generic
   `complete(system, user) -> str` so the SAME model instance can serve:
     - intent routing (LLMRouter)
     - the reflection agent (below)
     - EXPLAIN answers
     - report generation (shared into LocalLLMReportGenerator)
   Loading one model and sharing the handle avoids a second GPU copy.

2. `ReflectionAgent` — the agentic controller. After the deterministic
   detect → aggregate → classify stages run, this agent reads a compact
   "case state" and DECIDES the next PROCESS action:
     - PROCEED          : evidence is coherent and sufficient → write report
     - RE_AGGREGATE     : re-run aggregation at an adjusted confidence
                          threshold (noisy/low-confidence detections are
                          muddying the differential) → then re-classify
                          and reflect again
     - FLAG_FOR_REVIEW  : borderline / contradictory / sparse → write report
                          but mark it for mandatory human review with the
                          agent's stated concern

Safety boundary: the reflection agent does NOT make or change the diagnosis.
The deterministic `HybridClassifier` remains authoritative. The agent only
decides process actions (look again / proceed / escalate). Its output schema
has no diagnosis field, so it is structurally incapable of overriding the
classifier. On any parse failure it fails safe to FLAG_FOR_REVIEW.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class LLMClient(Protocol):
    """Anything with this shape can drive the agentic components."""
    def complete(self, system: str, user: str) -> str: ...


class QwenLLMClient:
    """
    Reusable wrapper around the local Qwen3 model (the same one used for
    report generation). Lazy-loads on first use.

    Pass the same instance to the router, the reflection agent, the
    orchestrator's `llm_explain`, and (optionally) the report generator so
    only one copy of the weights is resident.
    """

    def __init__(
        self,
        model_path: str | None = None,
        adapter_path: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None

    # -- loading ---------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        # Reuse the project's own loader so behaviour matches report gen.
        from report.src.llm_infer import load_model_and_tokenizer  # type: ignore

        print("Loading Qwen agent model...", flush=True)
        self._model, self._tokenizer = load_model_and_tokenizer(
            self.model_path, self.adapter_path
        )
        print("Qwen agent model loaded.", flush=True)

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    def attach(self, model, tokenizer) -> "QwenLLMClient":
        """Inject an already-loaded model/tokenizer (to share one copy)."""
        self._model = model
        self._tokenizer = tokenizer
        return self

    # -- generic completion ---------------------------------------------

    def complete(self, system: str, user: str) -> str:
        from report.src.llm_infer import generate_from_messages  # type: ignore

        self._ensure_loaded()
        print("Running Qwen agent generation...", flush=True)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        out = generate_from_messages(
            self._model,
            self._tokenizer,
            messages,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        print("Qwen agent generation complete.", flush=True)
        return out


# ---------------------------------------------------------------------------
# Agent decision schema
# ---------------------------------------------------------------------------

class AgentAction(str, Enum):
    PROCEED = "proceed"
    RE_AGGREGATE = "re_aggregate"
    FLAG_FOR_REVIEW = "flag_for_review"


@dataclass
class AgentDecision:
    action: AgentAction
    reason: str
    conf_threshold: float | None = None   # only meaningful for RE_AGGREGATE
    raw: str = ""                          # raw LLM text, for audit
    error: bool = False 

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action.value, 
                "reason": self.reason,
                "conf_threshold": self.conf_threshold, 
                "error": self.error
                }


# ---------------------------------------------------------------------------
# Reflection agent
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a PROCESS-CONTROL agent inside a hematology diagnosis pipeline. "
    "A separate deterministic classifier already owns the diagnosis; you must "
    "NOT make, change, or second-guess the diagnosis itself. Your only job is "
    "to decide the next PROCESS action based on data quality and internal "
    "coherence of the evidence.\n\n"
    "Choose EXACTLY ONE action and reply with ONLY a JSON object, no prose:\n"
    '{"action": <one of "proceed", "re_aggregate", "flag_for_review">, '
    '"reason": "<one short sentence>", '
    '"conf_threshold": <float 0.1-0.9, only if action is re_aggregate>}\n\n'
    "Guidance:\n"
    "- re_aggregate: choose this when detection quality looks noisy — e.g. a "
    "high fraction of non-WBC/None detections, low mean detection confidence, "
    "and the differential could plausibly sharpen if low-confidence detections "
    "are dropped. Provide a stricter conf_threshold than the current one. Only "
    "available while re-aggregation budget remains: if "
    "re_aggregate_already_used is true in the input, do not choose this "
    "action.\n"
    "- flag_for_review: choose this when the picture is borderline or "
    "contradictory — e.g. blast burden near the 20% threshold, very few "
    "informative cells, low classifier confidence, or morphology that conflicts "
    "with the predicted class.\n"
    "- proceed: choose this when the evidence is coherent and sufficient.\n"
    "Examples:\n\n"
    "case_state: {\"blast_pct\": 6.5, \"qc\": {\"mean_det_conf\": 0.48, \"pct_class_none\": 34.0}, "
    "\"n_cells_informative\": 65}\n"
    "→ {\"action\": \"re_aggregate\", \"reason\": \"high artifact fraction and low mean confidence "
    "suggest differential may sharpen with a stricter threshold\", \"conf_threshold\": 0.4}\n\n"
    "case_state: {\"blast_pct\": 22.3, \"qc\": {\"mean_det_conf\": 0.42, \"pct_class_none\": 31.0}, "
    "\"n_cells_informative\": 18}\n"
    "→ {\"action\": \"flag_for_review\", \"reason\": \"blast burden near 20% threshold with few "
    "informative cells that re-aggregation cannot increase\"}\n"

)


class ReflectionAgent:
    """
    LLM-driven process controller. Reads the case state, returns an
    `AgentDecision`. Fails safe to FLAG_FOR_REVIEW on any error.
    """

    def __init__(
        self,
        llm: LLMClient,
        min_conf_threshold: float = 0.25,
        max_conf_threshold: float = 0.9,
    ):
        self.llm = llm
        self.min_conf = min_conf_threshold
        self.max_conf = max_conf_threshold

    def decide(
        self,
        case_state: dict[str, Any],
        *,
        re_aggregate_used: bool,
        current_conf_threshold: float,
        iteration: int,
    ) -> AgentDecision:
        user = json.dumps(
            {
                "case_state": case_state,
                "current_conf_threshold": current_conf_threshold,
                "re_aggregate_already_used": re_aggregate_used,
                "iteration": iteration,
            },
            indent=2,
        )

        try:
            print("Running reflection agent...", flush=True)
            raw = self.llm.complete(_SYSTEM_PROMPT, user)

        except Exception as e:
            return AgentDecision(
                AgentAction.FLAG_FOR_REVIEW,
                f"reflection LLM call failed ({e}); flagging for safety",
                raw="",
                error=True
            )
        print("Reflection agent complete.", flush=True)


        decision = self._parse(raw)

        # ---- enforce the safety / sanity envelope ----
        # Disallow re_aggregate if already used, and clamp the threshold.
        if decision.action == AgentAction.RE_AGGREGATE:
            print("REAGREGGATION LOOOP)")
            if re_aggregate_used:
                return AgentDecision(
                    AgentAction.FLAG_FOR_REVIEW,
                    "re_aggregation budget exhausted; escalating instead",
                    raw=raw,
                )
            ct = decision.conf_threshold
            if ct is None or not (self.min_conf <= ct <= self.max_conf):
                # Pick a sensible stricter default if the model gave none/garbage.
                ct = min(self.max_conf, max(self.min_conf, current_conf_threshold + 0.15))
            # A re_aggregate that doesn't tighten the threshold is pointless.
            if ct <= current_conf_threshold:
                ct = min(self.max_conf, current_conf_threshold + 0.1)
            decision.conf_threshold = round(ct, 3)
        return decision

    # ------------------------------------------------------------------

    def _parse(self, raw: str) -> AgentDecision:
        """Extract the first JSON object; fail safe to FLAG_FOR_REVIEW."""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return AgentDecision(
                AgentAction.FLAG_FOR_REVIEW,
                "could not parse agent JSON; flagging for safety",
                raw=raw,
            )
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return AgentDecision(
                AgentAction.FLAG_FOR_REVIEW,
                "invalid agent JSON; flagging for safety",
                raw=raw,
                error=True
            )

        action_str = str(obj.get("action", "")).strip().lower()
        try:
            action = AgentAction(action_str)
        except ValueError:
            return AgentDecision(
                AgentAction.FLAG_FOR_REVIEW,
                f"unrecognised action {action_str!r}; flagging for safety",
                raw=raw,
            )

        reason = str(obj.get("reason", "")).strip() or "(no reason given)"
        conf = obj.get("conf_threshold")
        conf_threshold = float(conf) if isinstance(conf, (int, float)) else None
        return AgentDecision(action, reason, conf_threshold, raw=raw)


# ---------------------------------------------------------------------------
# Compact case-state builder — what the agent actually sees
# ---------------------------------------------------------------------------

def build_case_state(findings, classification, dataset_source: str = "lld") -> dict[str, Any]:
    """
    Build the compact, decision-relevant view of the case for the agent.
    Deliberately small: differential, blast burden, QC quality signals, and
    the classifier's output (so the agent can judge coherence, not re-derive
    the diagnosis).
    """
    rr = findings.report_ready
    qc = rr.get("qc", {})
    state: dict[str, Any] = {
        "dataset_source": dataset_source,
        "differential_pct": findings.cell_percentages_clinical,
        "blast_pct": rr.get("blast_pct"),
        "flags": rr.get("flags", {}),
        "n_cells_informative": rr.get("n_cells_informative", findings.n_cells_identified_wbc),
        "n_cells_artifact": rr.get("n_cells_artifact"),
        "qc": {
            "mean_det_conf": qc.get("mean_det_conf"),
            "pct_class_none": qc.get("pct_class_none"),
        },
    }
    if classification is not None:
        state["classifier_output"] = {
            "predicted_class": classification.predicted_class,
            "confidence": classification.confidence,
            "rationale": classification.rationale,
        }
    return state
