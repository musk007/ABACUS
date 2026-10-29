import csv
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from agentic_hematology.orchestrator import (
    Intent,
    OrchestratorRequest,
    LLMRouter,
    RuleBasedRouter
)
from agentic_hematology.agent_controller import QwenLLMClient

import gc
import torch

MODEL = "/home/roba.majzoub/research/AgenticHematology/report_llm/pretrained_models/Qwen3-VL-4B-Instruct"
REPORT_LORA = ("/home/roba.majzoub/research/AgenticHematology/report_llm/fold0/runs/verl/"
               "sft_qwen3_4b_instruct_lora/global_step_24/lora_adapter_lm")


def make_llm_router(adapter_path, keep_adapter_active: bool):
    client = QwenLLMClient(model_path=MODEL, adapter_path=adapter_path, max_new_tokens=256, temperature=0.0)
    if not keep_adapter_active:
        return LLMRouter(llm_complete=client.complete, fallback=RuleBasedRouter()), client

    def lora_complete(system: str, user: str) -> str:          # bypasses the adapter-disabling in .complete()
        from report.src.llm_infer import generate_from_messages
        client._ensure_loaded()
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return generate_from_messages(client._model, client._tokenizer, msgs, max_new_tokens=256, temperature=0.0)

    return LLMRouter(llm_complete=lora_complete, fallback=RuleBasedRouter()), client

import re as _re
from agentic_hematology.orchestrator import tools_for_intent


def make_intent_canonical_router():
    router, client = make_llm_router(None, keep_adapter_active=False)
    llm = router.llm_complete

    def complete(system: str, user: str) -> str:
        raw = llm(system, user)
        try:
            payload = json.loads(_re.search(r"\{.*\}", raw, _re.DOTALL).group(0))
            payload["tool_sequence"] = tools_for_intent(Intent(payload["intent"]))
            return json.dumps(payload)
        except Exception:
            return raw          # unparseable output is rejected exactly as before
    router.llm_complete = complete
    return router, client

ROUTERS = {
    "rule_based":      lambda: (RuleBasedRouter(), None),
    "llm_base":        lambda: make_llm_router(None, keep_adapter_active=False),       # deployed setting
    "llm_report_lora": lambda: make_llm_router(REPORT_LORA, keep_adapter_active=True), # ablation
    "llm_intent_canonical": make_intent_canonical_router,
}
INTENT_LABELS = [intent.value for intent in Intent]


def expected_tools(intent: Intent) -> list[str]:
    return {
        Intent.FULL_REPORT: ["detect", "aggregate", "classify", "reflect", "report", "validate"],
        Intent.DETECT_ONLY: ["detect", "aggregate"],
        Intent.CLASSIFY_ONLY: ["detect", "aggregate", "classify"],
        Intent.REPORT_FROM_JSON: ["classify", "reflect", "report", "validate"],
        Intent.EXPLAIN: ["explain"],
    }[intent]
examples = [
    # instruction, ground-truth intent, image_paths, precomputed_findings
    ("Analyze this blood smear", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("Diagnose the patient", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("Produce a diagnostic report", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("Analyze this blood smear", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("Diagnose the patient", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("Produce a diagnostic report", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("Generate a full hematology report", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("Run the complete diagnostic workflow", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("Evaluate this case and write the report", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("Please analyze the images and generate the final diagnosis report", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("Perform the full pipeline on this patient", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("Review these smear images and prepare a diagnostic summary", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("Process this case from images to validated report", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("Generate a full diagnostic report for this patient", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("I need a complete hematology report from these smear images", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("Process this case fully — detection, classification, and reporting", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("Give me the diagnostic workup for this patient's smear", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("Full pipeline please, I want the report at the end", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("Analyze this patient's blood smear and generate a clinical report", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("Run everything from cell detection to final impression", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("I have new smear tiles, generate the complete diagnostic report", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("Do a full leukemia workup on this case", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("Take these images and give me a signed-off style report", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("Complete analysis needed — I want the end-to-end result", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("Start to finish, process this patient's images into a report", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("Can you produce the whole diagnostic report from these tiles?", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("New patient case — run detection, classification, and reporting", Intent.FULL_REPORT, ["dummy_image.png"], None),
    ("I want a full report, not just the classification", Intent.FULL_REPORT, ["patient_017.png"], None),
    ("Please generate the complete case report from these images", Intent.FULL_REPORT, ["patient_018.png"], None),
    ("Run the full agentic pipeline on this smear", Intent.FULL_REPORT, ["patient_019.png"], None),
    ("Give me everything — counts, classification, and the report", Intent.FULL_REPORT, ["patient_020.png"], None),

    # DETECT_ONLY
    ("Locate all white blood cells", Intent.DETECT_ONLY, ["dummy_image.png"], None),
    ("Find all leukocytes", Intent.DETECT_ONLY, ["dummy_image.png"], None),
    ("Count the WBCs", Intent.DETECT_ONLY, ["dummy_image.png"], None),
    ("Detect cells only", Intent.DETECT_ONLY, ["dummy_image.png"], None),
    ("Only identify the white blood cells in the images", Intent.DETECT_ONLY, ["dummy_image.png"], None),
    ("Find and count the cells without diagnosis", Intent.DETECT_ONLY, ["dummy_image.png"], None),
    ("Run cell detection but do not classify leukemia", Intent.DETECT_ONLY, ["dummy_image.png"], None),
    ("Return the detected WBC locations", Intent.DETECT_ONLY, ["dummy_image.png"], None),
    ("Extract the cell bounding boxes only", Intent.DETECT_ONLY, ["dummy_image.png"], None),
    ("Just localize the leukocytes in this smear", Intent.DETECT_ONLY, ["dummy_image.png"], None),
    ("Just detect the cells in these images", Intent.DETECT_ONLY, ["patient_021.png"], None),
    ("Find all the WBCs in this smear, nothing else", Intent.DETECT_ONLY, ["patient_022.png"], None),
    ("Run cell detection only on these tiles", Intent.DETECT_ONLY, ["patient_023.png"], None),
    ("I only need bounding boxes, skip classification", Intent.DETECT_ONLY, ["patient_024.png"], None),
    ("Locate the cells in this patient's images", Intent.DETECT_ONLY, ["patient_025.png"], None),
    ("Can you just run detection on these smear tiles?", Intent.DETECT_ONLY, ["patient_026.png"], None),
    ("Give me the cell locations, no need for a report", Intent.DETECT_ONLY, ["patient_027.png"], None),
    ("Detect cells only, I'll handle classification myself", Intent.DETECT_ONLY, ["patient_028.png"], None),
    ("I want raw detections, not a diagnosis", Intent.DETECT_ONLY, ["patient_029.png"], None),
    ("Just find where the cells are in this image", Intent.DETECT_ONLY, ["patient_030.png"], None),
    ("Run the detector on these tiles and stop there", Intent.DETECT_ONLY, ["patient_031.png"], None),
    ("No classification needed, just detect the objects", Intent.DETECT_ONLY, ["patient_032.png"], None),
    ("Please identify cell locations only in this smear", Intent.DETECT_ONLY, ["patient_033.png"], None),
    ("I need detection output for downstream processing elsewhere", Intent.DETECT_ONLY, ["patient_034.png"], None),
    ("Skip everything after detection", Intent.DETECT_ONLY, ["patient_035.png"], None),
    ("Only run the detection stage on this case", Intent.DETECT_ONLY, ["patient_036.png"], None),
    ("Give me bounding boxes for all cells in these tiles", Intent.DETECT_ONLY, ["patient_037.png"], None),
    ("Just want to see what cells were found, no diagnosis", Intent.DETECT_ONLY, ["patient_038.png"], None),
    ("Detect objects in this smear, don't classify them", Intent.DETECT_ONLY, ["patient_039.png"], None),
    ("Run detection only — I'm just checking image quality", Intent.DETECT_ONLY, ["patient_040.png"], None),
    ("Locate all white blood cells", Intent.DETECT_ONLY, ["dummy_image.png"], None),
    ("Find all leukocytes", Intent.DETECT_ONLY, ["dummy_image.png"], None),
    ("Count the WBCs", Intent.DETECT_ONLY, ["dummy_image.png"], None),

    # CLASSIFY_ONLY
    ("What leukemia does this patient have?", Intent.CLASSIFY_ONLY, ["dummy_image.png"], None),
    ("Predict the diagnosis", Intent.CLASSIFY_ONLY, ["dummy_image.png"], None),
    ("Determine the leukemia subtype", Intent.CLASSIFY_ONLY, ["dummy_image.png"], None),
    ("Classify this case", Intent.CLASSIFY_ONLY, ["dummy_image.png"], None),
    ("Only predict the leukemia class", Intent.CLASSIFY_ONLY, ["dummy_image.png"], None),
    ("Tell me whether this is ALL, AML, APML, CLL, or CML", Intent.CLASSIFY_ONLY, ["dummy_image.png"], None),
    ("Run classification without generating a report", Intent.CLASSIFY_ONLY, ["dummy_image.png"], None),
    ("Identify the leukemia category for this patient", Intent.CLASSIFY_ONLY, ["dummy_image.png"], None),
    ("Give me the predicted leukemia label only", Intent.CLASSIFY_ONLY, ["dummy_image.png"], None),
    ("Classify the patient-level diagnosis from the smear images", Intent.CLASSIFY_ONLY, ["dummy_image.png"], None),
    ("Classify this patient's leukemia subtype", Intent.CLASSIFY_ONLY, ["patient_041.png"], None),
    ("Tell me what type of leukemia this is, no report needed", Intent.CLASSIFY_ONLY, ["patient_042.png"], None),
    ("Run detection and classification, but skip the report", Intent.CLASSIFY_ONLY, ["patient_043.png"], None),
    ("What subtype does this patient have?", Intent.CLASSIFY_ONLY, ["patient_044.png"], None),
    ("I just need the diagnosis label, not the full write-up", Intent.CLASSIFY_ONLY, ["patient_045.png"], None),
    ("Classify the cells and give me the patient-level label", Intent.CLASSIFY_ONLY, ["patient_046.png"], None),
    ("Determine ALL vs AML vs CML for this case, no report", Intent.CLASSIFY_ONLY, ["patient_047.png"], None),
    ("Run the classifier on this smear and tell me the result", Intent.CLASSIFY_ONLY, ["patient_048.png"], None),
    ("I need the attribute classification and final class only", Intent.CLASSIFY_ONLY, ["patient_049.png"], None),
    ("Just tell me the leukemia class for this patient", Intent.CLASSIFY_ONLY, ["patient_050.png"], None),
    ("Give me the classification output, skip report generation", Intent.CLASSIFY_ONLY, ["patient_051.png"], None),
    ("Process this case up through classification only", Intent.CLASSIFY_ONLY, ["patient_052.png"], None),
    ("What's the predicted subtype based on these tiles?", Intent.CLASSIFY_ONLY, ["patient_053.png"], None),
    ("Classify cells and aggregate, but don't write a report", Intent.CLASSIFY_ONLY, ["patient_054.png"], None),
    ("I want the diagnostic label without the narrative report", Intent.CLASSIFY_ONLY, ["patient_055.png"], None),
    ("Run through classification and stop before reporting", Intent.CLASSIFY_ONLY, ["patient_056.png"], None),
    ("Tell me the dominant cell type and leukemia class", Intent.CLASSIFY_ONLY, ["patient_057.png"], None),
    ("Just classify, I'll write my own report", Intent.CLASSIFY_ONLY, ["patient_058.png"], None),
    ("Give me the classification decision for this smear", Intent.CLASSIFY_ONLY, ["patient_059.png"], None),
    ("Need the subtype prediction only, not the full pipeline", Intent.CLASSIFY_ONLY, ["patient_060.png"], None),
    ("What leukemia does this patient have?", Intent.CLASSIFY_ONLY, ["dummy_image.png"], None),
    ("Predict the diagnosis", Intent.CLASSIFY_ONLY, ["dummy_image.png"], None),
    ("Determine the leukemia subtype", Intent.CLASSIFY_ONLY, ["dummy_image.png"], None),

    # REPORT_FROM_JSON
    ("Create a report from these findings", Intent.REPORT_FROM_JSON, [], {"report_ready": {}}),
    ("Generate a report using the aggregated results", Intent.REPORT_FROM_JSON, [], {"report_ready": {}}),
    ("Summarize this JSON into a report", Intent.REPORT_FROM_JSON, [], {"report_ready": {}}),
    ("Write the diagnostic report from the precomputed findings", Intent.REPORT_FROM_JSON, [], {"report_ready": {}}),
    ("Use the existing aggregation output to generate the report", Intent.REPORT_FROM_JSON, [], {"report_ready": {}}),
    ("Convert these structured findings into a hematology report", Intent.REPORT_FROM_JSON, [], {"report_ready": {}}),
    ("Generate the narrative report without rerunning detection", Intent.REPORT_FROM_JSON, [], {"report_ready": {}}),
    ("Create the final report from the saved patient statistics", Intent.REPORT_FROM_JSON, [], {"report_ready": {}}),
    ("Use this JSON file to produce the report", Intent.REPORT_FROM_JSON, [], {"report_ready": {}}),
    ("Draft a report based only on the supplied aggregated findings", Intent.REPORT_FROM_JSON, [], {"report_ready": {}}),
    ("Write a report from this precomputed JSON", Intent.REPORT_FROM_JSON, None, {"report_ready": {}}),
    ("I already have the findings, just generate the report", Intent.REPORT_FROM_JSON, None, {"report_ready": {}}),
    ("Here's the classification output, produce a report from it", Intent.REPORT_FROM_JSON, None, {"report_ready": {}}),
    ("Turn this JSON into a diagnostic report, no images needed", Intent.REPORT_FROM_JSON, None, {"report_ready": {}}),
    ("I have precomputed cell counts, generate the report from them", Intent.REPORT_FROM_JSON, None, {"report_ready": {}}),
    ("Skip detection, I already ran it — just report from this file", Intent.REPORT_FROM_JSON, None, {"report_ready": {}}),
    ("Use this JSON to write the final report", Intent.REPORT_FROM_JSON, None, {"report_ready": {}}),
    ("These are the saved findings, generate the report text", Intent.REPORT_FROM_JSON, None, {"report_ready": {}}),
    ("No new images, just build the report from existing data", Intent.REPORT_FROM_JSON, None, {"report_ready": {}}),
    ("Report generation only, from this stored JSON", Intent.REPORT_FROM_JSON, None, {"report_ready": {}}),
    ("I have the pipeline output saved, please write it up", Intent.REPORT_FROM_JSON, None, {"report_ready": {}}),
    ("Convert these precomputed results into a readable report", Intent.REPORT_FROM_JSON, None, {"report_ready": {}}),
    ("Just need the narrative report, I already have the numbers", Intent.REPORT_FROM_JSON, None, {"report_ready": {}}),
    ("Generate the report text from this existing findings file", Intent.REPORT_FROM_JSON, None, {"report_ready": {}}),
    ("Reuse this saved analysis and produce a report", Intent.REPORT_FROM_JSON, None, {"report_ready": {}}),
    ("From this JSON, write the diagnostic report", Intent.REPORT_FROM_JSON, None, {"report_ready": {}}),
    ("I re-ran nothing, just format this data into a report", Intent.REPORT_FROM_JSON, None, {"report_ready": {}}),
    ("Given these precomputed attributes, generate the write-up", Intent.REPORT_FROM_JSON, None, {"report_ready": {}}),
    ("Build the report from cached results, no image analysis", Intent.REPORT_FROM_JSON, None, {"report_ready": {}}),
    ("Report only, using this existing findings JSON", Intent.REPORT_FROM_JSON, None, {"report_ready": {}}),
    ("Create a report from these findings", Intent.REPORT_FROM_JSON, [], {"report_ready": {}}),
    ("Generate a report using the aggregated results", Intent.REPORT_FROM_JSON, [], {"report_ready": {}}),
    ("Summarize this JSON into a report", Intent.REPORT_FROM_JSON, [], {"report_ready": {}}),

    # EXPLAIN
    ("Why did the model predict AML?", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Explain the diagnosis", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Why is this considered CML?", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Explain why the model chose this leukemia class", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("What evidence supports the predicted diagnosis?", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Why was this case not classified as ALL?", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Interpret the model's reasoning for this case", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Justify the classification result", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Explain which cell populations influenced the prediction", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("What findings led to this diagnosis?", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Why was this patient classified as AML?", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Explain the reasoning behind this diagnosis", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("What led to the CML impression for this case?", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Justify why the blast threshold was considered met here", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Can you explain why this cohort was flagged for review?", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Why did the system pick lymphoblast as the dominant cell type?", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Explain the differential alert triggered on this patient", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("What evidence supports the APML impression here?", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Walk me through why this case was ambiguous", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Explain why basophilia was flagged as absent in this report", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Why wasn't this classified as ALL instead of AML?", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Give me the rationale behind the recommended workup", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Explain how the cohort morphology led to this conclusion", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("What made the model flag this as a mixed-lineage tie?", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Explain the reasoning for the borderline blast call", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Why is this report inconsistent with a chronic pattern?", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Justify the dominant cell type selection for this case", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Explain why no report was generated automatically here", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("What's the basis for the CLL impression in this patient?", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Explain the QC warning attached to this case", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Why did the model predict AML?", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Explain the diagnosis", Intent.EXPLAIN, ["dummy_image.png"], None),
    ("Why is this considered CML?", Intent.EXPLAIN, ["dummy_image.png"], None),
    ]

invalid_examples = [
    # Image-dependent task without image inputs
    ("Detect all white blood cells.",[],None,),
    ("Analyze this blood smear and diagnose the patient.", [], None,),
    ("Classify the leukemia subtype from these images.", [], None,),
    ("Locate and count all leukocytes in the smear.",[],None,),
    ("Generate a full diagnostic report for this patient",[], None),

    # Classification requested without aggregation
    ("Skip aggregation and classify the leukemia directly.",["dummy_image.png"],None,),
    ("Determine the leukemia subtype immediately after detection.",["dummy_image.png"],None,),
    ("Classify the patient without aggregating the detected cells.",["dummy_image.png"],None,),

    # Report generation requested before classification
    ("Generate the final report before classifying the leukemia.",["dummy_image.png"],None,),
    ("Write a diagnostic report using only the detected cells.",["dummy_image.png"],None,),
    ("Produce the report without running leukemia classification.",["dummy_image.png"],None,),
]
# ---- clean the example set -------------------------------------------------
def _norm(paths):
    return list(paths or [])

seen, deduped = set(), []
for ins, intent, imgs, find in examples:
    key = (ins, intent, bool(imgs), find is not None)
    if key not in seen:
        seen.add(key)
        deduped.append((ins, intent, _norm(imgs), find))
print(f"valid examples: {len(examples)} -> {len(deduped)} after removing duplicates")
examples = deduped
invalid_examples = [(ins, _norm(imgs), find) for ins, imgs, find in invalid_examples]


def input_condition(imgs, find) -> str:
    return {(True, False): "images", (False, True): "findings",
            (True, True): "images+findings", (False, False): "none"}[(bool(imgs), find is not None)]


def plan_is_executable(tools, imgs, find) -> bool:
    if tools == ["explain"]:
        return True
    if "detect" in tools and not imgs:
        return False
    if "detect" not in tools and find is None:
        return False
    if "classify" in tools and "aggregate" not in tools and find is None:
        return False
    if "report" in tools and "classify" not in tools:
        return False
    if "validate" in tools and "report" not in tools:
        return False
    return True


def evaluate(name, router) -> dict:
    rows = []
    for i, (ins, gt, imgs, find) in enumerate(examples):
        d = router.route(OrchestratorRequest(case_id=f"{name}_{i}", image_paths=imgs,
                                             instruction=ins, precomputed_findings=find))
        rows.append({
            "instruction": ins, "input_condition": input_condition(imgs, find),
            "ground_truth_intent": gt.value, "predicted_intent": d.intent.value,
            "intent_correct": d.intent == gt,
            "exact_plan_match": d.intent == gt and d.tool_sequence == expected_tools(gt),
            "predicted_tool_sequence": d.tool_sequence,
            "llm_bypassed": find is not None and not imgs,          # deterministic shortcut in route()
            "llm_plan_rejected": bool(d.precondition_violation_caught),
            "rationale": d.rationale,
        })

    inv = []
    for i, (ins, imgs, find) in enumerate(invalid_examples):
        d = router.route(OrchestratorRequest(case_id=f"{name}_inv_{i}", image_paths=imgs,
                                             instruction=ins, precomputed_findings=find))
        inv.append({"instruction": ins, "input_condition": input_condition(imgs, find),
                    "predicted_intent": d.intent.value, "predicted_tool_sequence": d.tool_sequence,
                    "llm_plan_rejected": bool(d.precondition_violation_caught),
                    "final_plan_executable": plan_is_executable(d.tool_sequence, imgs, find),
                    "rationale": d.rationale})

    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    llm_rows = [r for r in rows if not r["llm_bypassed"]]
    y_true = [r["ground_truth_intent"] for r in rows]
    y_pred = [r["predicted_intent"] for r in rows]
    summary = {
        "router": name, "n_valid": len(rows), "n_valid_llm_decided": len(llm_rows), "n_invalid": len(inv),
        "intent_accuracy": mean([r["intent_correct"] for r in rows]),
        "exact_plan_accuracy": mean([r["exact_plan_match"] for r in rows]),
        "intent_accuracy_excl_bypass": mean([r["intent_correct"] for r in llm_rows]),
        "exact_plan_accuracy_excl_bypass": mean([r["exact_plan_match"] for r in llm_rows]),
        "llm_plan_rejection_rate_valid": mean([r["llm_plan_rejected"] for r in llm_rows]),
        "invalid_final_plan_executable": mean([r["final_plan_executable"] for r in inv]),
        "invalid_llm_plan_rejected": mean([r["llm_plan_rejected"] for r in inv]),
        "per_input_condition": {c: {"n": len(rs), "intent_acc": mean([r["intent_correct"] for r in rs]),
                                    "plan_acc": mean([r["exact_plan_match"] for r in rs])}
                                for c in sorted({r["input_condition"] for r in rows})
                                for rs in [[r for r in rows if r["input_condition"] == c]]},
        "labels": INTENT_LABELS,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=INTENT_LABELS).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=INTENT_LABELS,
                                                       zero_division=0, output_dict=True),
    }
    print(f"\n== {name}")
    print(classification_report(y_true, y_pred, labels=INTENT_LABELS, zero_division=0))
    print({k: v for k, v in summary.items() if not isinstance(v, (dict, list))})
    return summary, rows, inv


out_dir = Path("outputs/router_eval")
out_dir.mkdir(parents=True, exist_ok=True)
all_summaries = {}
for name, factory in ROUTERS.items():
    router, client = factory()
    summary, rows, inv = evaluate(name, router)
    all_summaries[name] = summary
    (out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / f"{name}_valid.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / f"{name}_invalid.json").write_text(json.dumps(inv, indent=2), encoding="utf-8")
    del router, client
    gc.collect()
    torch.cuda.empty_cache()

pct = lambda x: "—" if x is None else f"{100 * x:.1f}"
lines = ["| Router | Intent acc. | Plan acc. | Plan acc. (LLM-decided) | LLM plan rejected | Invalid → executable plan |",
         "|---|---|---|---|---|---|"]
for n, s in all_summaries.items():
    lines.append(f"| {n} | {pct(s['intent_accuracy'])} | {pct(s['exact_plan_accuracy'])} | "
                 f"{pct(s['exact_plan_accuracy_excl_bypass'])} | {pct(s['llm_plan_rejection_rate_valid'])} | "
                 f"{pct(s['invalid_final_plan_executable'])} |")
(out_dir / "router_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))