python - <<'EOF'
import csv, json
from pathlib import Path
R = Path('/home/roba.majzoub/research/AgenticHematology/outputs')
E = R / 'cv_eval_v2'
summary = json.load(open(E / 'summary.json'))
flag_only = json.load(open(E / 'ablation_flag_only.json'))
rows = [r for r in csv.DictReader(open(E / 'per_patient.csv'))]
b = lambda v: v == 'True'

# ---- 6-row ablation table (uniform schema) ----
def fmt(a):
    return {"accuracy_all": a["accuracy_all"], "coverage": a["coverage"],
            "selective_accuracy": a["selective_accuracy"],
            "errors_caught": a["errors_caught"] if isinstance(a["errors_caught"], str)
                             else f'{a["errors_caught"]}/{a["n_errors"]}',
            "conf_baseline_same_coverage": a["conf_baseline_same_coverage"]}
abl = summary["ablation_reflector_guard"]
table = {
    "none__no_guard":      fmt(abl["no_reflector__no_guard"]),
    "none__guard":         fmt(abl["no_reflector__guard"]),
    "flag_only__no_guard": fmt(flag_only["flag_only__no_guard"]),
    "flag_only__guard":    fmt(flag_only["flag_only__guard"]),
    "full__no_guard":      fmt(abl["reflector__no_guard"]),
    "full__guard":         fmt(abl["reflector__guard"]),
}

# ---- outcome categories over the 46 evaluable patients ----
def outcome(r):
    flagged, correct, pre_ok = b(r["flagged"]), b(r["correct"]), b(r["correct_noreflect"])
    if not flagged:
        if correct: return "proceeded_correct"
        return "error_introduced_unflagged" if pre_ok else "error_missed"
    if correct: return "flagged_correct_conservative"
    return "error_introduced_then_flagged" if pre_ok else "error_caught"
ev = [r for r in rows if r["patient_id"] != "24"]
outcomes = {}
for r in ev:
    outcomes.setdefault(outcome(r), []).append(r["patient_id"])
consistency_restored = [r["patient_id"] for r in ev if b(r["re_aggregated"])
                        and b(r.get("lineage_conflict_noreflect")) and not b(r.get("lineage_conflict"))]

# ---- per-patient traces for every re-aggregated or flagged patient ----
folds = json.load(open('/home/roba.majzoub/research/AgenticHematology/agentic_hematology/wbc_unified/config/cv_folds.json'))["folds"]
fold_of = {p: f for f, ps in enumerate(folds) for p in ps}
cases = {}
for r in rows:
    if b(r["re_aggregated"]) or b(r["flagged"]):
        p = r["patient_id"]
        tr = json.load(open(R / 'v2_reflect' / f'cv_fold{fold_of[p]}' / p / f'case_{p}_agent_trace.json'))
        cases[p] = {"true": r["true"], "pred_noreflect": r["pred_noreflect"], "conf_noreflect": float(r["conf_noreflect"]),
                    "pred": r["pred"], "confidence": float(r["confidence"]),
                    "cohort_noreflect": r.get("cohort_type_noreflect"), "cohort": r.get("cohort_type"),
                    "flagged": b(r["flagged"]), "outcome": outcome(r) if p != "24" else "excluded_disputed_reference",
                    "agent_actions": tr.get("agent_actions", []), "review_reasons": tr.get("review_reasons", [])}

out = {
    "description": "Reflection agent: ablation (reflector x lineage guard), outcome counts, per-case traces. "
                   "flag_only applies the full reflector's flags to pre-reflection predictions (post hoc). "
                   "Patient 24 excluded from metrics (disputed reference).",
    "n_evaluable": len(ev),
    "ablation_table": table,
    "outcome_counts": {k: len(v) for k, v in sorted(outcomes.items())},
    "outcome_patients": dict(sorted(outcomes.items())),
    "lineage_consistency_restored_by_reaggregation": consistency_restored,
    "changed_by_reflection": summary["classification"]["changed_by_reflection"],
    "re_aggregated_patients": summary["agent"]["re_aggregated_patients"],
    "flagged_patients": summary["agent"]["flagged_patients"],
    "n_llm_errors": summary["agent"]["n_llm_errors"],
    "cases": cases,
}
(E / 'reflection_agent_results.json').write_text(json.dumps(out, indent=2))
print(json.dumps({k: out[k] for k in ("outcome_counts", "outcome_patients", "lineage_consistency_restored_by_reaggregation")}, indent=2))
print("Saved", E / 'reflection_agent_results.json')
EOF