#!/usr/bin/env python3
"""Rebuild a per-patient table from raw orchestrator case outputs.

Emits the columns ci_tables.py consumes, derived only from the files each run
writes, so it can be pointed at any run (v2_reflect, v2_zeroshot, v3_c1c2...).

    python build_per_patient.py \
        --runs-root outputs/v2_reflect \
        --noreflect-root outputs/v2_noreflect \
        --folds-json agentic_hematology/wbc_unified/config/cv_folds.json \
        --out per_patient_derived.csv

Reads, per patient:
    case_<pid>_classification.json              final prediction + confidence
    case_<pid>_pre_reflection_intermediates.json  prediction before reflection
    case_<pid>_agent_trace.json                 flags, actions, errors
    case_<pid>_report.md                        cohort cell type (lineage guard)

The lineage guard follows orchestrator.lineage_conflict: it fires only when the
predicted subtype is one of the five known classes and the report carries a
cohort line, so an indeterminate prediction is never counted as a conflict.
Without --noreflect-root the guard cannot be evaluated on the pre-reflection
cohort for re-aggregated patients; those are reported and left unflagged.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

MYELOID = {"AML", "APML", "CML"}
LYMPHOID = {"ALL", "CLL"}
COHORT_RE = re.compile(r"\*\*Cohort morphology \(n = \d+ ([^)]+?)\):\*\*")

FIELDS = [
    "fold", "patient_id", "true", "pred", "confidence", "correct",
    "pred_noreflect", "conf_noreflect", "correct_noreflect",
    "flagged", "re_aggregated", "n_reflect", "agent_error",
    "cohort_type", "cohort_type_noreflect",
    "lineage_conflict", "lineage_conflict_noreflect",
    "validation_passed", "enriched", "missing",
]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def find_case_dir(root: Path, pid: str) -> Path | None:
    """Accepts cv_fold*/<pid>/, <pid>/, or files sitting directly in the root."""
    hits = sorted(root.glob(f"**/case_{pid}_classification.json"))
    return hits[0].parent if hits else None


def cohort_of(case_dir: Path | None, pid: str) -> str | None:
    if case_dir is None:
        return None
    for name in (f"case_{pid}_report.md", f"case_{pid}_report_FAILED_VALIDATION.md"):
        p = case_dir / name
        if p.is_file():
            m = COHORT_RE.search(p.read_text(encoding="utf-8"))
            if m:
                return m.group(1).strip().lower()
    return None


def conflicts(pred: str | None, cohort: str | None) -> bool | None:
    """None when the check does not apply, matching the deployed guard."""
    if not pred or not cohort:
        return None
    if pred not in MYELOID | LYMPHOID:
        return None
    return ("lympho" in cohort) == (pred in MYELOID)


def enriched_of(case_dir: Path, pid: str) -> bool | None:
    p = case_dir / f"case_{pid}_report.md"
    if not p.is_file():
        return None
    return "**Morphologic interpretation:**" in p.read_text(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--noreflect-root", type=Path, default=None,
                    help="--no-reflect reruns, needed for the guard on re-aggregated patients")
    ap.add_argument("--folds-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cv = json.loads(args.folds_json.read_text(encoding="utf-8"))
    truth = cv["diagnosis"]

    rows, missing, guard_gaps = [], [], []
    for fold, pids in enumerate(cv["folds"]):
        for pid in pids:
            pid = str(pid)
            case_dir = find_case_dir(args.runs_root, pid)
            if case_dir is None:
                missing.append(pid)
                rows.append({**{k: "" for k in FIELDS}, "fold": fold,
                             "patient_id": pid, "true": truth.get(pid), "missing": True})
                continue

            clf = load(case_dir / f"case_{pid}_classification.json") or {}
            pre = load(case_dir / f"case_{pid}_pre_reflection_intermediates.json") or {}
            trace = load(case_dir / f"case_{pid}_agent_trace.json") or {}
            val = load(case_dir / f"case_{pid}_validation.json") or {}

            pred = clf.get("predicted_class")
            pre_clf = pre.get("leukemia_classification") or {}
            pred_pre = pre_clf.get("predicted_class")
            actions = trace.get("agent_actions") or []
            re_agg = any(a.get("action") == "re_aggregate" for a in actions)

            cohort = cohort_of(case_dir, pid)
            if re_agg and args.noreflect_root:
                cohort_pre = cohort_of(find_case_dir(args.noreflect_root, pid), pid)
            elif re_agg:
                cohort_pre = None
                guard_gaps.append(pid)
            else:
                cohort_pre = cohort          # identical when nothing was re-aggregated

            lc, lc_pre = conflicts(pred, cohort), conflicts(pred_pre, cohort_pre)
            rows.append({
                "fold": fold, "patient_id": pid, "true": truth.get(pid),
                "pred": pred, "confidence": clf.get("confidence"),
                "correct": pred == truth.get(pid),
                "pred_noreflect": pred_pre, "conf_noreflect": pre_clf.get("confidence"),
                "correct_noreflect": pred_pre == truth.get(pid),
                "flagged": bool(trace.get("flagged_for_review", False)),
                "re_aggregated": re_agg,
                "n_reflect": trace.get("n_reflect_iterations", len(actions)),
                "agent_error": any(a.get("error") for a in actions),
                "cohort_type": cohort, "cohort_type_noreflect": cohort_pre,
                "lineage_conflict": "" if lc is None else lc,
                "lineage_conflict_noreflect": "" if lc_pre is None else lc_pre,
                "validation_passed": val.get("validation_passed"),
                "enriched": enriched_of(case_dir, pid),
                "missing": False,
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    done = [r for r in rows if not r["missing"]]
    print(f"{len(done)}/{len(rows)} patients from {args.runs_root}")
    print(f"  re-aggregated : {sum(bool(r['re_aggregated']) for r in done)}")
    print(f"  flagged       : {sum(bool(r['flagged']) for r in done)}")
    print(f"  changed pred  : {[r['patient_id'] for r in done if r['pred'] != r['pred_noreflect']]}")
    if missing:
        print(f"  MISSING       : {missing}")
    if guard_gaps:
        print(f"  guard not evaluable pre-reflection (pass --noreflect-root): {guard_gaps}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()