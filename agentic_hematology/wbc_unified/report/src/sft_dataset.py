"""Build SFT JSONL: detection summary -> GT report markdown."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from data_preprocessing.report_protocol import build_llm_messages, render_template

MORPH_MARKER = "**Morphologic interpretation:**"
_SPARSE_WARNING = "SPARSE ANNOTATION SKEW"


def patient_split_from_folds(folds_json: str, fold: int) -> tuple[set[str], set[str]]:
    """Test = the held-out fold; train = the other four."""
    import json
    folds = json.loads(open(folds_json).read())["folds"]
    test = set(folds[fold])
    train = {p for k, f in enumerate(folds) if k != fold for p in f}
    return train, test

def strip_label(case: dict[str, Any]) -> dict[str, Any]:
    """Remove the filename diagnosis and the QC warning derived from it (absent at deployment)."""
    case = copy.deepcopy(case)
    case.pop("metadata_filename_diagnosis", None)
    rr = case.get("report_ready") or {}
    rr.pop("metadata_filename_diagnosis", None)
    if "qc" in rr:
        rr["qc"]["sparse_annotation_skew_warning"] = False
    return case


def split_gt_report(report_md: str) -> tuple[str, str | None]:
    """Invert LLM_report_generator.insert_section -> (template_md, paragraph or None)."""
    idx = report_md.find(MORPH_MARKER)
    if idx == -1:
        return report_md, None
    start = report_md.rfind("\n", 0, idx) + 1
    for anchor in ("**Diagnostic flags:**", "**Impression:**"):
        a = report_md.find(anchor, idx)
        if a != -1:
            end = report_md.rfind("\n", 0, a) + 1
            return report_md[:start] + report_md[end:], report_md[start:end].strip()
    # section was appended at the end
    return report_md[:start].rstrip() + "\n", report_md[start:].strip()


def template_matches_gt(template_md: str, gt_template_md: str) -> bool:
    """Parity check: regenerated (label-free) template vs. GT template minus the label-derived warning."""
    gt = gt_template_md.replace(f", {_SPARSE_WARNING}", "").replace(
        f"  **WARNINGS:** {_SPARSE_WARNING}.", ""
    )
    return template_md == gt


def has_cohort(case: dict[str, Any]) -> bool:
    """Same skip rule as LLM_report_generator: no cohort -> no paragraph."""
    rr = case.get("report_ready") or {}
    return bool(rr.get("qc", {}).get("n_cells_in_cohort", 0)) and "blast_morphology" in rr

def build_sft_jsonl(
    stats_json: Path,
    reports_gt_dir: Path,
    out_path: Path,
    exclude: set[str] | None = None,
    system_prompt: str | None = None,
) -> int:
    """GT stats JSON + GT reports -> JSONL of (label-free JSON + template) -> morphology paragraph."""
    exclude = exclude or set()
    cases = json.loads(Path(stats_json).read_text(encoding="utf-8"))
    skipped: dict[str, list[str]] = {
        "excluded": [], "no_report": [], "no_paragraph": [], "template_mismatch": []
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for pid, raw_case in cases.items():
            pid = str(pid)
            if pid in exclude:
                skipped["excluded"].append(pid)
                continue
            gt_path = reports_gt_dir / f"case_{pid}_report.md"
            if not gt_path.is_file():
                skipped["no_report"].append(pid)
                continue

            gt_template, paragraph = split_gt_report(gt_path.read_text(encoding="utf-8"))
            if not paragraph or not has_cohort(raw_case):
                # GT skipped the case or its number guard dropped the paragraph: no target
                skipped["no_paragraph"].append(pid)
                continue

            case = strip_label(raw_case)
            template_md = render_template(pid, case)
            if not template_matches_gt(template_md, gt_template):
                skipped["template_mismatch"].append(pid)  # recorded, not skipped

            record = {
                "patient_id": pid,
                "messages": build_llm_messages(case, template_md) + [{"role": "assistant", "content": paragraph}],
                "case_json": json.dumps(case, ensure_ascii=False),
                "template_md": template_md,
                "cohort_cell_type": case["report_ready"]["qc"].get("cohort_cell_type") or "",
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1

    for reason, ids in skipped.items():
        if ids:
            print(f"[build_sft_jsonl] {reason}: {ids}")
    if skipped["template_mismatch"]:
        print("[build_sft_jsonl] WARNING: template_report.py does not reproduce the GT templates.")
    return n
