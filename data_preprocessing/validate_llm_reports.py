"""
validate_enriched_reports.py
============================
Validates LLM-enriched hematology reports against the source JSON.

Two checks per case:

1. CONSISTENCY — numbers in the template-generated sections (specimen,
   differential, blast burden sentence, QC) must match the JSON exactly
   (one-decimal-place tolerance).

2. HALLUCINATION GUARD — every numeric token in the LLM-generated
   "Morphologic interpretation" section must already appear in the JSON
   or in the rest of the report. Catches invented percentages.

Reports are paired to JSON cases by filename: case_<N>_report.md → "<N>".

Usage
-----
    python validate_enriched_reports.py \\
        --reports-dir enriched_reports/ \\
        --case-json   patient_WBC_stats_NoOveralp.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PCT_TOLERANCE = 0.11   # one-decimal-place rounding wiggle room
NUM_TOL = 0.06         # float-comparison tolerance for hallucination check

# Numbers always allowed in LLM output even if not in the inputs.
DEFAULT_WHITELIST: set[float] = {20.0, 10.0, 5.0, 2.0, 65.0}   # must match report_protocol.NUMERIC_WHITELIST

CELL_TYPE_MAP = {
    "Lymphoblasts": "lymphoblast",
    "Myeloblasts": "myeloblast",
    "Monoblasts": "monoblast",
    "Promonocytes": "promonocyte",
    "Abnormal promyelocytes": "abnormal promyelocyte",
    "Atypical lymphocytes": "atypical lymphocyte",
    "Lymphocytes": "lymphocyte",
    "Monocytes": "monocyte",
    "Neutrophils": "neutrophil",
    "Eosinophils": "eosinophil",
    "Basophils": "basophil",
    "Myelocytes": "myelocyte",
    "Metamyelocytes": "metamyelocyte",
}


def _build_label_lookup() -> dict[str, str]:
    """Build a case-insensitive lookup that accepts singular OR plural labels.

    The upstream template generator falls back to `cell.capitalize()` for any
    cell type not in its CELL_PRINT_NAME map, which produces a singular label
    (e.g. "Promonocyte" instead of "Promonocytes"). The validator must handle
    both forms gracefully.
    """
    out: dict[str, str] = {}
    for display, key in CELL_TYPE_MAP.items():
        norm = display.lower()
        out[norm] = key
        # Plural → singular
        if norm.endswith("s"):
            out[norm[:-1]] = key
        # Singular → plural (covers edge cases like "Promonocyte")
        else:
            out[norm + "s"] = key
    return out


_LABEL_LOOKUP = _build_label_lookup()

DIAGNOSIS_ALIASES = {
    "APML": ["APML", "APL", "Acute Promyelocytic Leukemia"],
    "APL":  ["APL", "APML", "Acute Promyelocytic Leukemia"],
    "AML":  ["AML", "Acute Myeloid Leukemia"],
    "ALL":  ["ALL", "Acute Lymphoblastic Leukemia"],
    "CML":  ["CML", "Chronic Myeloid Leukemia"],
    "CLL":  ["CLL", "Chronic Lymphocytic Leukemia", "Small Lymphocytic Lymphoma"],
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class Result:
    case_id: str
    passed: bool = True
    consistency_issues: list[str] = field(default_factory=list)
    hallucination_issues: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    label_qc: list[str] = field(default_factory=list)   # impression vs dataset label; not a failure


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

NUMERIC_TOKEN_RE = re.compile(r"(?<![A-Za-z_])(\d+(?:\.\d+)?)")
CASE_ID_RE = re.compile(r"case_([^_]+(?:_[^_]+)*?)_report")


def parse_report(text: str) -> tuple[dict[str, Any], str, list[str]]:
    """
    Returns (parsed_fields, llm_section_text, parse_errors).
    llm_section_text is "" if no Morphologic interpretation block is present.
    """
    parsed: dict[str, Any] = {}
    errors: list[str] = []

    # Case ID from H1
    m = re.search(r"#\s+Hematology Report\s+—\s+Case\s+(\S+)", text)
    parsed["case_id"] = m.group(1) if m else None
    if not m:
        errors.append("could not parse case id from title")

    # Specimen line
    m = re.search(
        r"\*\*Specimen:\*\*\s*Peripheral blood smear,\s*(\d+)\s*fields of view,\s*"
        r"(\d+)\s*of\s*(\d+)\s*annotated objects classified as informative WBCs\s*"
        r"\((\d+)\s*artefacts excluded\)\.",
        text,
    )
    if m:
        parsed["n_images"] = int(m.group(1))
        parsed["n_wbc"] = int(m.group(2))
        parsed["n_total"] = int(m.group(3))
        parsed["n_artefacts"] = int(m.group(4))
    else:
        errors.append("could not parse specimen line")

    # Differential table
    parsed["differential"] = {}
    for label, pct in re.findall(r"\|\s*([A-Za-z\s\-]+?)\s*\|\s*([\d.]+)%\s*\|", text):
        clean = label.strip()
        if clean == "Cell type":
            continue
        key = _LABEL_LOOKUP.get(clean.lower())
        if key:
            parsed["differential"][key] = float(pct)
        else:
            errors.append(f"unrecognised cell label in differential: {clean!r}")

    # Impression
    m = re.search(r"\*\*Impression:\*\*\s*(.+?)\.", text)
    parsed["impression"] = m.group(1).strip() if m else None

    # Blast-equivalent burden (the template always uses this phrase).
    m = re.search(r"blast-equivalent burden(?: of| is)?\s*([\d.]+)%", text)
    parsed["blast_pool_pct"] = float(m.group(1)) if m else None

    # "<Cell type>s comprise N% of WBCs" -> checked against that cell type's own percentage.
    parsed["comprise"] = {}
    for label, pct in re.findall(
        r"(Lymphoblasts|Myeloblasts|Monoblasts|Abnormal promyelocytes)\s+comprise\s*([\d.]+)%", text
    ):
        key = _LABEL_LOOKUP.get(label.lower())
        if key:
            parsed["comprise"][key] = float(pct)

    # QC line
    m = re.search(
        r"\*\*QC:\*\*\s*(\d+)\s*FOVs;\s*(\d+)/(\d+)\s*cells classifiable\s*"
        r"\(([\d.]+)%\s*artefact\);\s*cohort cell count\s*=\s*(\d+)[;.]",
        text,
    )
    if m:
        parsed["qc_n_images"] = int(m.group(1))
        parsed["qc_n_wbc"] = int(m.group(2))
        parsed["qc_n_total"] = int(m.group(3))
        parsed["qc_artefact_pct"] = float(m.group(4))
        parsed["qc_n_cohort"] = int(m.group(5))
    else:
        errors.append("could not parse QC line")

    # Morphologic interpretation section (LLM-generated, optional)
    llm_section = ""
    m = re.search(
        r"\*\*Morphologic interpretation:\*\*\s*(.+?)(?=\n\*\*|\Z)",
        text, re.DOTALL,
    )
    if m:
        llm_section = m.group(1).strip()

    return parsed, llm_section, errors


# ---------------------------------------------------------------------------
# Consistency check (template numbers vs JSON)
# ---------------------------------------------------------------------------

def compare_case(parsed: dict[str, Any], case: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    rr = case["report_ready"]
    qc = rr["qc"]

    scalars = {
        "n_images":         case["n_images"],
        "n_wbc":            case["n_cells_identified_wbc"],
        "n_total":          case["n_cells_total"],
        "n_artefacts":      qc["n_artifacts"],
        "qc_n_images":      qc["n_fields_of_view"],
        "qc_n_wbc":         qc["n_identified_wbc"],
        "qc_n_total":       qc["n_annotated_cells"],
        "qc_n_cohort":      qc["n_cells_in_cohort"],
    }
    for k, expected in scalars.items():
        got = parsed.get(k)
        if got is not None and got != expected:
            issues.append(f"{k}: report={got}, json={expected}")

    # Artefact percentage
    if "qc_artefact_pct" in parsed and qc["n_annotated_cells"]:
        expected = round(qc["n_artifacts"] * 100 / qc["n_annotated_cells"], 1)
        if abs(parsed["qc_artefact_pct"] - expected) > PCT_TOLERANCE:
            issues.append(
                f"qc_artefact_pct: report={parsed['qc_artefact_pct']}, json={expected}"
            )

    # Blast pool percentage
    if parsed.get("blast_pool_pct") is not None:
        expected = round(rr["blast_pool_percentage_of_wbc"], 1)
        if abs(parsed["blast_pool_pct"] - expected) > PCT_TOLERANCE:
            issues.append(
                f"blast_pool_pct: report={parsed['blast_pool_pct']}, json={expected}"
            )

    # Differential percentages
    clinical = {k.lower(): v for k, v in case.get("cell_percentages_clinical", {}).items()}
    for cell, report_pct in parsed.get("comprise", {}).items():
        expected = round(float(clinical.get(cell, 0.0)), 1)
        if abs(report_pct - expected) > PCT_TOLERANCE:
            issues.append(f"rationale {cell}: report={report_pct}, json={expected}")
    for cell, report_pct in parsed.get("differential", {}).items():
        if cell.lower() not in clinical:
            issues.append(f"differential {cell!r} missing in json")
            continue
        expected = round(float(clinical[cell.lower()]), 1)
        if abs(report_pct - expected) > PCT_TOLERANCE:
            issues.append(f"differential {cell}: report={report_pct}, json={expected}")

    # Impression vs metadata diagnosis (with aliases)
    diag = case.get("metadata_filename_diagnosis")
    impression = parsed.get("impression") or ""
    if diag and impression:
        aliases = DIAGNOSIS_ALIASES.get(diag, [diag])
        lower = impression.lower()
        if not any(a.lower() in lower for a in aliases):
            issues.append(f"LABEL_QC impression mismatch: report={impression!r}, json={diag!r}")

    return issues


# ---------------------------------------------------------------------------
# Hallucination check (LLM section vs allowed numbers)
# ---------------------------------------------------------------------------

def _flatten_floats(obj: Any, out: set[float]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _flatten_floats(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _flatten_floats(v, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        f = float(obj)
        out.add(f)
        # Also add common rendered forms (one-decimal-place rounding).
        out.add(round(f, 1))
    elif isinstance(obj, str):
        for tok in NUMERIC_TOKEN_RE.findall(obj):
            try:
                out.add(float(tok))
            except ValueError:
                pass


def collect_allowed(case: dict[str, Any], report_text_excl_llm: str) -> set[float]:
    allowed: set[float] = set(DEFAULT_WHITELIST)
    _flatten_floats(case, allowed)
    for tok in NUMERIC_TOKEN_RE.findall(report_text_excl_llm):
        try:
            allowed.add(float(tok))
        except ValueError:
            pass
    return allowed


def find_novel(llm_text: str, allowed: set[float]) -> list[str]:
    novel: list[str] = []
    seen: set[str] = set()
    for tok in NUMERIC_TOKEN_RE.findall(llm_text):
        if tok in seen:
            continue
        seen.add(tok)
        try:
            v = float(tok)
        except ValueError:
            continue
        if not any(abs(v - a) <= NUM_TOL for a in allowed):
            novel.append(tok)
    return novel


# ---------------------------------------------------------------------------
# Per-file orchestrator
# ---------------------------------------------------------------------------

def validate_one(case_id: str, case: dict[str, Any], report_text: str) -> Result:
    parsed, llm_section, parse_errors = parse_report(report_text)
    r = Result(case_id=case_id, parse_errors=parse_errors)

    issues = compare_case(parsed, case)
    r.label_qc = [i for i in issues if i.startswith("LABEL_QC")]
    r.consistency_issues = [i for i in issues if not i.startswith("LABEL_QC")]

    if llm_section:
        # Exclude the LLM section itself from the source-of-truth text;
        # otherwise the LLM's own outputs would "vouch for" themselves.
        report_excl_llm = report_text.replace(llm_section, "")
        allowed = collect_allowed(case, report_excl_llm)
        novel = find_novel(llm_section, allowed)
        if novel:
            # Include short context for each novel number.
            for tok in novel:
                m = re.search(rf"(?<![A-Za-z_]){re.escape(tok)}", llm_section)
                start = max(0, m.start() - 30) if m else 0
                end = min(len(llm_section), m.end() + 30) if m else 0
                snippet = llm_section[start:end].replace("\n", " ").strip()
                r.hallucination_issues.append(f"{tok!r}: '...{snippet}...'")

    r.passed = not (r.consistency_issues or r.hallucination_issues or r.parse_errors)
    return r


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--reports-dir", required=True, help="Directory of *.md reports.")
    p.add_argument("--case-json", required=True, help="Single JSON file keyed by case id.")
    p.add_argument("--quiet", action="store_true", help="Only print FAIL cases.")
    args = p.parse_args()

    cases = json.loads(Path(args.case_json).read_text())
    paths = sorted(Path(args.reports_dir).glob("*.md"))
    if not paths:
        print(f"No *.md files in {args.reports_dir}", file=sys.stderr)
        return 2

    n_pass = n_fail = 0
    for path in paths:
        m = CASE_ID_RE.search(path.stem)
        if not m:
            print(f"[SKIP] {path.name}: cannot parse case id")
            continue
        cid = m.group(1)
        if cid not in cases:
            print(f"[SKIP] {path.name}: case {cid!r} not in JSON")
            continue

        r = validate_one(cid, cases[cid], path.read_text())
        if r.passed:
            n_pass += 1
            if not args.quiet:
                print(f"[PASS] case {cid}")
        else:
            n_fail += 1
            print(f"[FAIL] case {cid}")
            for issue in r.parse_errors:
                print(f"   parse: {issue}")
            for issue in r.consistency_issues:
                print(f"   consistency: {issue}")
            for issue in r.hallucination_issues:
                print(f"   hallucination: {issue}")

    print(f"\nTotal: {n_pass} pass, {n_fail} fail")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())