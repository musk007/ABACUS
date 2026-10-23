#!/usr/bin/env python3
"""Deterministic numeric containment check across report systems.

Extracts every number from each report and tests it against the patient's
evidence record. A figure is admissible when it appears in the record (as a
value or a count), is one of the guideline thresholds, or is a small integer
that cannot carry clinical meaning. Everything else is a containment violation:
a figure the report states and the evidence does not support.

This is the arithmetic complement to the LLM judge. The judge catches numbers
that are real but misused; this catches numbers that were never there.

    python check_numeric_containment.py \
      --records ../outputs/v2_reflect \
      --system halo=../outputs/v2_reflect \
      --system zeroshot=../outputs/v2_zeroshot \
      --system gpt=../outputs/llm_baseline/gpt \
      --system gemini=../outputs/llm_baseline/gemini \
      --system medgemma=../outputs/llm_baseline/medgemma \
      --out ../outputs/paper_results/numeric_containment.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

NUMBER_RE = re.compile(r"(?<![A-Za-z_0-9.])(\d+(?:\.\d+)?)")
GUIDELINE = {"2", "5", "10", "20", "65"}          # thresholds the protocol allows
BODY_MARKERS = ("## Quantitative Cell Summary", "## Agentic Diagnosis", "## Cell Grounding")


def numbers_in(obj: Any, out: set[str]) -> set[str]:
    """Every number appearing anywhere in the record, at several roundings."""
    if isinstance(obj, dict):
        for v in obj.values():
            numbers_in(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            numbers_in(v, out)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        f = float(obj)
        out.add(f"{f:g}")
        out.add(f"{f:.0f}")
        out.add(f"{f:.1f}")
        out.add(f"{f:.2f}")
        if f.is_integer():
            out.add(str(int(f)))
    elif isinstance(obj, str):
        for m in NUMBER_RE.findall(obj):
            out.add(m)
    return out


def allowed_set(record: dict, extra: dict | None = None) -> set[str]:
    allowed = numbers_in(record, set()) | set(GUIDELINE)
    if extra:
        allowed |= numbers_in(extra, set())
    # derived quantities a faithful report may legitimately state
    rr = record
    tot, inf = rr.get("n_cells_total"), rr.get("n_cells_informative")
    art = rr.get("n_cells_artifact")
    for a, b in ((art, tot), (inf, tot)):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b:
            pct = 100.0 * a / b
            allowed |= {f"{pct:.0f}", f"{pct:.1f}", f"{pct:.2f}"}
    return allowed


def normalise(tok: str) -> set[str]:
    """A token matches if any of its roundings is admissible."""
    f = float(tok)
    forms = {tok, f"{f:g}", f"{f:.0f}", f"{f:.1f}", f"{f:.2f}"}
    if f.is_integer():
        forms.add(str(int(f)))
    return forms


def body_of(text: str) -> str:
    for marker in BODY_MARKERS:
        text = text.split(marker)[0]
    return text.strip()


def load_records(root: Path) -> dict[str, dict]:
    recs: dict[str, dict] = {}
    for p in sorted(root.glob("**/case_*_final_record.json")):
        recs[re.sub(r"^case_|_final_record$", "", p.stem)] = json.loads(p.read_text())["report_ready"]
    for p in sorted(root.glob("**/case_*_pre_reflection_intermediates.json")):
        pid = re.sub(r"^case_|_pre_reflection_intermediates$", "", p.stem)
        recs.setdefault(pid, json.loads(p.read_text())["aggregation"]["report_ready"])
    return recs


def load_classifications(root: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(root.glob("**/case_*_classification.json")):
        out[re.sub(r"^case_|_classification$", "", p.stem)] = json.loads(p.read_text())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--system", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--full-report", action="store_true",
                    help="check the whole report instead of the comparable body")
    ap.add_argument("--show", type=int, default=5, help="print N example violations per system")
    args = ap.parse_args()

    records = load_records(args.records)
    clfs = load_classifications(args.records)
    if not records:
        raise SystemExit(f"No records under {args.records}")

    summary: dict[str, Any] = {}
    per_case: list[dict[str, Any]] = []

    for spec in args.system:
        name, _, path = spec.partition("=")
        d = Path(path)
        counts, examples = [], []
        for p in sorted(d.glob("**/case_*_report.md")):
            pid = re.sub(r"^case_|_report$", "", p.stem)
            if pid not in records:
                continue
            text = p.read_text(encoding="utf-8")
            if not args.full_report:
                text = body_of(text)
            text = re.sub(r"t\(\d+;\d+\)|del\(\d+[pq]?\)|trisomy\s+\d+|CD\d+|\bIGHV\b|\bTP53\b|\bBCR::ABL1\b|\bPML::RARA\b|\bKMT2A\b", " ", text)
            allowed = allowed_set(records[pid], clfs.get(pid))
            bad = [t for t in NUMBER_RE.findall(text)
                   if not (normalise(t) & allowed) and (float(t) >= 10 or '.' in t)]
            counts.append(len(bad))
            per_case.append({"system": name, "patient_id": pid,
                             "n_numbers": len(NUMBER_RE.findall(text)),
                             "n_violations": len(bad), "violations": sorted(set(bad))[:20]})
            if bad and len(examples) < args.show:
                examples.append({"patient_id": pid, "violations": sorted(set(bad))[:8]})

        n = len(counts)
        summary[name] = {
            "n_reports": n,
            "reports_with_violation": sum(c > 0 for c in counts),
            "violation_rate": round(sum(c > 0 for c in counts) / n, 3) if n else None,
            "mean_violations_per_report": round(statistics.mean(counts), 2) if n else None,
            "max_violations": max(counts) if n else None,
            "examples": examples,
        }
        s = summary[name]
        print(f"{name:10s} {s['reports_with_violation']:>3}/{n} reports with a violation "
              f"({100 * (s['violation_rate'] or 0):.1f}%), mean {s['mean_violations_per_report']} per report")
        for e in examples:
            print(f"             patient {e['patient_id']}: {', '.join(e['violations'])}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"guideline_thresholds": sorted(GUIDELINE), "body_only": not args.full_report,
         "summary": summary, "per_case": per_case}, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()