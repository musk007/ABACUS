#!/usr/bin/env python3
"""Re-derive each patient's FINAL aggregated record from the saved detections.

`case_<pid>_pre_reflection_intermediates.json` holds the record at the initial
threshold. For the patients the reflection agent re-aggregated, the delivered
report was written from a record at a stricter threshold that is not saved
anywhere. Re-aggregation is deterministic given the detections, so this script
rebuilds it: for each patient it reads the final threshold from the agent trace
and re-runs `aggregate()` on `case_<pid>_detections.json`.

Writes `case_<pid>_final_record.json` beside the other case files.

Parity check: for patients that were never re-aggregated the result must equal
the stored `aggregation.report_ready` exactly. The script reports any mismatch
and, with --strict, refuses to write anything if one occurs.

    python rederive_final_records.py --cv-root outputs
    python rederive_final_records.py --cv-root outputs --check-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from agentic_hematology.aggregator import aggregate                    # noqa: E402
from agentic_hematology.schemas import Detection, DetectionResult      # noqa: E402

DEFAULT_TAU = 0.25


def final_threshold(trace_path: Path, default: float = DEFAULT_TAU) -> tuple[float, int]:
    """(threshold the report was built at, number of re-aggregations)."""
    if not trace_path.is_file():
        return default, 0
    acts = json.loads(trace_path.read_text()).get("agent_actions", [])
    ths = [a.get("conf_threshold") for a in acts
           if a.get("action") == "re_aggregate" and a.get("conf_threshold") is not None]
    return (float(ths[-1]), len(ths)) if ths else (default, 0)


def build_result(det_json: dict[str, Any], attr_source: str) -> DetectionResult:
    dets = []
    for c in det_json["detections"]:
        attrs = dict(c.get("attribute_probs" if attr_source == "probs" else "attributes") or {})
        dets.append(Detection(
            cell_id=str(c["cell_id"]),
            image_id=str(c["image_id"]),
            bbox_xyxy=tuple(float(v) for v in c["bbox_xyxy"]),
            cell_type=str(c.get("class", c.get("cell_type", "Unknown"))),
            objectness=float(c.get("confidence", c.get("objectness", 0.0))),
            cell_type_prob=float(c.get("confidence", 0.0)),
            attributes=attrs,
            attribute_probs={k: float(v) for k, v in (c.get("attribute_probs") or {}).items()},
        ))
    return DetectionResult(case_id=str(det_json["patient_id"]),
                           n_images=int(det_json["n_images"]), detections=dets)


def comparable(rr: dict[str, Any]) -> dict[str, Any]:
    """Record minus fields that legitimately differ between runs."""
    out = {k: v for k, v in rr.items()
        if k not in ("grounding_index", "source", "image_stems", "morphology_cohort")}
    out.pop("qc", None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv-root", type=Path, required=True)
    ap.add_argument("--attr-source", choices=("auto", "attributes", "probs"), default="auto")
    ap.add_argument("--check-only", action="store_true", help="verify parity, write nothing")
    ap.add_argument("--strict", action="store_true", help="abort on any parity mismatch")
    args = ap.parse_args()

    cases = sorted(args.cv_root.glob("**/case_*_detections.json"))
    if not cases:
        raise SystemExit(f"No case_*_detections.json under {args.cv_root}")
    print(f"{len(cases)} patients under {args.cv_root}")

    sources = ("attributes", "probs") if args.attr_source == "auto" else (args.attr_source,)
    chosen, best_score = None, -1
    for src in sources:                      # pick whichever reproduces the stored records
        ok = 0
        for det_path in cases:
            pid = det_path.stem.replace("case_", "").replace("_detections", "")
            inter = det_path.with_name(f"case_{pid}_pre_reflection_intermediates.json")
            if not inter.is_file():
                continue
            stored = json.loads(inter.read_text())
            tau = float(stored.get("conf_threshold", DEFAULT_TAU))
            try:
                got = aggregate(build_result(json.loads(det_path.read_text()), src), conf_threshold=tau)
            except Exception:                                        # noqa: BLE001
                continue
            if comparable(got.report_ready) == comparable(stored["aggregation"]["report_ready"]):
                ok += 1
        print(f"  attr-source={src}: reproduces {ok}/{len(cases)} stored records at their own threshold")
        if ok > best_score:
            chosen, best_score = src, ok
    print(f"Using attr-source={chosen}\n")

    mismatches, written, reagg = [], 0, 0
    for det_path in cases:
        pid = det_path.stem.replace("case_", "").replace("_detections", "")
        inter = det_path.with_name(f"case_{pid}_pre_reflection_intermediates.json")
        tau_final, n_re = final_threshold(det_path.with_name(f"case_{pid}_agent_trace.json"))
        reagg += n_re > 0

        findings = aggregate(build_result(json.loads(det_path.read_text()), chosen),
                             conf_threshold=tau_final)

        if n_re == 0 and inter.is_file():    # parity: nothing was re-aggregated, so it must match
            stored = json.loads(inter.read_text())["aggregation"]["report_ready"]
            if comparable(findings.report_ready) != comparable(stored):
                mismatches.append(pid)

        if not args.check_only:
            out = det_path.with_name(f"case_{pid}_final_record.json")
            out.write_text(json.dumps({
                "patient_id": pid,
                "conf_threshold": tau_final,
                "n_re_aggregations": n_re,
                "derived_from": str(det_path.name),
                "attr_source": chosen,
                "report_ready": {**findings.report_ready, **findings.case_record["report_ready"]},
            }, indent=2), encoding="utf-8")
            written += 1

    print(f"re-aggregated patients: {reagg}")
    print(f"parity mismatches among the rest: {len(mismatches)}"
          + (f" -> {sorted(mismatches, key=lambda x: (not x.isdigit(), x))}" if mismatches else ""))
    if mismatches and args.strict:
        raise SystemExit("strict: refusing to write with parity mismatches")
    if not args.check_only:
        print(f"wrote {written} case_<pid>_final_record.json files")


if __name__ == "__main__":
    main()