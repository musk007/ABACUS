#!/usr/bin/env python3
"""LLM baseline: write an impression report from HALO's intermediate evidence.

Hands the aggregated per-patient record (differential, blast burden, cohort
morphology, flags, QC) and the deterministic subtype prediction to a language
model and asks it for the whole report, with no template and no numeric
containment. This is the ablation of the deterministic renderer: the model gets
exactly the evidence HALO computes, and everything downstream of the evidence is
left to it.

Reports land in <out>/case_<pid>_report.md alongside a manifest.

    python llm_impression_baseline.py \
      --cv-root outputs/v2_reflect \
      --out outputs/llm_baseline/gpt \
      --backend openai --model gpt-5

    # local vLLM / any OpenAI-compatible server
    python llm_impression_baseline.py --cv-root outputs/v2_reflect \
      --out outputs/llm_baseline/medgemma --backend openai \
      --model medgemma-27b-text-it --base-url http://localhost:8000/v1

Keys come from the environment: OPENAI_API_KEY, ANTHROPIC_API_KEY,
GOOGLE_API_KEY. Never pass a key on the command line.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """You are a hematopathologist writing the report for an automated peripheral blood smear analysis.

You are given a structured record of findings computed from the smear, and a subtype prediction from a separate classifier. Write the complete report from them.

Rules:
- Use ONLY the findings in the record. Do not invent cells, counts, percentages or features that are not there.
- Every number you write must appear in the record, or be one of the guideline thresholds 2, 5, 10, 20.
- The record describes leukocyte morphology only. Do not comment on red cells, platelets, dysplasia, staining quality or field placement.
- Write an impression describing what the smear shows, not a confirmed diagnosis. Name candidate subtypes only as differential considerations, each with the test that would confirm it.
- State plainly when the evidence is thin (few informative cells, high artifact fraction, a blast burden near 20%, or a prediction the morphology does not support).

Output a markdown report with these sections, in this order:
**Specimen:** / **Differential:** (a table) / **Cohort morphology:** / **Morphologic interpretation:** / **Diagnostic flags:** / **Impression:** / **Differential considerations:** / **Recommended workup:** / **QC:** / **Not assessed:**
Output the report only, with no preamble."""

USER_TEMPLATE = """Structured findings record (JSON):
```json
{record}
```

Classifier output: predicted subtype {subtype}, confidence {confidence:.2f}.
{review}
Write the report."""


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #

def complete_openai(system: str, user: str, model: str, base_url: str | None, max_tokens: int) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="local") if base_url else OpenAI()
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_completion_tokens=max_tokens,
    )
    return r.choices[0].message.content or ""


def complete_anthropic(system: str, user: str, model: str, base_url: str | None, max_tokens: int) -> str:
    import anthropic

    client = anthropic.Anthropic()
    r = client.messages.create(
        model=model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


def complete_gemini(system: str, user: str, model: str, base_url: str | None, max_tokens: int) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client()
    r = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens),
    )
    return r.text or ""


BACKENDS = {"openai": complete_openai, "anthropic": complete_anthropic, "gemini": complete_gemini}


def complete(backend: str, system: str, user: str, model: str,
             base_url: str | None, max_tokens: int, retries: int = 3) -> str:
    fn = BACKENDS[backend]
    for attempt in range(retries):
        try:
            return fn(system, user, model, base_url, max_tokens)
        except Exception as e:                                   # noqa: BLE001
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"    [retry {attempt + 1}/{retries} in {wait}s] {type(e).__name__}: {e}", flush=True)
            time.sleep(wait)
    return ""


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #

def find_cases(cv_root: Path) -> dict[str, dict[str, Path]]:
    """{pid: {'intermediates', 'classification', 'trace'}} over cv_fold*/<pid>/."""
    cases: dict[str, dict[str, Path]] = {}
    for clf in sorted(cv_root.glob("**/case_*_classification.json")):
        pid = re.sub(r"^case_|_classification$", "", clf.stem)
        inter = clf.with_name(f"case_{pid}_pre_reflection_intermediates.json")
        if not inter.is_file():
            print(f"  [skip] {pid}: no intermediates beside {clf.name}")
            continue
        entry = {"intermediates": inter, "classification": clf}
        final = clf.with_name(f"case_{pid}_final_record.json")
        if final.is_file():
            entry["record"] = final
        trace = clf.with_name(f"case_{pid}_agent_trace.json")
        if trace.is_file():
            entry["trace"] = trace
        cases[pid] = entry
    return cases


def slim_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Accepts a final_record payload or a pre-reflection intermediates payload."""
    rr = payload.get("report_ready") or payload["aggregation"]["report_ready"]
    keep = ("patient_id", "n_images", "n_cells_total", "n_cells_informative", "n_cells_artifact",
            "class_counts", "differential_pct", "blast_pct", "diagnostic_flags",
            "blast_morphology", "cohort_cell_type", "dominant_cell_type", "dominant_cell_pct", "qc")
    return {k: rr[k] for k in keep if k in rr}



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv-root", type=Path, required=True, help="dir holding cv_fold*/<pid>/case_*.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="openai")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (local vLLM etc.)")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--exclude", nargs="*", default=[], help="patient ids to skip")
    ap.add_argument("--limit", type=int, default=0, help="first N patients only (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list cases and print one prompt, call nothing")
    args = ap.parse_args()

    cases = find_cases(args.cv_root)
    for pid in args.exclude:
        cases.pop(pid, None)
    if not cases:
        raise SystemExit(f"No cases found under {args.cv_root}")
    pids = sorted(cases, key=lambda p: (not p.isdigit(), int(p) if p.isdigit() else 0))
    if args.limit:
        pids = pids[: args.limit]
    print(f"{len(pids)} patients from {args.cv_root}")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for i, pid in enumerate(pids, 1):
        dst = args.out / f"case_{pid}_report.md"
        if dst.is_file() and not args.overwrite:
            print(f"[{i}/{len(pids)}] {pid}: exists, skipping")
            manifest.append({"patient_id": pid, "report_md": str(dst), "status": "cached"})
            continue

        src = cases[pid].get("record", cases[pid]["intermediates"])
        record = json.loads(src.read_text())
        clf = json.loads(cases[pid]["classification"].read_text())
        flagged = False
        if "trace" in cases[pid]:
            flagged = bool(json.loads(cases[pid]["trace"].read_text()).get("flagged_for_review"))
        user = USER_TEMPLATE.format(
            record=json.dumps(slim_record(record), indent=2),
            subtype=clf.get("predicted_class", "unknown"),
            confidence=float(clf.get("confidence", 0.0)),
            review=("This case was flagged for human review by the reflection agent.\n" if flagged else ""),
        )

        if args.dry_run:
            print(f"\n--- prompt for patient {pid} ({len(user)} chars) ---\n{user[:2000]}")
            break

        print(f"[{i}/{len(pids)}] {pid}: generating", flush=True)
        text = complete(args.backend, SYSTEM_PROMPT, user, args.model, args.base_url, args.max_tokens).strip()
        m = re.search(r"\*\*Specimen", text)          # drop any reasoning preamble
        if m:
            text = text[m.start():].strip()
        dst.write_text(text, encoding="utf-8")
        manifest.append({
            "patient_id": pid, "report_md": str(dst), "status": "ok",
            "n_chars": len(text), "record": str(src),
            "predicted_class": clf.get("predicted_class"), "confidence": clf.get("confidence"),
            "record": str(src),
        })

    if not args.dry_run:
        (args.out / "manifest.json").write_text(json.dumps({
            "backend": args.backend, "model": args.model, "cv_root": str(args.cv_root),
            "system_prompt": SYSTEM_PROMPT, "n_reports": len(manifest), "cases": manifest,
        }, indent=2), encoding="utf-8")
        print(f"\nWrote {len(manifest)} reports -> {args.out}")


if __name__ == "__main__":
    main()