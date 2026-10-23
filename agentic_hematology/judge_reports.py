#!/usr/bin/env python3
"""LLM-as-judge evaluation of generated hematology reports.

Scores each report against the evidence record it should have been derived from,
on seven criteria, and optionally runs a blinded pairwise comparison between two
systems. Several judges can be run and their agreement reported, so no single
model's opinion carries the result.

Two modes:

  absolute   each system's reports scored independently, 1-5 per criterion
  pairwise   two systems compared side by side, blinded and order-randomised

    # score three systems with two judges
    python judge_reports.py absolute \
      --system halo=outputs/cv_eval_v2/reports \
      --system zeroshot=outputs/cv_eval_zeroshot/reports \
      --system llm_only=outputs/llm_baseline/gpt \
      --records outputs/v2_reflect \
      --judge openai:gpt-5 --judge anthropic:claude-opus-4-6 \
      --out outputs/paper_results/judge/absolute.json

    # head to head against the references
    python judge_reports.py pairwise \
      --system halo=outputs/cv_eval_v2/reports \
      --system reference=LLM_reports \
      --records outputs/v2_reflect \
      --judge openai:gpt-5 \
      --out outputs/paper_results/judge/pairwise_halo_vs_reference.json

Keys come from the environment: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from itertools import combinations
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# rubric
# --------------------------------------------------------------------------- #

CRITERIA = {
    "evidence_faithfulness":
        "Every statement follows from the record. 5 = no claim goes beyond the findings; "
        "3 = one mild overreach; 1 = describes cells or features that are not in the record.",
    "numeric_fidelity":
        "Numbers are correct AND attached to the right quantity. 5 = every figure matches the record "
        "and refers to what the record says it refers to; 3 = a figure is misattributed to the wrong "
        "population; 1 = invented or contradictory figures.",
    "hedging_calibration":
        "Confidence matches evidence strength. 5 = hedges where the evidence is thin (few informative "
        "cells, high artifact fraction, blast burden near 20%, morphology at odds with the prediction) "
        "and commits where it is strong; 1 = confident language on weak evidence, or needless hedging "
        "on a clear picture.",
    "clinical_appropriateness":
        "A hematologist would accept the impression, the differential considerations and the workup for "
        "this picture. 5 = appropriate and correctly prioritised; 1 = clinically wrong or misleading.",
    "completeness":
        "The salient findings and triggered flags are addressed. 5 = nothing important in the record is "
        "ignored; 1 = a major finding is missing.",
    "scope_discipline":
        "Stays within leukocyte morphology: no treatment advice, no claims about red cells, platelets, "
        "dysplasia, staining or field placement, no assertion of a confirmed subtype. 5 = fully within "
        "scope; 1 = repeated violations.",
    "internal_consistency":
        "The sections agree with one another: the interpretation, the impression, the differential and "
        "the flags tell the same story. 5 = fully consistent; 1 = the impression contradicts the body.",
}

BINARY = {
    "asserts_diagnosis":
        "true if the report asserts a leukemia subtype as established fact anywhere outside the "
        "differential considerations, rather than describing what the smear shows.",
    "unsupported_number":
        "true if a figure appears that is not in the record and is not one of the guideline "
        "thresholds 2, 5, 10, 20.",
    "misattributed_number":
        "true if a figure from the record is used for the wrong quantity, population or "
        "denominator, or a split distribution is described as uniform.",
    "out_of_scope_claim":
        "true if the report comments on red cells, platelets, dysplasia, staining quality or field "
        "placement, or recommends treatment.",
}

JUDGE_SYSTEM = """You are an experienced hematopathologist auditing automated blood smear reports.

You are given the structured record of findings computed from a smear and a report written from it. Judge the report against the record only. You are not judging writing style, and you are not being asked whether the underlying cell classification was right: assume the record is what it is, and assess whether the report uses it honestly and well.

Reply with a single JSON object and nothing else."""

ABSOLUTE_TEMPLATE = """Structured findings record:
```json
{record}
```

Report:
---
{report}
---

Score the report on each criterion from 1 to 5:
{criteria}

Also answer these true/false questions:
{binary}

Reply with JSON exactly of this shape:
{{{scores_shape}, "binary": {binary_shape}, "notes": "<one sentence on the most serious problem, or 'none'>"}}"""

PAIRWISE_TEMPLATE = """Structured findings record:
```json
{record}
```

Report A:
---
{report_a}
---

Report B:
---
{report_b}
---

Which report is better, judged against the record on: faithfulness to the evidence, correct and correctly attributed numbers, hedging that matches the strength of the evidence, clinical appropriateness, and staying within leukocyte morphology?

Reply with JSON exactly of this shape:
{{"winner": "A" | "B" | "tie", "margin": "slight" | "clear", "reason": "<one sentence>"}}"""


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #

def _openai(system: str, user: str, model: str, base_url: str | None) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=base_url) if base_url else OpenAI()
    r = client.chat.completions.create(
        model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
    )
    return r.choices[0].message.content or ""


def _anthropic(system: str, user: str, model: str, base_url: str | None) -> str:
    import anthropic

    r = anthropic.Anthropic().messages.create(
        model=model, max_tokens=1500, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


_GEMINI_CLIENT = None

def _gemini(system: str, user: str, model: str, base_url: str | None) -> str:
    global _GEMINI_CLIENT
    from google import genai
    from google.genai import types

    if _GEMINI_CLIENT is None:
        _GEMINI_CLIENT = genai.Client()
    r = _GEMINI_CLIENT.models.generate_content(
        model=model, contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system, response_mime_type="application/json",
            max_output_tokens=1500),
    )
    return r.text or ""


BACKENDS = {"openai": _openai, "anthropic": _anthropic, "gemini": _gemini}

def load_records(root: Path) -> dict[str, dict]:
    recs: dict[str, dict] = {}
    for p in sorted(root.glob("**/case_*_final_record.json")):
        pid = re.sub(r"^case_|_final_record$", "", p.stem)
        recs[pid] = json.loads(p.read_text())["report_ready"]
    for p in sorted(root.glob("**/case_*_pre_reflection_intermediates.json")):   # fallback
        pid = re.sub(r"^case_|_pre_reflection_intermediates$", "", p.stem)
        recs.setdefault(pid, json.loads(p.read_text())["aggregation"]["report_ready"])
    return recs


def ask_judge(judge: str, system: str, user: str, base_url: str | None = None, retries: int = 3) -> dict:
    backend, model = judge.split(":", 1)
    for attempt in range(retries):
        try:
            raw = BACKENDS[backend](system, user, model, base_url)
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                raise ValueError(f"no JSON in judge reply: {raw[:200]}")
            return json.loads(m.group(0))
        except Exception as e:                                   # noqa: BLE001
            if attempt == retries - 1:
                return {"error": f"{type(e).__name__}: {e}"}
            time.sleep(2 ** attempt)
    return {"error": "unreachable"}


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #

def load_reports(spec: str) -> tuple[str, dict[str, str]]:
    """'name=dir' -> (name, {pid: report body}); matches case_<pid>_report.md.

    HALO reports carry appendices (quantitative summary, prediction, grounding)
    that the LLM-only baselines do not produce, so the body is compared.
    """
    name, _, path = spec.partition("=")
    d = Path(path)
    if not d.is_dir():
        raise SystemExit(f"Not a directory: {d}")
    out = {}
    for p in sorted(d.glob("**/case_*_report.md")):
        pid = re.sub(r"^case_|_report$", "", p.stem)
        text = p.read_text(encoding="utf-8")
        for marker in ("## Quantitative Cell Summary", "## Agentic Diagnosis", "## Cell Grounding"):
            text = text.split(marker)[0]
        out[pid] = text.strip()
    if not out:
        raise SystemExit(f"No case_*_report.md under {d}")
    return name, out




def slim(rr: dict) -> dict:
    keep = ("patient_id", "n_images", "n_cells_total", "n_cells_informative", "n_cells_artifact",
            "class_counts", "differential_pct", "blast_pct", "diagnostic_flags",
            "blast_morphology", "cohort_cell_type", "qc")
    return {k: rr[k] for k in keep if k in rr}


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #

def run_absolute(systems: dict[str, dict[str, str]], records: dict[str, dict],
                 judges: list[str], base_url: str | None, limit: int) -> dict:
    criteria_txt = "\n".join(f"- {k}: {v}" for k, v in CRITERIA.items())
    binary_txt = "\n".join(f"- {k}: {v}" for k, v in BINARY.items())
    scores_shape = ", ".join(f'"{k}": <1-5>' for k in CRITERIA)
    binary_shape = "{" + ", ".join(f'"{k}": <true|false>' for k in BINARY) + "}"

    rows: list[dict[str, Any]] = []
    pids = sorted(set.intersection(*(set(r) for r in systems.values())) & set(records),
                  key=lambda p: (not p.isdigit(), int(p) if p.isdigit() else 0))
    if limit:
        pids = pids[:limit]
    print(f"{len(pids)} patients x {len(systems)} systems x {len(judges)} judges "
          f"= {len(pids) * len(systems) * len(judges)} calls")

    for i, pid in enumerate(pids, 1):
        clf_path = next(Path("../outputs/v2_reflect").glob(f"cv_fold*/{pid}/case_{pid}_classification.json"), None)
        clf = json.loads(clf_path.read_text()) if clf_path else {}
        rec = json.dumps({**slim(records[pid]),
                          "classifier": {"predicted_subtype": clf.get("predicted_class"),
                                         "confidence": clf.get("confidence")}}, indent=2)
        for sysname, reports in systems.items():
            user = ABSOLUTE_TEMPLATE.format(
                record=rec, report=reports[pid], criteria=criteria_txt, binary=binary_txt,
                scores_shape=scores_shape, binary_shape=binary_shape)
            for judge in judges:
                print(f"[{i}/{len(pids)}] {pid} {sysname} <- {judge}", flush=True)
                out = ask_judge(judge, JUDGE_SYSTEM, user, base_url)
                rows.append({"patient_id": pid, "system": sysname, "judge": judge, **out})
    return {"mode": "absolute", "criteria": list(CRITERIA), "binary": list(BINARY), "rows": rows}


def run_pairwise(systems: dict[str, dict[str, str]], records: dict[str, dict],
                 judges: list[str], base_url: str | None, limit: int, seed: int) -> dict:
    if len(systems) < 2:
        raise SystemExit("pairwise needs at least two --system entries")
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    pids = sorted(set.intersection(*(set(r) for r in systems.values())) & set(records),
                  key=lambda p: (not p.isdigit(), int(p) if p.isdigit() else 0))
    if limit:
        pids = pids[:limit]

    for a, b in combinations(systems, 2):
        print(f"\n=== {a} vs {b} ({len(pids)} patients) ===")
        for i, pid in enumerate(pids, 1):
            flip = rng.random() < 0.5                    # blind: randomise which side is A
            first, second = (b, a) if flip else (a, b)
            user = PAIRWISE_TEMPLATE.format(
                record=json.dumps(slim(records[pid]), indent=2),
                report_a=systems[first][pid], report_b=systems[second][pid])
            for judge in judges:
                print(f"[{i}/{len(pids)}] {pid} <- {judge}", flush=True)
                out = ask_judge(judge, JUDGE_SYSTEM, user, base_url)
                w = str(out.get("winner", "")).upper()
                winner = first if w == "A" else second if w == "B" else "tie"
                rows.append({"patient_id": pid, "pair": f"{a}|{b}", "judge": judge,
                             "shown_as_A": first, "winner": winner,
                             "margin": out.get("margin"), "reason": out.get("reason"),
                             **({"error": out["error"]} if "error" in out else {})})
    return {"mode": "pairwise", "rows": rows}


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #

def summarise(result: dict) -> dict:
    rows = [r for r in result["rows"] if "error" not in r]
    if result["mode"] == "pairwise":
        out: dict[str, Any] = {}
        for pair in sorted({r["pair"] for r in rows}):
            sub = [r for r in rows if r["pair"] == pair]
            a, b = pair.split("|")
            out[pair] = {"n": len(sub),
                         a: sum(r["winner"] == a for r in sub),
                         b: sum(r["winner"] == b for r in sub),
                         "tie": sum(r["winner"] == "tie" for r in sub),
                         "position_bias_A_rate": round(
                             sum(r["winner"] == r["shown_as_A"] for r in sub) / max(len(sub), 1), 3)}
        return out

    out = {}
    for sysname in sorted({r["system"] for r in rows}):
        sub = [r for r in rows if r["system"] == sysname]
        entry: dict[str, Any] = {"n": len(sub)}
        for c in CRITERIA:
            vals = [float(r[c]) for r in sub if isinstance(r.get(c), (int, float))]
            entry[c] = {"mean": round(statistics.mean(vals), 3),
                        "sd": round(statistics.pstdev(vals), 3) if len(vals) > 1 else 0.0,
                        "n": len(vals)} if vals else None
        for bkey in BINARY:
            vals = [bool((r.get("binary") or {}).get(bkey)) for r in sub if isinstance(r.get("binary"), dict)]
            entry[bkey] = {"rate": round(sum(vals) / len(vals), 3), "count": sum(vals), "n": len(vals)} \
                if vals else None
        out[sysname] = entry

    judges = sorted({r["judge"] for r in rows})
    if len(judges) > 1:                                   # judge agreement on the headline criterion
        by_key = {}
        for r in rows:
            v = r.get("evidence_faithfulness")
            if isinstance(v, (int, float)):
                by_key.setdefault((r["patient_id"], r["system"]), {})[r["judge"]] = float(v)
        paired = [(d[judges[0]], d[judges[1]]) for d in by_key.values()
                  if judges[0] in d and judges[1] in d]
        if len(paired) > 2:
            xs, ys = zip(*paired)
            try:
                out["_judge_agreement"] = {
                    "criterion": "evidence_faithfulness", "judges": judges[:2], "n": len(paired),
                    "pearson_r": round(statistics.correlation(xs, ys), 3),
                    "exact_agreement": round(sum(x == y for x, y in paired) / len(paired), 3),
                    "within_one": round(sum(abs(x - y) <= 1 for x, y in paired) / len(paired), 3)}
            except statistics.StatisticsError:
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("absolute", "pairwise"))
    ap.add_argument("--system", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--records", type=Path, required=True, help="dir holding case_*_findings.json")
    ap.add_argument("--judge", action="append", required=True, metavar="BACKEND:MODEL")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint for a local judge")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    systems = dict(load_reports(s) for s in args.system)
    records = load_records(args.records)
    if not records:
        raise SystemExit(f"No records under {args.records}")
    for name, reps in systems.items():
        print(f"{name}: {len(reps)} reports")

    if args.mode == "absolute":
        result = run_absolute(systems, records, args.judge, args.base_url, args.limit)
    else:
        result = run_pairwise(systems, records, args.judge, args.base_url, args.limit, args.seed)

    result["summary"] = summarise(result)
    result["judges"] = args.judge
    result["systems"] = {k: len(v) for k, v in systems.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + json.dumps(result["summary"], indent=2))
    n_err = sum("error" in r for r in result["rows"])
    print(f"\nWrote {args.out}  ({len(result['rows'])} judgements, {n_err} failed)")


if __name__ == "__main__":
    main()