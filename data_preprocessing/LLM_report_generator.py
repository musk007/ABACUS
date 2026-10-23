
"""
enrich_report_with_llm.py

Standalone post-processor that adds an LLM-generated "Morphologic
interpretation" paragraph to an already-generated template report. The
original report is not modified — the new section is inserted in-memory
and the enriched report is written to a separate output path.

Reads:
- The case JSON (the same payload your template generator consumes).
- The existing markdown report for that case.

Writes:
- An enriched markdown report with one extra section inserted between
  the "Cohort morphology" paragraph and the "Diagnostic flags" line.

LLM hallucination guard:
- After generation, every numeric token in the LLM output is checked
  against the union of numbers present in the case JSON and the source
  report. Novel numbers trigger one retry with feedback; a second
  failure causes the section to be dropped entirely.

Usage
-----
    export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY

    python enrich_report_with_llm.py \\
        --case-json cases.json \\
        --case-id 1 \\
        --report-in  template_reports/case_1_report.md \\
        --report-out enriched_reports/case_1_report.md \\
        --backend claude        # or openai

    # Batch mode — process every report in a directory:
    python enrich_report_with_llm.py \\
        --case-json cases.json \\
        --reports-in  template_reports/ \\
        --reports-out enriched_reports/ \\
        --backend claude
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from report_protocol import run_report_protocol


# ---------------------------------------------------------------------------
# Hardcoded API keys and models
# ---------------------------------------------------------------------------
# WARNING: do not commit this file to a public git repo with real keys.


# OPENAI_API_KEY = ""



OPENAI_MODEL    = "gpt-6-astra"





# ---------------------------------------------------------------------------
# Numeric containment guard
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

class BaseBackend(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        ...



class OpenAIBackend(BaseBackend):
    def __init__(self, model: str = OPENAI_MODEL, max_completion_tokens: int = 1000):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError("pip install openai") from e
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        self.max_tokens = max_completion_tokens

    def complete(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


def build_backend(name: str) -> BaseBackend:
    if name == "openai":
        return OpenAIBackend()
    raise ValueError(f"Unknown backend: {name!r}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case-json", required=True, help="Path to the cases JSON.")
    p.add_argument("--reports-out", required=True, help="Directory for accepted reference reports.")
    p.add_argument("--case-id", default=None, help="Optional single case id.")
    p.add_argument("--backend", choices=["openai"], default="openai")
    args = p.parse_args()

    cases = json.loads(Path(args.case_json).read_text())
    ids = [args.case_id] if args.case_id else list(cases)
    out_dir = Path(args.reports_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = build_backend(args.backend)

    log, n_ok = {}, 0
    for cid in ids:
        report_md, info = run_report_protocol(cid, cases[cid], backend.complete)
        log[cid] = info
        if info["validation"]["passed"]:          # rejected reports are not written
            (out_dir / f"case_{cid}_report.md").write_text(report_md)
            n_ok += 1
        print(f"[{'PASS' if info['validation']['passed'] else 'FAIL'}] case {cid} "
              f"enriched={info['enriched']} attempts={info['attempts']} issues={info['issues']}")

    (out_dir / "_protocol_log.json").write_text(json.dumps(log, indent=2))
    print(f"\n{n_ok}/{len(ids)} reports accepted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
