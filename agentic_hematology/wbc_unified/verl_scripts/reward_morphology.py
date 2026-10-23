"""GRPO reward for the morphologic-interpretation task (data_source=leukemia_morph)."""
from __future__ import annotations

import json
import re

DATA_SOURCE = "leukemia_morph"
NUMERIC_RE = re.compile(r"(?<![A-Za-z_])(\d+(?:\.\d+)?)")
WHITELIST = {"20", "10", "5", "2"}          # guideline thresholds
MARKER = "**Morphologic interpretation:**"
BANNED = (
    "fab", "recommend", "workup", "bone marrow", "flow cytometry",
    "impression:", "qc:", "differential considerations",
)


def _coerce(x):
    return x.item() if hasattr(x, "item") else x


def _allowed_numbers(extra: dict) -> set[str]:
    allowed = set(WHITELIST)
    allowed.update(NUMERIC_RE.findall(extra.get("case_json", "")))
    allowed.update(NUMERIC_RE.findall(extra.get("template_md", "")))
    return allowed


def _numeric_score(text: str, allowed: set[str]) -> tuple[float, int]:
    toks = NUMERIC_RE.findall(text)
    novel = [t for t in toks if t not in allowed]
    return (1.0 if not novel else 0.0), len(novel)

def _format_score(text: str) -> float:
    s = 0.0
    if text.strip().startswith(MARKER):
        s += 0.5
    body = text.replace(MARKER, " ")
    n_sent = len([x for x in re.split(r"(?<=[.!?])\s+", body.strip()) if len(x) > 20])
    if 3 <= n_sent <= 5:
        s += 0.3
    elif 2 <= n_sent <= 6:
        s += 0.15
    if "\n-" not in text and "\n*" not in text and "#" not in text:
        s += 0.2                                   # prose, no lists or headers
    return min(1.0, s)


ATTR_WORDS = {
    "cell_size": ("size", "small", "medium", "large", "pleomorph"),
    "nuclear_chromatio": ("chromatin",),
    "nuclear_shape": ("contour", "nuclear shape", "irregular", "regular", "cleaved", "folded"),
    "nucleolus": ("nucleol",),
    "cytoplasm": ("cytoplasm", "scanty", "abundant"),
    "cytoplasmic_basophilia": ("basophil",),
    "cytoplasmic_vacuoles": ("vacuol",),
}


def _content_score(text: str, extra: dict) -> float:
    low = text.lower()
    try:
        case = json.loads(extra.get("case_json", "{}"))
    except json.JSONDecodeError:
        case = {}
    morph = (case.get("report_ready") or {}).get("blast_morphology") or {}

    covered = sum(
        1 for k, words in ATTR_WORDS.items()
        if k in morph and any(w in low for w in words)
    )
    score = 0.6 * min(1.0, covered / 4.0)

    cohort = str(extra.get("cohort_cell_type", "")).lower()
    if cohort and cohort.split()[-1] in low:
        score += 0.2

    # borderline dominance (<65%) should be qualified in the text
    borderline = [v for v in morph.values() if 0 < float(v.get("dominance_pct", 100)) < 65]
    hedges = ("borderline", "not uniform", "heterogen", "minority", "only", "rather than")
    if borderline:
        score += 0.2 if any(h in low for h in hedges) else 0.0
    else:
        score += 0.2

    return max(0.0, min(1.0, score - 0.5 * sum(1 for b in BANNED if b in low)))


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if str(_coerce(data_source)) != DATA_SOURCE:
        return 0.0
    extra = _coerce(extra_info) or {}
    extra = dict(extra) if isinstance(extra, dict) else {}

    text = (solution_str or "").strip()
    if not text:
        return {"score": 0.0, "acc": 0.0}

    allowed = _allowed_numbers(extra)
    num_score, n_novel = _numeric_score(text, allowed)
    fmt = _format_score(text)
    content = _content_score(text, extra)

    n_words = len(text.split())
    length_pen = 0.0 if n_words <= 140 else min(0.3, (n_words - 140) / 200.0)

    total = 0.45 * num_score + 0.30 * content + 0.25 * fmt - length_pen
    total = round(max(0.0, min(1.0, total)), 4)
    return {
        "score": total,
        "acc": total,
        "num_score": round(num_score, 4),
        "n_novel_numbers": float(n_novel),
        "content_score": round(content, 4),
        "fmt_score": round(fmt, 4),
        "n_chars": float(len(text)),
        "length_pen": round(length_pen, 4)
    }