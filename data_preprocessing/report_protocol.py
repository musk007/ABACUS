from __future__ import annotations
import json

from collections import Counter, defaultdict
import copy
import re
from typing import Any, Callable

# from report_prompt import SYSTEM_PROMPT, USER_TEMPLATE
try:
    from data_preprocessing.report_prompt import SYSTEM_PROMPT, USER_TEMPLATE
except ModuleNotFoundError:
    from report_prompt import SYSTEM_PROMPT, USER_TEMPLATE

ATTRIBUTE_VALUE_MAPS: dict[str, dict[int, str]] = {
    "cell_size":              {0: "small",   1: "medium",     2: "large",               4: "n_a"},
    "nuclear_chromatio":      {0: "open",    1: "coarse",                               4: "n_a"},
    "nuclear_shape":          {0: "regular", 1: "irregular",  2: "cleaved_or_folded",   4: "n_a"},
    "nucleolus":              {0: "inconspicuous", 1: "prominent",                      4: "n_a"},
    "cytoplasm":              {0: "scanty",  1: "abundant",                             4: "n_a"},
    "cytoplasmic_basophilia": {0: "slight",  1: "moderate",                             4: "n_a"},
    "cytoplasmic_vacuoles":   {0: "absent",  1: "prominent",                            4: "n_a"},
}
ATTRIBUTE_KEYS = list(ATTRIBUTE_VALUE_MAPS.keys())
N_A_CODE = 4
PERCENT_DECIMALS = 2

# Strict 0-indexed schema matching your AttriDet text file findings (none = 0)
YOLO_CLASS_NAMES: dict[int, str] = {
    0:  "none",
    1:  "myeloblast",
    2:  "lymphoblast",
    3:  "neutrophil",
    4:  "atypical lymphocyte",
    5:  "promonocyte",
    6:  "monoblast",
    7:  "lymphocyte",
    8:  "myelocyte",
    9:  "abnormal promyelocyte",
    10: "monocyte",
    11: "metamyelocyte",
    12: "eosinophil",
    13: "basophil"
}

CELL_TYPE_PRIORITY: dict[str, int] = {
    "abnormal promyelocyte": 1,   # APML — emergency (DIC), near-pathognomonic
    "myeloblast":            2,   # AML-defining blast
    "lymphoblast":           3,   # ALL-defining blast
    "monoblast":             4,   # AML monocytic blast-equivalent
    "promonocyte":           5,   # AML monocytic, one step more mature
    "atypical lymphocyte":   6,   # reactive/mimic flag — supportive only
    "lymphocyte":            7,   # mature lymphoid — meaningful only in CLL context
}

COHORT_ELIGIBLE = set(CELL_TYPE_PRIORITY)        # only these can be selected as a cohort
DEFAULT_PRIORITY = 50                             # any unseen type sorts last


TIER1_BLAST_TYPES = {"myeloblast", "lymphoblast", "monoblast"}
MIN_BLAST_TIE_COUNT = 3                            # floor below which a tie is treated as noise

CLINICAL_GROUPS: dict[str, set[str]] = {
    "blasts":               {"myeloblast", "lymphoblast", "monoblast"},
    "abnormal_precursors":  {"abnormal promyelocyte"},
    "intermediate_myeloid": {"promonocyte", "myelocyte", "metamyelocyte"},
    "mature_granulocytes":  {"neutrophil", "eosinophil", "basophil"},
    "lymphoid":             {"lymphocyte", "atypical lymphocyte"},
    "monocytic":            {"monocyte"},
    "unidentified":         {"none"},
}

BLAST_THRESHOLD_PCT = 20.0
BASOPHILIA_THRESHOLD_PCT = 2.0
LOW_CELL_COUNT_THRESHOLD = 30


def _percentages_from_counts(counts: dict, denominator: int) -> dict:
    if not denominator:
        return {}
    return {k: round(v / denominator * 100, PERCENT_DECIMALS) for k, v in counts.items()}

def _attribute_summary(attribute_counts: dict) -> dict:
    out = {}
    for attr_key in ATTRIBUTE_KEYS:
        raw = dict(sorted(attribute_counts.get(attr_key, {}).items(), key=lambda kv: -kv[1]))
        informative = {k: v for k, v in raw.items() if k != "n_a"}
        n_inf = sum(informative.values())
        out[attr_key] = {
            "counts": raw,
            "n_informative_cells": n_inf,
            "n_na_cells": raw.get("n_a", 0),
            "percentages": _percentages_from_counts(informative, n_inf)
        }
    return out

def _select_cohort_from_counts(counts: dict) -> tuple[set[str], bool]:
    candidates = {ct: counts.get(ct, 0) for ct in COHORT_ELIGIBLE if counts.get(ct, 0) > 0}
    if not candidates:
        return set(), False

    max_count = max(candidates.values())
    tied = [ct for ct, v in candidates.items() if v == max_count]

    ambiguous = (
        len([ct for ct in tied if ct in TIER1_BLAST_TYPES]) >= 2
        and max_count >= MIN_BLAST_TIE_COUNT
    )

    # Deterministic AND medically ordered: lowest rank wins among ties.
    winner = min(tied, key=lambda ct: CELL_TYPE_PRIORITY.get(ct, DEFAULT_PRIORITY))
    return {winner}, ambiguous

def _build_differential_alerts(counts, pct_clinical, group_percentages,
                               blast_pool_pct, cohort_ambiguous) -> list[dict]:
    alerts = []
    neutrophil_pct      = pct_clinical.get("neutrophil", 0.0)
    dominant_cell       = max(pct_clinical, key=pct_clinical.get) if pct_clinical else None
    precursors_present  = group_percentages.get("intermediate_myeloid", 0.0) >= 10.0
    basophils_absent    = counts.get("basophil", 0) == 0

    if (dominant_cell == "neutrophil"
            and precursors_present
            and blast_pool_pct < BLAST_THRESHOLD_PCT
            and basophils_absent):
        alerts.append({
            "code": "CML_PATTERN_NO_BASOPHILIA",
            "severity": "review",
            "message": ("Smear shows a predominance of neutrophils followed by myelocytes "
                        "and other granulocytic precursors. The absence of identifiable "
                        "basophils is unusual for CML and raises the possibility of "
                        "alternative diagnoses such as a leukemoid reaction or severe "
                        "infection/sepsis."),
            "recommendation": ("Correlate with the CBC, differential count, and BCR::ABL1 "
                               "testing."),
            "suggested_action": "FLAG_FOR_REVIEW",
        })

    # Mixed-lineage blast tie (MPAL — outside the 5-class taxonomy)
    if cohort_ambiguous:
        alerts.append({
            "code": "MIXED_LINEAGE_BLAST_TIE",
            "severity": "review",
            "message": ("Co-dominant myeloid and lymphoid blast populations. Pattern is "
                        "compatible with mixed-phenotype acute leukemia, which lies outside "
                        "the five supported subtypes."),
            "recommendation": "Flow cytometry / immunophenotyping required to resolve lineage.",
            "suggested_action": "FLAG_FOR_REVIEW",
        })

    # Blast burden inconsistent with a chronic picture
    mature_lymphoid_dominant = pct_clinical.get("lymphocyte", 0.0) >= 50.0
    if blast_pool_pct >= BLAST_THRESHOLD_PCT and mature_lymphoid_dominant:
        alerts.append({
            "code": "BLASTS_IN_CHRONIC_PATTERN",
            "severity": "review",
            "message": ("Blast pool meets the acute threshold alongside a mature lymphoid "
                        "background — incompatible with a stable chronic process."),
            "recommendation": "Consider transformation/blast crisis; correlate clinically.",
            "suggested_action": "FLAG_FOR_REVIEW",
        })

    return alerts

def _build_report_ready(rec: dict, pct_clinical: dict) -> dict:
    counts = rec["cell_counts"]

    n_wbc = sum(counts.get(ct, 0) for ct in counts if ct != "none")

    group_counts = {g: sum(counts.get(ct, 0) for ct in members) for g, members in CLINICAL_GROUPS.items()}
    group_percentages = _percentages_from_counts(group_counts, n_wbc)

    blast_pool_count = group_counts["blasts"] + group_counts["abnormal_precursors"]
    blast_pool_pct = round((blast_pool_count / n_wbc * 100), PERCENT_DECIMALS) if n_wbc else 0.0
    myeloid_blastlike_count = sum(counts.get(ct, 0) for ct in ("myeloblast", "monoblast", "promonocyte"))
    myeloid_blastlike_pct = round((myeloid_blastlike_count / n_wbc * 100), PERCENT_DECIMALS) if n_wbc else 0.0

    flags = {
        "blasts_present":                  blast_pool_count > 0,
        "blast_threshold_met":             blast_pool_pct >= BLAST_THRESHOLD_PCT,
        "abnormal_promyelocytes_present":  counts.get("abnormal promyelocyte", 0) > 0,
        "atypical_lymphocytes_present":    counts.get("atypical lymphocyte", 0) > 0,
        "basophilia_present":              pct_clinical.get("basophil", 0.0) >= BASOPHILIA_THRESHOLD_PCT,
        "eosinophilia_present":            pct_clinical.get("eosinophil", 0.0) >= 5.0,
        "left_shifted_myeloid":            group_percentages.get("intermediate_myeloid", 0.0) >= 10.0,
        "monocytosis_present":             pct_clinical.get("monocyte", 0.0) >= 10.0,
    }

    # Deterministic, medically-ranked cohort selection (returns ambiguity flag for tier-1 blast ties)
    cohort_types, cohort_ambiguous = _select_cohort_from_counts(counts)
    merged_cohort_attrs = {k: Counter() for k in ATTRIBUTE_KEYS}
    n_cohort = 0
    for ct in cohort_types:
        n_cohort += counts.get(ct, 0)
        for attr_key in ATTRIBUTE_KEYS:
            for state, count in rec["celltype_attribute_counts"].get(ct, {}).get(attr_key, {}).items():
                merged_cohort_attrs[attr_key][state] += count

    cohort_summary = _attribute_summary(merged_cohort_attrs)
    blast_morphology = {}
    for k in ATTRIBUTE_KEYS:
        pcts = cohort_summary[k]["percentages"]
        dom_state = next(iter(pcts.keys())) if pcts else None
        dom_pct = next(iter(pcts.values())) if pcts else 0.0
        blast_morphology[k] = {
            "dominant": next(iter(pcts.keys())) if pcts else None,
            "dominance_pct": next(iter(pcts.values())) if pcts else 0.0,
            "distribution": pcts,
        }

    is_sparse_skew_suspected = (
        blast_pool_count > 0
        and blast_pool_pct < BLAST_THRESHOLD_PCT
        and n_wbc < LOW_CELL_COUNT_THRESHOLD
    )

    # Differential contradiction layer — patterns that point outside / against the 5-class taxonomy
    alerts = _build_differential_alerts(
        counts=counts,
        pct_clinical=pct_clinical,
        group_percentages=group_percentages,
        blast_pool_pct=blast_pool_pct,
        cohort_ambiguous=cohort_ambiguous,
    )

    qc = {
        "n_annotated_cells": sum(counts.values()),
        "n_identified_wbc": n_wbc,
        "n_artifacts": counts.get("none", 0),
        "n_fields_of_view": rec["n_images"],
        "n_cells_in_cohort": n_cohort,
        "low_cell_count_warning": n_wbc < LOW_CELL_COUNT_THRESHOLD,
        "sparse_annotation_skew_warning": is_sparse_skew_suspected,
        "cohort_selection_ambiguous": cohort_ambiguous,
        "differential_alerts": alerts,
        "requires_review": any(a["severity"] == "review" for a in alerts),
        "cohort_cell_type": next(iter(cohort_types), None),
    }

    return {
        "blast_pool_percentage_of_wbc": blast_pool_pct,
        "myeloid_blastlike_percentage_of_wbc": myeloid_blastlike_pct,
        "dominant_cell_type": max(pct_clinical, key=pct_clinical.get) if pct_clinical else None,
        "dominant_cell_pct": max(pct_clinical.values()) if pct_clinical else 0.0,
        "cohort_cell_type": next(iter(cohort_types), None),
        "diagnostic_flags": flags,
        "blast_morphology": blast_morphology,
        "qc": qc,
    }

def new_store() -> dict:
    """Per-patient counters shared by the reference and inference paths."""
    return {
        "cell_counts": Counter(),
        "attribute_counts": {k: Counter() for k in ATTRIBUTE_KEYS},
        "celltype_attribute_counts": defaultdict(lambda: {k: Counter() for k in ATTRIBUTE_KEYS}),
    }


def add_cell(store: dict, cell_type: str, attr_labels: dict[str, str]) -> None:
    """Add one cell. Artefacts ('none') are counted but contribute no morphology."""
    store["cell_counts"][cell_type] += 1
    if cell_type == "none":
        return
    for attr_key, label in attr_labels.items():
        store["attribute_counts"][attr_key][label] += 1
        store["celltype_attribute_counts"][cell_type][attr_key][label] += 1

# Annotation states the attribute predictor cannot output are mapped onto its vocabulary.

PRED_ATTR_KEY = {
    "Nuclear_Chromatin": "nuclear_chromatio",
    "Nuclear_Shape": "nuclear_shape",
    "Nucleolus": "nucleolus",
    "Cytoplasm": "cytoplasm",
    "Cytoplasmic_Basophilia": "cytoplasmic_basophilia",
    "Cytoplasmic_Vacuoles": "cytoplasmic_vacuoles",
}
CELL_SIZE_STATES = ("small", "medium", "large")


def labels_from_annotation_codes(codes: list[int]) -> dict[str, str]:
    """Reference path: AttriDet codes -> state labels. N/A and undocumented codes are missing."""
    out = {}
    for attr_key, code in zip(ATTRIBUTE_KEYS, codes):
        label = ATTRIBUTE_VALUE_MAPS[attr_key].get(code)
        if code == N_A_CODE or label is None:
            continue
        out[attr_key] = label
    return out


def labels_from_prediction(attributes: dict) -> dict[str, str]:
    """Inference path: predicted probabilities -> state labels (0.5 threshold, size argmax)."""
    out = {}
    probs = [attributes.get(f"Cell_Size_{s}") for s in CELL_SIZE_STATES]
    if all(isinstance(p, (int, float)) for p in probs):
        out["cell_size"] = CELL_SIZE_STATES[max(range(3), key=lambda i: probs[i])]
    for pred_name, attr_key in PRED_ATTR_KEY.items():
        v = attributes.get(pred_name)
        if isinstance(v, (int, float)):
            out[attr_key] = ATTRIBUTE_VALUE_MAPS[attr_key][int(v >= 0.5)]
    return out


def normalise_cell_type(name: str) -> str:
    n = str(name).strip().lower()
    return "none" if n in {"none", "unknown", ""} else n

def build_case_record(store: dict, *, n_images: int, diagnosis: str | None = None) -> dict:
    """Per-patient record in the reference schema. Used for both reference and predicted cells."""
    rec = {
        "metadata_filename_diagnosis": diagnosis,
        "n_images": n_images,
        "cell_counts": dict(store["cell_counts"].most_common()),
        "attribute_counts": {k: dict(store["attribute_counts"][k].most_common()) for k in ATTRIBUTE_KEYS},
        "celltype_attribute_counts": {
            ct: {k: dict(store["celltype_attribute_counts"][ct][k].most_common()) for k in ATTRIBUTE_KEYS}
            for ct in store["celltype_attribute_counts"]
        },
    }

    c_all = rec["cell_counts"]
    c_clin = {k: v for k, v in c_all.items() if k != "none"}
    pct_clin = _percentages_from_counts(c_clin, sum(c_clin.values()))

    return {
        "metadata_filename_diagnosis": diagnosis,
        "n_images": n_images,
        "n_cells_total": sum(c_all.values()),
        "n_cells_identified_wbc": sum(c_clin.values()),
        "cell_counts": c_all,
        "cell_percentages_all": _percentages_from_counts(c_all, sum(c_all.values())),
        "cell_percentages_clinical": pct_clin,
        "attributes": _attribute_summary(rec["attribute_counts"]),
        "report_ready": _build_report_ready(rec, pct_clin),
    }

# ---------------------------------------------------------------------------
# LLM enrichment (identical for reference construction and inference)
# ---------------------------------------------------------------------------
NUMERIC_TOKEN_RE = re.compile(r"(?<![A-Za-z_])(\d+(?:\.\d+)?)")
NUMERIC_WHITELIST: set[str] = {"20", "10", "5", "2", "65"}   # guideline thresholds
# LLM_HIDDEN_KEYS = ("metadata_filename_diagnosis",)     # QC-only; absent at inference

LLM_KEEP_KEYS = (
    "n_images", "n_cells_total", "n_cells_identified_wbc",
    "cell_counts", "cell_percentages_clinical", "report_ready",
)


def llm_view(record: dict) -> dict:
    """The record as the LLM sees it: label removed, and only the fields the
    morphologic interpretation is allowed to draw on."""
    return {k: copy.deepcopy(record[k]) for k in LLM_KEEP_KEYS if k in record}


def build_llm_messages(record: dict, template_md: str) -> list[dict[str, str]]:
    """The single enrichment prompt. Also used to build SFT/GRPO training prompts."""
    user = USER_TEMPLATE.format(
        json_blob=json.dumps(llm_view(record), indent=2),
        report_md=template_md,
    )
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def _flatten_numbers(obj: Any, out: set[str]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _flatten_numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _flatten_numbers(v, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        out.add(str(obj))
        if isinstance(obj, float):
            out.add(f"{obj:.1f}")
            out.add(f"{round(obj, 1)}")
    elif isinstance(obj, str):
        out.update(NUMERIC_TOKEN_RE.findall(obj))


def allowed_numbers(record: dict, template_md: str) -> set[str]:
    allowed = set(NUMERIC_WHITELIST)
    _flatten_numbers(llm_view(record), allowed)
    allowed.update(NUMERIC_TOKEN_RE.findall(template_md))
    return allowed


def find_novel_numbers(text: str, allowed: set[str]) -> list[str]:
    novel, seen = [], set()
    for tok in NUMERIC_TOKEN_RE.findall(text):
        if tok not in allowed and tok not in seen:
            novel.append(tok)
            seen.add(tok)
    return novel


def insert_section(report_md: str, section: str) -> str:
    section = section.rstrip() + "\n"
    for anchor in ("**Diagnostic flags:**", "**Impression:**"):
        idx = report_md.find(anchor)
        if idx != -1:
            line_start = report_md.rfind("\n", 0, idx) + 1
            return report_md[:line_start] + section + "\n" + report_md[line_start:]
    return report_md.rstrip() + "\n\n" + section


def enrich(record: dict, template_md: str,
           complete: Callable[[str, str], str] | None,
           max_retries: int = 1) -> tuple[str, dict]:
    """Add the Morphologic interpretation section. Returns (report_md, info).
    The template is returned unchanged if there is no cohort, no LLM, the model skips,
    or every attempt fails numeric containment."""
    info: dict[str, Any] = {"enriched": False, "attempts": 0, "issues": []}
    rr = record.get("report_ready", {})
    if complete is None:
        info["issues"].append("no LLM backend")
        return template_md, info
    if rr.get("qc", {}).get("n_cells_in_cohort", 0) == 0 or "blast_morphology" not in rr:
        info["issues"].append("no cohort to interpret")
        return template_md, info

    messages = build_llm_messages(record, template_md)
    system, base_user = messages[0]["content"], messages[1]["content"]
    user = base_user
    allowed = allowed_numbers(record, template_md)

    for attempt in range(max_retries + 1):
        info["attempts"] = attempt + 1
        output = (complete(system, user) or "").strip()
        if not output or output.upper().startswith("SKIP"):
            info["issues"].append("model skipped")
            return template_md, info
        novel = find_novel_numbers(output, allowed)
        if not novel:
            info["enriched"] = True
            info["section"] = output
            return insert_section(template_md, output), info
        info["issues"].append(f"attempt {attempt + 1}: novel numbers {novel}")
        user = (base_user + "\n\n---RETRY FEEDBACK---\n"
                f"Your previous output contained numeric value(s) not present in the inputs: "
                f"{', '.join(novel)}. Regenerate using ONLY numbers that appear verbatim in the "
                f"JSON or the report. Do not perform any new calculation.")
    return template_md, info


# ---------------------------------------------------------------------------
# End-to-end protocol: template -> enrichment -> validation
# ---------------------------------------------------------------------------
def render_template(case_id: str, record: dict) -> str:
    try:
        from data_preprocessing.leukemia_report_generator import generate_report
    except ModuleNotFoundError:
        from leukemia_report_generator import generate_report
    return generate_report(case_id, record)


def validate(case_id: str, record: dict, report_md: str):
    try:
        from data_preprocessing.validate_llm_reports import validate_one
    except ModuleNotFoundError:
        from validate_llm_reports import validate_one
    return validate_one(case_id, record, report_md)


def run_report_protocol(case_id: str, record: dict,
                        complete: Callable[[str, str], str] | None) -> tuple[str, dict]:
    template_md = render_template(case_id, record)
    report_md, info = enrich(record, template_md, complete)
    result = validate(case_id, record, report_md)
    info["validation"] = {
        "passed": result.passed,
        "parse_errors": result.parse_errors,
        "consistency_issues": result.consistency_issues,
        "hallucination_issues": result.hallucination_issues,
    }
    return report_md, info