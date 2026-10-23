"""
leukemia_report_generator.py
============================
Template-based hematology report generator that consumes the structured
`report_ready` block (and surrounding statistics) produced by an upstream
multi-image cell-detection + attribute-extraction pipeline, and emits a
short, fixed-format diagnostic report.

Design choices
--------------
1. Reports are generated from the *findings* (blast %, dominant cell type,
   diagnostic flags, blast morphology), NOT from the metadata diagnosis
   label. The label is treated as ground truth for QC only.
2. Routing is rule-based and deterministic so the output is reproducible
   and auditable (no stochastic LLM in the loop).
3. Supported impressions: ALL, AML (incl. AML with monocytic differentiation),
   APML, CML, CLL, and an "unclassified abnormal" fallback.
4. Each report section maps 1:1 to a section of the input JSON, so the
   structure can later be swapped for an LLM prompt template if desired.

Usage
-----
    python leukemia_report_generator.py cases.json
    python leukemia_report_generator.py cases.json --case-id 12
    python leukemia_report_generator.py cases.json --out reports/

The input JSON is expected to be either:
  (a) a dict keyed by case id, where each value contains the case payload
      shown in the examples, OR
  (b) a single case payload at the top level.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Constants & lookups
# ---------------------------------------------------------------------------

BLAST_LIKE_CELL_TYPES = {
    "lymphoblast",
    "myeloblast",
    "monoblast",
    "abnormal promyelocyte",
}

# Maps the upstream dominant-cell label to a clinically meaningful blast
# lineage descriptor used in the report's impression line.
BLAST_LINEAGE_DESCRIPTION = {
    "lymphoblast": "lymphoid (lymphoblasts)",
    "myeloblast": "myeloid (myeloblasts)",
    "monoblast": "monocytic (monoblasts)",
    "abnormal promyelocyte": "promyelocytic (abnormal promyelocytes)",
}

# Friendly print names for cell types in the differential table.
CELL_PRINT_NAME = {
    "lymphoblast": "Lymphoblasts",
    "myeloblast": "Myeloblasts",
    "monoblast": "Monoblasts",
    "abnormal promyelocyte": "Abnormal promyelocytes",
    "neutrophil": "Neutrophils",
    "eosinophil": "Eosinophils",
    "basophil": "Basophils",
    "monocyte": "Monocytes",
    "lymphocyte": "Lymphocytes",
    "atypical lymphocyte": "Atypical lymphocytes",
    "myelocyte": "Myelocytes",
    "metamyelocyte": "Metamyelocytes",
    "promonocyte": "Promonocytes",
}

COHORT_DESCRIBABLE_CELL_TYPES = {
    "lymphoblast", "myeloblast", "monoblast", "abnormal promyelocyte",
    "promonocyte", "atypical lymphocyte", "lymphocyte",
}

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Impression:
    """Final routed impression with supporting rationale."""
    primary: str               # e.g. "Acute Lymphoblastic Leukemia (ALL)"
    rationale: str             # one-sentence why
    differential: list[str]    # short list of alternatives
    recommended_workup: list[str]


# ---------------------------------------------------------------------------
# Routing logic — pure functions on the `report_ready` block
# ---------------------------------------------------------------------------

def classify_case(
    report_ready: dict[str, Any],
    clinical_percentages: dict[str, float] | None = None,
) -> Impression:
    """
    Route a case to an impression based on diagnostic flags, blast burden,
    and dominant cell type. The routing tree mirrors WHO/ICC top-of-the-tree
    decisions and is intentionally conservative.
    """
    flags = report_ready["diagnostic_flags"]
    dominant = report_ready["dominant_cell_type"]
    blast_pct = report_ready["blast_pool_percentage_of_wbc"]
    clinical = clinical_percentages or {}
    lymphoblast_pct = float(clinical.get("lymphoblast", 0.0))
    myeloblast_pct = float(clinical.get("myeloblast", 0.0))
    monoblast_pct = float(clinical.get("monoblast", 0.0))
    promonocyte_pct = float(clinical.get("promonocyte", 0.0))
    myeloid_blast_like_pct = report_ready.get(
        "myeloid_blastlike_percentage_of_wbc",
        myeloblast_pct + monoblast_pct + promonocyte_pct)
    abnormal_promyelocyte_pct = float(clinical.get("abnormal promyelocyte", 0.0))

    # 1. APML — abnormal promyelocytes are a hard, specific finding.
    if flags.get("abnormal_promyelocytes_present"):
        return Impression(
            primary="Findings consistent with an acute leukaemia showing abnormal promyelocyte morphology",
            rationale=(
                f"Abnormal promyelocytes comprise {abnormal_promyelocyte_pct:.1f}% of WBCs, "
                f"with a blast-equivalent burden of {blast_pct:.1f}%, "
                f"meeting criteria for acute leukaemia."
            ),
            differential=[
                "Acute promyelocytic leukaemia — requires urgent PML::RARA confirmation.",
                "AML, not otherwise specified (less likely given promyelocyte morphology).",
            ],
            recommended_workup=[
                "URGENT: PML::RARA fusion testing (RT-PCR or FISH) — APL is a "
                "medical emergency requiring immediate ATRA initiation on clinical suspicion.",
                "Flow cytometric immunophenotyping (typically CD33+, CD13+, "
                "CD117+/-, HLA-DR negative, CD34 negative).",
                "Coagulation panel (PT/PTT, fibrinogen, D-dimer) to assess for DIC.",
                "Bone marrow aspirate and biopsy for confirmation.",
                "Cytogenetics for t(15;17) and screening for variant RARA translocations.",
            ],
        )

    # 2. Acute leukaemia with high blast burden (>= 20% WHO threshold).
    if flags.get("blast_threshold_met"):
        if dominant == "lymphoblast":
            return Impression(
                primary="Findings consistent with an acute leukaemia of lymphoblastic morphology",
                rationale=(
                    f"Lymphoblasts comprise {lymphoblast_pct:.1f}% of WBCs "
                    f"(blast-equivalent burden of {blast_pct:.1f}%), "
                    f"exceeding the 20% blast threshold for acute leukaemia."
                ),
                differential=[
                    "B- or T-lymphoblastic leukaemia/lymphoma.",
                    "Mixed-phenotype acute leukaemia (to be excluded by immunophenotyping).",
                ],
                recommended_workup=[
                    "Flow cytometric immunophenotyping for lineage assignment "
                    "(B-ALL: CD19, CD22, CD79a, CD10, TdT; T-ALL: cytoCD3, CD7, TdT).",
                    "Bone marrow aspirate and trephine biopsy.",
                    "Cytogenetics and FISH for recurrent ALL abnormalities "
                    "(BCR::ABL1, KMT2A rearrangements, ETV6::RUNX1, TCF3::PBX1, hypodiploidy).",
                    "Molecular studies including BCR::ABL1-like signature where available.",
                    "CSF examination to assess CNS involvement.",
                ],
            )
        if dominant == "monoblast":
            return Impression(
                primary="Findings consistent with an acute leukaemia showing monocytic differentiation",
                rationale=(
                    f"Monoblasts comprise {monoblast_pct:.1f}% of WBCs "
                    f"(blast-equivalent burden of {blast_pct:.1f}%), "
                    f"consistent with a monocytic-lineage acute leukaemia."
                ),
                differential=[
                    "Acute myelomonocytic leukaemia.",
                    "AML with KMT2A rearrangement (frequently monocytic).",
                    "Blastic plasmacytoid dendritic cell neoplasm (to be excluded by immunophenotyping).",
                ],
                recommended_workup=[
                    "Flow cytometric immunophenotyping (CD33, CD64, CD11b, CD14, "
                    "lysozyme; CD34 often negative in monoblastic AML).",
                    "Cytochemistry: non-specific esterase (NSE) positive, MPO often weak/negative.",
                    "Bone marrow aspirate and trephine biopsy.",
                    "Cytogenetics and FISH including KMT2A (11q23) rearrangements.",
                    "Molecular panel including NPM1, FLT3-ITD/TKD, and AML fusion screen.",
                ],
            )
        if dominant == "myeloblast":
            return Impression(
                primary="Findings consistent with an acute leukaemia of myeloblastic morphology",
                rationale=(
                    f"Myeloblasts comprise {myeloblast_pct:.1f}% of WBCs "
                    f"(blast-equivalent burden of {blast_pct:.1f}%), "
                    f"exceeding the 20% blast threshold for acute leukaemia."
                ),
                differential=[
                    "AML with recurrent genetic abnormalities (per WHO/ICC).",
                    "Acute myeloid leukaemia.",
                    "Mixed-phenotype acute leukaemia (to be excluded by immunophenotyping).",
                ],
                recommended_workup=[
                    "Flow cytometric immunophenotyping (CD13, CD33, CD117, MPO, HLA-DR).",
                    "Bone marrow aspirate and trephine biopsy.",
                    "Cytogenetics and FISH for recurrent AML abnormalities.",
                    "Molecular panel: NPM1, FLT3-ITD/TKD, CEBPA, RUNX1, TP53, and AML fusion screen.",
                ],
            )
        # If mature cells dominate despite >=20% blasts, infer lineage from the
        # differential distribution so acute leukaemia is not dropped to fallback.
        if lymphoblast_pct >= 20.0 and lymphoblast_pct >= myeloid_blast_like_pct:
            return Impression(
                primary="Findings consistent with an acute leukaemia of lymphoblastic morphology",
                rationale=(
                    f"Lymphoblast burden is {lymphoblast_pct:.1f}% of informative WBCs, "
                    f"with total blast-equivalent burden {blast_pct:.1f}%, meeting criteria "
                    f"for acute leukaemia despite mature-cell predominance."
                ),
                differential=[
                    "B- or T-lymphoblastic leukaemia/lymphoma.",
                    "Mixed-phenotype acute leukaemia (requires immunophenotyping).",
                ],
                recommended_workup=[
                    "Flow cytometric immunophenotyping for lineage assignment.",
                    "Bone marrow aspirate and trephine biopsy.",
                    "Cytogenetics and FISH for recurrent ALL abnormalities.",
                    "Molecular profiling aligned to ALL risk stratification.",
                ],
            )
        if myeloid_blast_like_pct >= 20.0:
            return Impression(
                primary="Findings consistent with an acute leukaemia of myeloblastic morphology",
                rationale=(
                    f"Myeloid blast-like burden is {myeloid_blast_like_pct:.1f}% of informative WBCs "
                    f"(overall blast-equivalent burden {blast_pct:.1f}%), supporting AML even when "
                    f"mature myeloid cells are also prominent."
                ),
                differential=[
                    "AML with recurrent genetic abnormalities (per WHO/ICC).",
                    "Acute myeloid leukaemia.",
                    "Mixed-phenotype acute leukaemia (requires immunophenotyping).",
                ],
                recommended_workup=[
                    "Flow cytometric immunophenotyping (myeloid and monocytic markers).",
                    "Bone marrow aspirate and trephine biopsy.",
                    "Cytogenetics and FISH for recurrent AML abnormalities.",
                    "Molecular panel including NPM1, FLT3, CEBPA, RUNX1, and TP53.",
                ],
            )

    # 2b. Borderline acute pattern under sparse annotation (10% to <20% blasts).
    if 10.0 <= blast_pct < 20.0 and (
        dominant in {"lymphoblast", "myeloblast", "monoblast", "promonocyte"}
        or lymphoblast_pct >= 10.0
        or myeloid_blast_like_pct >= 10.0
    ):
        if dominant == "lymphoblast" or lymphoblast_pct >= myeloid_blast_like_pct:
            primary = "Findings suspicious for an early acute leukaemia of lymphoblastic morphology"
        else:
            primary = "Findings suspicious for an early acute leukaemia of myeloid morphology"
        return Impression(
            primary=primary,
            rationale=(
                f"The blast-equivalent burden is {blast_pct:.1f}% of WBCs. "
                f"While below the classic 20% acute threshold, this remains a "
                f"highly abnormal circulating blast population and is suspicious "
                f"for an incipient acute leukaemia pattern."
            ),
            differential=[
                "Incipient acute leukaemia phase.",
                "Myelodysplastic syndrome with excess blasts.",
                "Reactive leukemoid pattern with blast-like cells (less likely).",
            ],
            recommended_workup=[
                "Urgent bone marrow aspirate and trephine biopsy to establish marrow blast percentage.",
                "Flow cytometric immunophenotyping for definitive lineage assignment.",
                "Cytogenetics and FISH panel aligned to suspected lineage.",
                "Molecular testing for recurrent ALL/AML abnormalities.",
            ],
        )

    # 2c. Monocytic-skewed suspicious pattern under sparse annotation.
    if (
        flags.get("blasts_present")
        and (
            dominant == "promonocyte"
            or promonocyte_pct >= 15.0
            or (flags.get("monocytosis_present") and (promonocyte_pct >= 10.0 or myeloblast_pct >= 5.0))
        )
    ):
        return Impression(
            primary="Findings consistent with an acute leukaemia showing monocytic differentiation",
            rationale=(
                f"Monocytic-lineage predominance (promonocytes {promonocyte_pct:.1f}%) "
                f"with a blast-equivalent burden of {blast_pct:.1f}% is suspicious for a monocytic AML pattern "
                f"under sparse annotation conditions."
            ),
            differential=[
                "Acute monoblastic/monocytic leukaemia.",
                "Acute myelomonocytic leukaemia.",
                "Reactive monocytosis with immature forms (less likely).",
            ],
            recommended_workup=[
                "Flow cytometric immunophenotyping (CD64, CD14, CD11b, lysozyme, CD33).",
                "Bone marrow aspirate and trephine biopsy.",
                "Cytogenetics and targeted molecular AML panel.",
            ],
        )

    # 3. CML pattern: left-shifted myeloid series + basophilia, blasts present but
    #    below threshold.
    if flags.get("left_shifted_myeloid") and not flags.get("blast_threshold_met"):
        # Sparse annotations can undercall basophilia; allow a myeloid-dominant
        # left-shifted profile with circulating blasts to still trigger CML suspicion.
        is_cml_profile = flags.get("basophilia_present") or (
            dominant in {"neutrophil", "myelocyte"} and blast_pct > 0
        )
        if not is_cml_profile:
            is_cml_profile = dominant in {"metamyelocyte"} and blast_pct > 0
        basophil_pct = float(clinical.get("basophil", 0.0))
        basophil_note = ""
        if basophil_pct == 0.0:
            basophil_note = (
                " Absence of identifiable basophils is atypical for classic CML and raises "
                "consideration of alternative reactive processes such as leukemoid reaction "
                "or severe infection/sepsis."
            )
        if is_cml_profile and not flags.get("monocytosis_present"):
            return Impression(
                primary="Findings consistent with a chronic myeloproliferative process with left-shifted granulopoiesis",
                rationale=(
                    f"Left-shifted granulocytic series with the full spectrum of "
                    f"maturation (myelocytes, metamyelocytes, neutrophils), "
                    f"and a blast-equivalent burden of {blast_pct:.1f}% "
                    f"(below the 20% threshold for blast phase) is compatible with a "
                    f"chronic myeloproliferative pattern.{basophil_note}"
                ),
                differential=[
                    "Chronic myeloid leukaemia — requires BCR::ABL1 confirmation.",
                    "Leukemoid reaction or severe infection/sepsis, particularly if basophils are absent.",
                    "Other MPN (PV, ET, primary myelofibrosis).",
                    "Atypical CML, BCR::ABL1 negative.",
                    "CML in accelerated or blast phase (to be excluded by blast count and bone marrow morphology).",
                ],
                recommended_workup=[
                    "BCR::ABL1 testing by RT-PCR (qualitative and quantitative) — diagnostic.",
                    "FISH for t(9;22) / Philadelphia chromosome.",
                    "Bone marrow aspirate and biopsy with cytogenetics.",
                    "Baseline transcript level for IS-standardized monitoring during TKI therapy.",
                    "Sokal / EUTOS / ELTS risk score calculation at diagnosis.",
                ],
            )

    # 4. CLL pattern: atypical lymphocytes present, no blast threshold,
    #    no left shift.
    if flags.get("atypical_lymphocytes_present") and not flags.get("blast_threshold_met"):
        return Impression(
            primary="Findings consistent with a mature lymphoid proliferation",
            rationale=(
                f"The smear is dominated by atypical mature lymphocytes "
                f"({report_ready['dominant_cell_pct']:.1f}% of informative WBCs) "
                f"and no significant blast population."
            ),
            differential=[
                "Chronic lymphocytic leukaemia / small lymphocytic lymphoma.",
                "Mantle cell lymphoma in leukemic phase (to be excluded by immunophenotyping; CLL shows a CD5+/CD23+/cyclin D1- profile).",
                "Prolymphocytic leukemia (>55% prolymphocytes).",
                "Reactive lymphocytosis (e.g. viral) — typically polyclonal.",
            ],
            recommended_workup=[
                "Flow cytometric immunophenotyping (CLL panel: CD5, CD19, CD20, "
                "CD23, CD79b, FMC7, kappa/lambda light chain restriction).",
                "Matutes / CLL score calculation.",
                "FISH panel: del(13q), del(11q), del(17p)/TP53, trisomy 12.",
                "IGHV mutation status and TP53 sequencing (prognostic and predictive).",
                "Beta-2 microglobulin and standard staging workup (Rai/Binet).",
            ],
        )

    # 5. Fallback: abnormal smear that does not match a specific pattern.
    return Impression(
        primary="Abnormal peripheral blood smear, unclassified",
        rationale=(
            f"The smear shows abnormal findings (dominant cell type: "
            f"{dominant}, {report_ready['dominant_cell_pct']:.1f}% of WBCs; "
            f"blast-equivalent burden of {blast_pct:.1f}%) that do not meet criteria for "
            f"a specific WHO/ICC entity on morphology alone."
        ),
        differential=[
            "Further classification pending immunophenotyping and molecular studies.",
        ],
        recommended_workup=[
            "Flow cytometric immunophenotyping.",
            "Bone marrow aspirate and trephine biopsy.",
            "Cytogenetics, FISH, and a broad myeloid/lymphoid molecular panel.",
            "Correlation with clinical findings and CBC indices.",
        ],
    )


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _fmt_pct(x: float) -> str:
    return f"{x:.1f}%"


def build_specimen_line(case: dict[str, Any]) -> str:
    qc = case["report_ready"]["qc"]
    return (
        f"**Specimen:** Peripheral blood smear, {qc['n_fields_of_view']} fields of view, "
        f"{qc['n_identified_wbc']} of {qc['n_annotated_cells']} annotated objects "
        f"classified as informative WBCs ({qc['n_artifacts']} artefacts excluded)."
    )


def build_differential_table(case: dict[str, Any]) -> str:
    """Markdown table of the clinical (artefact-excluded) differential."""
    diff = case["cell_percentages_clinical"]
    # Sort by descending percentage for readability.
    rows = sorted(diff.items(), key=lambda kv: kv[1], reverse=True)
    lines = ["| Cell type | % of informative WBCs |", "|---|---|"]
    for cell, pct in rows:
        name = CELL_PRINT_NAME.get(cell, cell.capitalize())
        lines.append(f"| {name} | {_fmt_pct(pct)} |")
    return "\n".join(lines)


def build_blast_morphology_paragraph(case: dict[str, Any]) -> str:
    """
    Render the per-attribute dominance into a single readable sentence.
    Only emitted if a cohort of blast-like cells exists.
    """
    rr = case["report_ready"]
    cohort_n = rr["qc"]["n_cells_in_cohort"]
    cohort_type = rr.get("cohort_cell_type") or rr["dominant_cell_type"]
    if cohort_type not in COHORT_DESCRIBABLE_CELL_TYPES or cohort_n == 0:
        return ""
    cohort_label = CELL_PRINT_NAME.get(cohort_type, cohort_type.capitalize() + "s").lower()

    morph = rr["blast_morphology"]

    def d(key: str) -> tuple[str, float]:
        return morph[key]["dominant"], morph[key]["dominance_pct"]

    size, size_pct = d("cell_size")
    chrom, chrom_pct = d("nuclear_chromatio")
    shape, shape_pct = d("nuclear_shape")
    nuc, nuc_pct = d("nucleolus")
    cyto, cyto_pct = d("cytoplasm")
    baso, baso_pct = d("cytoplasmic_basophilia")
    vac, vac_pct = d("cytoplasmic_vacuoles")

    # cohort_label = CELL_PRINT_NAME.get(cohort_type, cohort_type).lower()

    return (
        f"**Cohort morphology (n = {cohort_n} {cohort_label}):** "
        f"predominantly {size} cells ({_fmt_pct(size_pct)}) with "
        f"{cyto} ({_fmt_pct(cyto_pct)}), {baso}ly basophilic ({_fmt_pct(baso_pct)}) "
        f"cytoplasm and {vac} cytoplasmic vacuolation ({_fmt_pct(vac_pct)}); "
        f"nuclei show {chrom} chromatin ({_fmt_pct(chrom_pct)}), "
        f"{shape} contours ({_fmt_pct(shape_pct)}), and {nuc} nucleoli "
        f"({_fmt_pct(nuc_pct)})."
    )


def build_flags_line(case: dict[str, Any]) -> str:
    flags = case["report_ready"]["diagnostic_flags"]
    positive = [k.replace("_", " ") for k, v in flags.items() if v]
    if not positive:
        return "**Diagnostic flags:** None triggered."
    return "**Diagnostic flags:** " + "; ".join(positive) + "."

ALERTS_RENDERED_INLINE = {"CML_PATTERN_NO_BASOPHILIA"}   # already stated in the CML rationale


def build_alerts_line(case: dict[str, Any]) -> str:
    alerts = case["report_ready"]["qc"].get("differential_alerts", [])
    items = [a for a in alerts if a.get("code") not in ALERTS_RENDERED_INLINE]
    if not items:
        return ""
    return "**Review alerts:** " + " ".join(
        f"{a['message']} {a.get('recommendation', '')}".strip() for a in items
    )


def build_qc_line(case: dict[str, Any]) -> str:
    qc = case["report_ready"]["qc"]
    warnings = []
    if qc.get("low_cell_count_warning"):
        warnings.append("LOW CELL COUNT")
    if qc.get("sparse_annotation_skew_warning"):
        warnings.append("SPARSE ANNOTATION SKEW")
    artefact_pct = (qc["n_artifacts"] / qc["n_annotated_cells"]) * 100 if qc["n_annotated_cells"] else 0.0
    base = (
        f"**QC:** {qc['n_fields_of_view']} FOVs; "
        f"{qc['n_identified_wbc']}/{qc['n_annotated_cells']} cells classifiable "
        f"({artefact_pct:.1f}% artefact); "
        f"cohort cell count = {qc['n_cells_in_cohort']}."
    )
    if warnings:
        base += "  **WARNINGS:** " + ", ".join(warnings) + "."
    return base


def build_impression_block(imp: Impression) -> str:
    workup = "\n".join(f"- {w}" for w in imp.recommended_workup)
    diff = "\n".join(f"- {d}" for d in imp.differential)
    return (
        f"**Impression:** {imp.primary}. Definitive classification requires "
        f"clinical correlation and ancillary studies.\n\n"
        f"{imp.rationale}\n\n"
        f"**Differential considerations:**\n{diff}\n\n"
        f"**Recommended workup:**\n{workup}"
    )


# ---------------------------------------------------------------------------
# Top-level report assembly
# ---------------------------------------------------------------------------

REPORT_TEMPLATE = """# Hematology Report — Case {case_id}

{specimen_line}

**Differential (clinical denominator):**

{differential_table}

{blast_morphology_paragraph}

{flags_line}

{alerts_line}

{impression_block}

{qc_line}

{scope_line}

---
*Automated multi-image peripheral blood smear analysis. Findings are intended to support — not replace — review by a board-certified hematopathologist.*
"""

def build_scope_line(case: dict[str, Any]) -> str:
    return (
        "**Not assessed:** erythrocyte and platelet morphology, dysplastic features, "
        "smear staining quality, and whether fields originate from the optimal "
        "monolayer zone. Findings are based on leukocyte morphology only."
    )


def generate_report(case_id: str, case: dict[str, Any]) -> str:
    impression = classify_case(
        case["report_ready"],
        case.get("cell_percentages_clinical", {}),
    )

    sections = {
        "case_id": case_id,
        "specimen_line": build_specimen_line(case),
        "differential_table": build_differential_table(case),
        "blast_morphology_paragraph": build_blast_morphology_paragraph(case),
        "flags_line": build_flags_line(case),
        "impression_block": build_impression_block(impression),
        "qc_line": build_qc_line(case),
        "scope_line": build_scope_line(case),
        "alerts_line": build_alerts_line(case),
    }

    report = REPORT_TEMPLATE.format(**sections)
    # Collapse any blank section gaps from optional paragraphs.
    return "\n".join(line for i, line in enumerate(report.splitlines())
                     if not (line == "" and i + 1 < len(report.splitlines())
                             and report.splitlines()[i + 1] == ""))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_cases(path: str) -> dict[str, dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    # If the top-level dict already looks like a single case payload,
    # wrap it under a synthetic id.
    if "report_ready" in data:
        return {"single": data}
    return data


def main() -> int:
    p = argparse.ArgumentParser(description="Generate hematology reports from structured case JSON.")
    p.add_argument("input_json", help="Path to the input JSON file.")
    p.add_argument("--case-id", help="Generate a report for a single case id only.")
    p.add_argument("--out", help="Directory to write per-case .md reports. If omitted, prints to stdout.")
    args = p.parse_args()

    cases = _load_cases(args.input_json)

    if args.case_id:
        if args.case_id not in cases:
            print(f"ERROR: case id '{args.case_id}' not found. Available: {sorted(cases.keys())}",
                  file=sys.stderr)
            return 2
        selected = {args.case_id: cases[args.case_id]}
    else:
        selected = cases

    if args.out:
        os.makedirs(args.out, exist_ok=True)

    for cid, case in selected.items():
        report = generate_report(cid, case)
        if args.out:
            out_path = os.path.join(args.out, f"case_{cid}_report.md")
            with open(out_path, "w") as f:
                f.write(report)
            print(f"Wrote {out_path}")
        else:
            print(report)
            print("\n" + "=" * 78 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
