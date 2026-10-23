#!/usr/bin/env python3
"""Copy every result file the paper relies on into outputs/paper_results/ and write MANIFEST.md.

    python collect_all_results.py
Files already in paper_results are kept. Missing files are listed in the manifest;
if a file is not at its expected path, the repository is searched for it by name.
"""
from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path("/home/roba.majzoub/research/AgenticHematology")
AH = REPO / "agentic_hematology"
CV = AH / "wbc_unified" / "cv"
OUT = REPO / "outputs" / "paper_results"

# (source, destination subfolder, what it supports)
FILES = [
    (REPO / "outputs/cv_eval_v2/summary.json", "patient_level", "Patient classification, reflection ablation, reports (Sec. 5.2-5.4)"),
    (REPO / "outputs/cv_eval_v2/per_patient.csv", "patient_level", "Per-patient predictions, flags, report scores"),
    (REPO / "outputs/cv_eval_v2/classification.md", "patient_level", "Accuracy, macro-F1, confusion matrix, errors"),
    (REPO / "outputs/cv_eval_v2/classifier_backends.md", "patient_level", "Table 1: RF vs XGBoost vs LightGBM"),
    (REPO / "outputs/cv_eval_v2/reports.md", "reports", "Report-generation metrics (Sec. 5.3)"),
    (REPO / "outputs/cv_eval_zeroshot/summary.json", "reports", "Zero-shot report generator (baseline)"),
    (REPO / "outputs/cv_eval_v2/ablation_reflector_guard.md", "reflection", "Table 3 rows: none/full reflector x guard"),
    (REPO / "outputs/cv_eval_v2/ablation_flag_only.json", "reflection", "Table 3 rows: flag-only reflector"),
    (REPO / "outputs/cv_eval_v2/reflection_agent_results.json", "reflection", "Outcome counts, per-case traces (Table 4)"),
    (REPO / "outputs/router_eval/router_ablation.md", "routing", "Table 5: router ablation"),
    (REPO / "outputs/router_eval/llm_intent_canonical_summary.json", "routing", "Deployed router: full metrics"),
    (REPO / "outputs/router_eval/llm_base_summary.json", "routing", "Free-planning router: full metrics"),
    (REPO / "outputs/router_eval/rule_based_summary.json", "routing", "Rule-based router: full metrics"),
    (REPO / "outputs/router_eval/llm_report_lora_summary.json", "routing", "Free planning with report LoRA"),
    (AH / "output/external_cohort/crop_ensemble_c1c2/summary.json", "external", "cAItomorph, C1+C2 crop ensemble (Sec. 5.6)"),
    (AH / "output/external_cohort/crop_ensemble_c1c2/caitomorph_per_patient.csv", "external", "cAItomorph per-patient p(acute), C1+C2"),
    (AH / "output/external_cohort/crop_ensemble_c1c2/lld_crop_cv_per_patient.csv", "external", "LLD crop-level CV per patient, C1+C2"),
    (AH / "output/external_cohort/crop_ensemble_c2only/summary.json", "external", "cAItomorph, C2-only crop ensemble"),
    (AH / "output/external_cohort/crop_ensemble_c2only/caitomorph_per_patient.csv", "external", "cAItomorph per-patient p(acute), C2 only"),
    (AH / "results/acute_vs_chronic_lld_v2.json", "external", "Table 2: LLD block (ours vs cAItomorph)"),
    (AH / "results/acute_vs_chronic_cait_v2.json", "external", "Table 2: cAItomorph block (ours vs cAItomorph)"),
    (AH / "runs/classifier/camera_comparison.json", "camera", "tab:camera: patient classifier by training/test camera"),
    (CV / "runs/attribute_probe_comparison.json", "camera", "Table 8 / tab:camera: attribute F1 by camera"),
    (CV / "runs/detector/train_folds/detection_per_class/per_class_summary.csv", "perception", "Table 7 / Fig. 1a: detection per class"),
    (CV / "runs/detector/train_folds/detection_per_class/per_class_by_fold.csv", "perception", "Detection per class and fold"),
    (CV / "runs/celltype/dinobloom_C1C2_celltype/pooled_summary.json", "perception", "Cell-type head, 14 classes (pooled)"),
    (CV / "runs/celltype/dinobloom_C1C2_celltype/pooled_per_class.csv", "perception", "Fig. 1a: cell-type F1 per class"),
    (CV / "runs/celltype/dinobloom_C1C2_group/pooled_summary.json", "perception", "Cell-group head, 6 groups (pooled)"),
    (CV / "runs/celltype/dinobloom_C1C2_group/pooled_per_class.csv", "perception", "Fig. 1b: cell-group F1 per group"),
]
# Files written directly into paper_results by earlier collectors
IN_PLACE = [
    ("perception.json", "Perception summary (detection, cell type, attributes)"),
    ("external_threshold_and_camera.json", "LLD-chosen threshold, cAItomorph operating point, camera bootstrap"),
    ("pipeline_acute_vs_chronic.json", "Pipeline acute vs chronic on LLD: 45/46"),
    ("report_generator_sft_vs_zeroshot.json", "Fine-tuned vs zero-shot report generator"),
    ("figures/perception_per_class.pdf", "Figure 1"),
    ("figures/perception_per_class.csv", "Figure 1a values"),
    ("figures/perception_per_group.csv", "Figure 1b values"),
]


def find(src: Path) -> Path | None:
    if src.is_file():
        return src
    hits = [p for p in REPO.rglob(src.name) if "paper_results" not in p.parts]
    return hits[0] if len(hits) == 1 else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Paper results manifest", "", "| File | Supports | Source |", "|---|---|---|"]
    missing = []
    for src, sub, what in FILES:
        found = find(src)
        dest = OUT / sub / (f"{src.parent.name}__{src.name}"
        if any(k in str(src) for k in ("crop_ensemble", "dinobloom_C1C2", "cv_eval_zeroshot")) else src.name)
        if found is None:
            missing.append((src, what))
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found, dest)
        note = "" if found == src else " (found by name search)"
        lines.append(f"| `{dest.relative_to(OUT)}` | {what} | `{found}`{note} |")
    for rel, what in IN_PLACE:
        if (OUT / rel).is_file():
            lines.append(f"| `{rel}` | {what} | written directly |")
        else:
            missing.append((OUT / rel, what))
    if missing:
        lines += ["", "## Missing", ""] + [f"- `{p}`: {w}" for p, w in missing]
    (OUT / "MANIFEST.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()