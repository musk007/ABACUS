import os
import pandas as pd




# --------------------------------------
#           cAItomorph
# --------------------------------------
# metadata = pd.read_csv("/l/users/roba.majzoub/cAItomorph/caitomorph_test_set/metadata.csv")
# print(metadata.columns) # 'patient_id', 'diagnosis_coarse', 'diagnosis_fine', 'number_of_cells'
# coarse_diagnosis = set(metadata['diagnosis_coarse'].tolist())
# fine_diagnosis = set(metadata['diagnosis_fine'].tolist())

# print(f"Coarse diagnosis:\n{coarse_diagnosis}") # {'MPN', 'MDS', 'Acute leukemia', 'Plasma cell neoplasm', 'Healthy', 'Reactive changes', 'MDS-MPN', 'Lymphoma'}
# print(f"\n\nFine diagnosis:\n{fine_diagnosis}") # {'MPN', 'MDS', 'Stem cell donor', 'B-cell neoplasm', 'ALL', 'MPN / MDS-RS-T', 'CML', 'T-cell neoplasm', 'ET', 'Reactive changes', 'AL', 'MM', 'PCL', 'CMML', 'MDS / MPN', 'HCL', 'AML', 'PV'}

# ALL_patients = metadata[metadata['diagnosis_fine']=="ALL"]
# AML_patients = metadata[metadata['diagnosis_fine']=="AML"]
# CML_patients = metadata[metadata['diagnosis_fine']=="CML"]

# print(f"There are {len(ALL_patients)} ALL patients")
# print(f"There are {len(AML_patients)} AML patients")
# print(f"There are {len(CML_patients)} CML patients")
# print(f"The total number of patients is {len(CML_patients)+len(AML_patients)+len(CML_patients)} classifiable patients")

# ALL_patients = list(ALL_patients['patient_id'])
# AML_patients = list(AML_patients['patient_id'])
# CML_patients = list(CML_patients['patient_id'])
# patients_dir = "/l/users/roba.majzoub/cAItomorph/caitomorph_test_set/patients"
# al_patients = os.listdir(patients_dir)

# all_lists = [ALL_patients, AML_patients, CML_patients]
# for i, lst in enumerate(all_lists):
#     for p in lst:
#         if p not in al_patients:
#             print(f"Patient {p} is not in the patients directory, with class {i}")


# c1 = pd.read_csv("/home/roba.majzoub/research/AgenticHematology/agentic_hematology/wbc_unified/cv/generated_C1/celltype_manifest.csv"); c1["domain"] = "C1"
# c2 = pd.read_csv("/home/roba.majzoub/research/AgenticHematology/agentic_hematology/wbc_unified/cv/generated_C2/celltype_manifest.csv"); c2["domain"] = "C2"
# pd.concat([c1, c2]).to_csv("/home/roba.majzoub/research/AgenticHematology/agentic_hematology/wbc_unified/cv/generated_C1/celltype_manifest_multicam.csv", index=False)
# pd.concat([c1, c2]).to_csv("/home/roba.majzoub/research/AgenticHematology/agentic_hematology/wbc_unified/cv/generated_C2/celltype_manifest_multicam.csv", index=False)


# # --------------------------------------
# #           AML-MLL
# # --------------------------------------
# metadata = pd.read_excel("/l/users/roba.majzoub/scale/cell-level/helmholtz/AML-Cytomorphology_MLL_Helmholtz.xlsx")
# print(metadata.columns) # 'dataset', 'image', 'class', 'label_type', 'split', 'patient_wise', 'patient', 'caption'

# classes = set(list(metadata['bag_label']))
# print(classes)


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic_hematology.aggregator import aggregate
from agentic_hematology.detection_agent_v2 import (
    EfficientNetAttributeClassifier, TwoStageDetectionAgent, YOLOv11Localizer,
)
from agentic_hematology.leukemia_classifier import HybridClassifier, LearnedClassifier
from agentic_hematology.report_generator import TemplateReportGenerator

YOLO = "/home/roba.majzoub/research/AgenticHematology/agentic_hematology/wbc_unified/cv/runs/detector/train/weights/best.pt"
ATTR = "/home/roba.majzoub/research/AgenticHematology/agentic_hematology/wbc_unified/cv/runs/attribute/train/best_attr.pt"
CLF  = "/home/roba.majzoub/research/AgenticHematology/agentic_hematology/wbc_unified/cv/runs/classifier/effnet/random_forest/leukemia_random_forest.pkl"
IMAGES = sorted(str(p) for p in Path("/home/roba.majzoub/research/AgenticHematology/agentic_hematology/wbc_unified/cv/generated/det_dataset/images/test").glob("15_*.png"))
assert IMAGES, "no images matched the glob"

detector = TwoStageDetectionAgent(
    localizer=YOLOv11Localizer(weights_path=YOLO, device="0", conf_threshold=0.25),
    attribute_classifier=EfficientNetAttributeClassifier(weights_path=ATTR, device="0"),
    attribute_head_name="EfficientNet",
)
det = detector.detect("1", IMAGES)
findings = aggregate(det)
classification = HybridClassifier(LearnedClassifier(CLF)).classify(findings, detection_result=det)

rec = findings.case_record
print("cells:", rec["n_cells_identified_wbc"], "cohort:", rec["report_ready"]["cohort_cell_type"])
print("flags:", rec["report_ready"]["diagnostic_flags"])

r = TemplateReportGenerator().generate(findings, classification)
print("valid:", r.protocol_info["validation"]["passed"])
print(r.protocol_markdown[:300])
print("appendix in report / in body:",
      "## Agentic Diagnosis" in r.markdown, "## Agentic Diagnosis" in r.protocol_markdown)


cd /home/roba.majzoub/research/AgenticHematology/agentic_hematology/wbc_unified/report
python - <<'EOF'
import sys; sys.path.insert(0, "..")
import pyarrow.parquet as pq, statistics as st
from verl_scripts.reward_morphology import compute_score

P = "/home/roba.majzoub/research/AgenticHematology/report_llm/data/verl/grpo_e2e/train.parquet"
rows = pq.read_table(P).to_pylist()
out = [compute_score(r["data_source"], r["reward_model"]["ground_truth"],
                     r["reward_model"]["ground_truth"], r["extra_info"]) for r in rows]
for k in ("num_score", "content_score", "fmt_score", "length_pen"):
    v = [o.get(k) for o in out if o.get(k) is not None]
    if v:
        print(f"{k:14s} mean={st.mean(v):.3f} min={min(v):.3f} max={max(v):.3f}")
    else:
        print(f"{k:14s} MISSING — edit not applied")
print("words:", st.mean([len(r['reward_model']['ground_truth'].split()) for r in rows]))
EOF