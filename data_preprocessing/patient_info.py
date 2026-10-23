import json
import csv
import os
import glob


from report_protocol import (

    ATTRIBUTE_KEYS,
    N_A_CODE,
    YOLO_CLASS_NAMES,
    add_cell,
    build_case_record,
    new_store,
    labels_from_annotation_codes
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "patient_WBC_stats_100X.json")
OUT_CSV_PATH = os.path.join(HERE, "patient_WBC_stats_100X.csv")

DATASET_ROOT = (
    "/l/users/roba.majzoub/Large Leukemia Dataset/"
    "Leukemia_Attr/LeukemiaAttri_Dataset"
)

# Restriced to a single canonical camera domain to prevent sensor multiplication
DOMAINS: list[str] = ["H_100X_C2"]
SPLITS = ["train", "test"]

# ---------------------------------------------------------------------------
# Semantics & Corrected Lookups
# ---------------------------------------------------------------------------
KNOWN_DIAGNOSES = {"ALL", "AML", "APML", "CLL", "CML"}

# ---------------------------------------------------------------------------
# Data Accumulator Storage Elements
# ---------------------------------------------------------------------------
def _new_patient_store() -> dict:
    store = new_store()
    store["metadata_filename_diagnosis"] = None
    store["filenames"] = set()
    return store

def parse_filename(fname: str) -> tuple[str, str]:
    """Parse patient id and diagnosis from 100X or 40X-style filenames."""
    stem = os.path.splitext(os.path.basename(fname))[0]
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Filename {fname} does not conform to expected LLD structure.")
    return parts[0], parts[-1]

# ---------------------------------------------------------------------------
# Pure AttriDet Ingestion Engine (JSON Double-Loading Removed)
# ---------------------------------------------------------------------------
def ingest_yolo_attribute_dir(stores: dict[str, dict], label_dir: str) -> None:
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    paths = sorted(glob.glob(os.path.join(label_dir, "*.txt")))

    for path in paths:
        fname = os.path.basename(path)
        pid, dx = parse_filename(fname)
        stem = os.path.splitext(fname)[0]
        
        if pid not in stores:
            stores[pid] = _new_patient_store()
        if stores[pid]["metadata_filename_diagnosis"] is None:
            stores[pid]["metadata_filename_diagnosis"] = dx
        stores[pid]["filenames"].add(stem)


        with open(path) as f:
            for row in f:
                parts = row.strip().split()
                # Targets 12-column AttriDet structure explicitly
                if not parts or len(parts) < 12:
                    continue
                    
                try:
                    cls = int(parts[0])
                except ValueError:
                    continue
                ct = YOLO_CLASS_NAMES.get(cls, "none")

                codes = []
                for idx in range(len(ATTRIBUTE_KEYS)):
                    try:
                        codes.append(int(parts[5 + idx]))
                    except (ValueError, IndexError):
                        codes.append(N_A_CODE)
                add_cell(stores[pid], ct, labels_from_annotation_codes(codes))

                

def load_patient_data() -> dict:
    stores: dict[str, dict] = {}
    
    for domain in DOMAINS:
        paths = _domain_paths(domain)

        for split in SPLITS:
            ingest_yolo_attribute_dir(stores, paths["yolo_attr"][split])

    return stores



def _domain_paths(domain: str) -> dict:
    root = os.path.join(DATASET_ROOT, domain)
    return {
        "yolo_attr": {s: os.path.join(root, "txt_labels", "AttriDet", s) for s in SPLITS},
    }

def compute_percentages(stores: dict) -> dict:
    ordered = sorted(stores.items(), key=lambda kv: (int(kv[0]) if kv[0].isdigit() else float("inf"), kv[0]))
    return {
        pid: build_case_record(s, n_images=len(s["filenames"]), diagnosis=s["metadata_filename_diagnosis"])
        for pid, s in ordered
    }

def write_csv(percentages: dict, path: str) -> None:
    all_cell_types = sorted({ct for rec in percentages.values() for ct in rec["cell_counts"]})
    header = ["patient_id", "metadata_filename_diagnosis", "n_images", "n_cells_total", "n_cells_identified_wbc"]
    for ct in all_cell_types:
        header += [f"celltype__{ct}__count", f"celltype__{ct}__pct_clinical"]
    header += ["blast_pool_pct_clinical", "qc__low_cell_count"]
    
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for pid, rec in percentages.items():
            rep = rec["report_ready"]
            row = [pid, rec["metadata_filename_diagnosis"], rec["n_images"], rec["n_cells_total"], rec["n_cells_identified_wbc"]]
            for ct in all_cell_types:
                row.append(rec["cell_counts"].get(ct, 0))
                row.append(rec["cell_percentages_clinical"].get(ct, 0.0))
            row += [rep["blast_pool_percentage_of_wbc"], rep["qc"]["low_cell_count_warning"]]
            writer.writerow(row)

if __name__ == "__main__":
    try:
        summary_data = load_patient_data()
        final_percentages = compute_percentages(summary_data)
        with open(OUT_PATH, "w") as out_f:
            json.dump(final_percentages, out_f, indent=2)
        write_csv(final_percentages, OUT_CSV_PATH)
        print(f"[SUCCESS] Pipeline completed safely. Direct aggregation (no overlap deduplication).")
    except Exception as e:
        print(f"[FATAL PIPELINE CRASH] Execution halted: {str(e)}")