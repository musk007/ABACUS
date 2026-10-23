# ABACUS — Abstention-Based Aggregation of Cellular Units for Subtyping

Traceable patient-level leukemia reporting from peripheral blood smears.

> Paper: *ABACUS: Traceable Patient-Level Leukemia Reporting from Peripheral
> Blood Smears*, ISBI 2027.

![ABACUS pipeline: cells are detected and typed per tile, aggregated
deterministically into a patient-level evidence record, classified into a
leukemia subtype, and written up by a fine-tuned VLM whose numeric tokens are
constrained to the evidence record.](pipeline_overall.png)

Every number in a generated report must already appear in the evidence record,
the deterministic template, or a fixed set of guideline thresholds; sections
that violate containment are regenerated once, then dropped.

> **Note on `pipeline/run.sh`.** The repository contains an older end-to-end
> driver (`pipeline/run.sh` and the `pipeline/sbatch_*.sh` wrappers) that trains
> the **EfficientNet-B0** attribute head and fine-tunes **Qwen3.5-2B** with an
> SFT **and** a GRPO stage. That is *not* the configuration reported in the
> paper. The commands below are the per-component invocations for the reported
> configuration: DinoBloom-L probe, SFT only. See §9.

---

## Layout

```
agentic_hematology/              orchestrator, agents, validators, eval scripts
├── orchestrator.py  pipeline.py  agent_controller.py  aggregator.py
├── leukemia_classifier.py  leukemia_features.py  tabular_classifier.py
├── train_leukemia_from_{dinobloom,efficientnet,stats}.py
├── eval_{caitomorph,router,cv,detection_patient}.py
├── eval_leukemia_classifiers_{dino,eff}.py
├── check_numeric_containment.py  reflection_evaluation_results.py
├── pool_celltype_folds.py  detection_perClass_folds.py  compare_attr_probes.py
└── wbc_unified/
    ├── cv/                     perception: detector, probes, cell-type head
    │   ├── train_{detector,attributes,cellType}.py
    │   ├── train_dinobloom_attributes_torch.py
    │   ├── data/               prepare_dataset.py  splits.py  merge_manifests.py
    │   ├── scripts/            pipeline_eval{,_dinobloom}.py
    │   └── utils/  models/  configs/
    ├── report/                 reference reports + report generator
    │   ├── scripts/            01–13 numbered stages
    │   └── src/
    ├── verl_scripts/           SFT/GRPO training, env.sh, export_lora.sh
    └── pipeline/               Slurm wrappers, legacy end-to-end driver
```

`cv/` scripts insert the repository root on `sys.path`, so they import
`agentic_hematology.*` directly. Run them from `wbc_unified/cv`, the
`report/scripts/` stages from `wbc_unified/report`, and everything else from
`agentic_hematology/`.

---

## 1. Installation

Conda env `SLA_Det`, Python 3.10, CUDA 11.8:

```bash
cd wbc_unified
CREATE_ENV=1 bash pipeline/install_env.sh all     # cv + verl/vLLM
bash pipeline/install_env.sh verify               # import check
```

`install_env.sh stage1` installs `cv/requirements.txt` only; `install_env.sh
llm` runs `verl_scripts/install.sh`. The LLM stage sources
`verl_scripts/env.sh`, which must exist before any verl command.

### verl

verl is not vendored. Clone it at the pinned commit and apply the patch:

```bash
git clone https://github.com/volcengine/verl.git third_party/verl
cd third_party/verl
git fetch --depth 1 origin cbd7f9f && git checkout cbd7f9f
git apply ../../verl_halo.patch
pip install -e .
```

The patch guards the `DTensor` / `DTensorSpec` imports in
`verl/utils/fsdp_utils.py`, which are unconditional on this commit and fail on
torch 2.4–2.5. On torch ≥ 2.6 the patch is unnecessary.

---

## 2. Data preparation

LLD (LeukemiaAttri). Run the three steps in order — `--export-folds` reads the
frozen fold assignment, so the split must exist first.

**1. Freeze the patient-grouped 5-fold split.** Every patient is held out
exactly once; folds are stratified by diagnosis and greedily balanced by
annotated cell count. Written once to
`wbc_unified/cv/data/configs/cv_folds.json` and read by every module, so the
folds are identical across the detector, the probes and the classifier.

```bash
python wbc_unified/cv/data/splits.py --data-root /path/to/LLD_organized/
```

**2. Build the detection dataset and the cell manifests.**

```bash
python wbc_unified/cv/data/prepare_dataset.py \
    --data-root /path/to/LLD_organized/ \
    --out wbc_unified/cv/generated \
    --strict --image-mode copy
```

Writes `generated/det_dataset/` (5-column YOLO labels),
`generated/attr_manifest.csv` (annotated cells only, for the attribute probe)
and `generated/celltype_manifest.csv` (all cells including the class-0 `None`
artefacts, which the cell-type head must be able to predict). Also writes
`wbc_unified/cv/configs/dataset.yaml`, which every downstream script reads to
locate the manifests.

**3. Export the per-fold detection datasets.** Re-run with `--export-folds` to
materialise `generated/det_dataset_fold{k}/` and
`wbc_unified/cv/configs/dataset_fold{k}.yaml`:

```bash
python wbc_unified/cv/data/prepare_dataset.py \
    --data-root /path/to/LLD_organized/ \
    --out wbc_unified/cv/generated \
    --strict --image-mode copy --export-folds
```

### Two cameras

`merge_manifests.py` merges the `C1` and `C2` manifests into one, adding a
`domain` column. Patient IDs are shared across cameras, so fold filtering by
patient keeps both views of a held-out patient together.

```bash
# prepare each camera separately, then:
python merge_manifests.py --c1 generated_C1 --c2 generated_C2 --out generated_C1C2
```

> **Resolve before release.** `cv/` contains `generated/`, `generated_C1_f/`,
> `generated_C2_f/` **and** `generated_C1C2/`, and `collect_perception.py`
> reads `runs/celltype/dinobloom_C1C2_{celltype,group}/pooled_summary.json`
> plus an attribute comparison keyed by *train camera × test camera*. The
> two-camera path was therefore run, and this merge step is part of the
> reported pipeline rather than optional. The paper describes the data as the
> `H_100X_C2` subset. Reconcile the two: either the paper's dataset
> description is wrong, or `collect_perception.py` is not the script behind the
> perception table.

---

## 3. Reference report construction

```bash
cd report
python scripts/01_aggregate_gt.py          # annotations -> evidence record
python scripts/02_aggregate_pred.py        # predictions -> evidence record
python scripts/04_generate_reports.py      # record -> deterministic template
python scripts/07_generate_reports_llm.py  # + morphological interpretation
python scripts/08_eval_llm_reports.py      # containment validation
```

Released reference reports: `reference_reports/`.

---

## 4. Training (per fold)

`$K` is the fold index, 0–4. No module uses early stopping or checkpoint
selection: every run trains for its full epoch budget and the final epoch is
what gets saved. The detector's `--patience` is inert unless `--val` is passed,
and the probes save only the last epoch by design — in CV the weights are
throwaway and the per-epoch curve is what is recorded.

### Detector

`train_detector.py` has no `--fold` flag — the fold is selected by the config
file written in §2 step 3.

```bash
cd wbc_unified/cv
python train_detector.py \
    --config configs/dataset_fold${K}.yaml \
    --model /path/to/yolo11m.pt \
    --epochs 100 --imgsz 640 --batch 16 \
    --device 0 --workers 8 \
    --project runs/detector --name fold${K} \
    --val --plots
```

Best weights land at `runs/detector/fold${K}/weights/best.pt`. Pass a local
path to `--model`: Ultralytics' auto-download writes a corrupt `.pt` on compute
nodes without egress, and `pipeline/lib_stage1.sh` checks the file is a valid
zip for exactly that reason. `--patience` only takes effect with `--val`.

### Attribute probe — DinoBloom-L (paper configuration)

The backbone is frozen, so crop embeddings are computed once and cached to
`runs/attribute_dinobloom/embeddings_l_224.pt`; the five folds then cost
seconds of MLP training each. The first fold builds the cache.

```bash
for k in 0 1 2 3 4; do
  python train_dinobloom_attributes_torch.py \
      --fold $k --epochs 40 --batch 64 --lr 1e-3 \
      --dinobloom-variant l \
      --dinobloom-weights /path/to/DinoBloom-L.pth \
      --manifest generated/attr_manifest.csv \
      --project runs/attribute_dinobloom \
      --device 0 --workers 2
done
```

A `--fold K` run writes `runs/attribute_dinobloom/cv_fold${K}/fold${K}_attr_dinobloom.pt`
plus `history_fold${K}.json`. Omitting `--fold` does the final full-data run
and writes `best_attr_dinobloom.pt` instead — **the per-fold and final
checkpoint names differ**, which matters when wiring up inference (§6).

### Attribute probe — EfficientNet-B0 (ablation)

Same manifest, same folds, same masked-BCE loss and 3-class `Cell_Size` head,
fine-tuned end to end rather than probed.

```bash
for k in 0 1 2 3 4; do
  python train_attributes.py \
      --fold $k --epochs 40 --batch 64 --lr 3e-4 \
      --backbone efficientnet_b0 \
      --project runs/attribute --name cv_fold$k \
      --device 0 --workers 2
done
```

### Cell-type head

Two backbones and two label granularities. `--target celltype` is the 14 LLD
classes; `--target group` collapses them to the 6 clinical groups used by
`leukemia_features.py` (worth running: `Basophil` has 16 training cells).

```bash
for k in 0 1 2 3 4; do
  python train_cellType.py \
      --fold $k --target celltype --backbone dinobloom \
      --dinobloom-variant l --dinobloom-weights /path/to/DinoBloom-L.pth \
      --manifest generated/celltype_manifest.csv \
      --epochs 40 --batch 64 --balance sampler \
      --project runs/celltype --device 0
done
```

Checkpoints go to `<project>/<backbone>_<target>_<train-dom>2<test-dom>_cv_fold${K}/fold${K}_celltype.pt`
— with the defaults above, `runs/celltype/dinobloom_celltype_all2all_cv_fold0/fold0_celltype.pt`.
`--train-domain` / `--test-domain` (`C1`, `C2`, or omitted for both) select
cameras and change that directory name.

Pool the five folds into the `pooled_summary.json` that
`collect_perception.py` reads:

```bash
cd ../..                       # agentic_hematology/
python pool_celltype_folds.py  # TODO: arguments
```

> **TODO.** `collect_perception.py` expects the pooled summaries under
> `runs/celltype/dinobloom_C1C2_celltype/` and `…_group/`, which
> `train_cellType.py`'s own naming scheme
> (`<backbone>_<target>_<dom>2<dom>_<run>`) does not produce. Record the
> `--project`/`--name` combination actually used, or whether
> `pool_celltype_folds.py` writes that directory itself.

### Subtype classifier

```bash
cd agentic_hematology
python train_leukemia_from_dinobloom.py   # TODO: arguments
```

`train_leukemia_from_efficientnet.py` and `train_leukemia_from_stats.py` are
the ablation arms; `train_patient_clf_cameras.py` and `export_camera_clf.py`
cover the per-camera variant.

### Report generator (LoRA SFT)

```bash
cd report
python scripts/10_build_grpo_e2e.py --rebuild-sft
python scripts/06_prepare_verl_data.py --mode sft
python scripts/11_verify_e2e_setup.py

bash ../verl_scripts/train.sh sft ${NGPUS}
bash ../verl_scripts/export_lora.sh ${EXPORT_STEP}      # EXPORT_STEP=50

python scripts/12_eval_lora_val.py \
    --parquet ${SFT_VAL} --adapter ${SFT_LORA_ADAPTER} \
    --stage sft --out metrics_sft_val.json
```

The GRPO stage (`train.sh grpo`) is **not used**; all reported results use the
SFT checkpoint. `cleanup_ray.sh`, `reward_report.py`, `reward_morphology.py`
and `reward_report_e2e.py` are GRPO-only.

> **TODO — two entry points.** `verl_scripts/` contains both `train.sh` (called
> as `train.sh sft $NGPUS` by `pipeline/run.sh`) and a standalone `run_sft.sh`.
> Say which one produced the reported adapter and delete or document the other.
>
> `$SFT_VAL`, `$SFT_LORA_ADAPTER` and `$SFT_SAVE_DIR` come from
> `verl_scripts/env.sh`. Commit a sanitised `env.sh.example` — the real one
> almost certainly carries cluster paths.
>
> `filter_lora_lm_only.{py,sh}` sits between training and export; confirm
> whether the released adapter went through it.

---

## 5. Hardware

From the Slurm wrappers, as actually run:

| Job | GPUs | CPUs | RAM | Walltime |
|---|---|---|---|---|
| Full pipeline (`sbatch_full.sh`) | 4 | 64 | 230 G | 72 h |
| Perception only (`sbatch_stage1.sh`) | 1 | 16 | 64 G | 72 h |
| Report validation (`sbatch_validate.sh`) | 1 | 8 | 32 G | 4 h |

```bash
cd wbc_unified && sbatch pipeline/sbatch_stage1.sh
```

`pipeline/wipe_checkpoints.sh` deletes training checkpoints (never datasets or
parquet); it requires `CONFIRM_WIPE=1` and supports `--dry-run`.

> **TODO.** Replace the measured walltimes above with actual elapsed time per
> fold — the table gives the *limits* requested, not the cost.

---

## 6. Inference

### Full pipeline (LLM router + reflector + lineage guard)

```bash
python run_orchestrator.py \
    --lld-split test \
    --backend dinobloom \
    --yolo-weights  wbc_unified/cv/runs/detector/fold${K}/weights/best.pt \
    --dinobloom-weights /path/to/DinoBloom-L.pth \
    --dinobloom-attr-weights wbc_unified/cv/runs/attribute_dinobloom/cv_fold${K}/fold${K}_attr_dinobloom.pt \
    --classifier-model <fold-k classifier>/leukemia_random_forest.pkl \
    --report-backend local-llm \
    --llm-model   /path/to/Qwen3-VL-4B-Instruct \
    --lora-adapter <sft adapter>/adapter_model.safetensors \
    --max-reflect-iterations 2 \
    --device 0 \
    --out outputs/full \
    --instruction "Generate a full diagnostic report"
```

### Ablations

```bash
# no reflector (lineage guard still runs)
python run_orchestrator.py ... --no-reflect --out outputs/no_reflect

# deterministic keyword routing instead of the LLM router
python run_orchestrator.py ... --no-router-agent --out outputs/rule_router

# fully deterministic
python run_orchestrator.py ... --no-reflect --no-router-agent --out outputs/deterministic
```

### Other intents

```bash
python run_orchestrator.py ... --instruction "detect only"
python run_orchestrator.py ... --instruction "classify only"

python run_orchestrator.py --case-id <id> \
    --findings-json outputs/full/<id>/case_<id>_pre_reflection_intermediates.json \
    --instruction "generate a report from these findings" \
    --out outputs/from_json
```

`--no-reflect` disables only the reflection loop. The reflector's three actions
are PROCEED, RE_AGGREGATE (re-runs aggregation at threshold + 0.15, then
re-classifies) and FLAG_FOR_REVIEW; it never writes a diagnosis. Because
RE_AGGREGATE changes the evidence record, **running with and without the
reflector produces different reports**, not merely different flags.

---

## 7. Evaluation

Run from `wbc_unified/cv` (first block) or `agentic_hematology/` (second).

| Paper result | Command |
|---|---|
| Perception, DinoBloom (per fold) | `scripts/pipeline_eval_dinobloom.py joint --fold $K --det-weights … --attr-weights … --json-out …` |
| Perception, DinoBloom (GT crops only) | `scripts/pipeline_eval_dinobloom.py attribute --fold $K --attr-weights … --json-out …` |
| Perception, EfficientNet (ablation) | `scripts/pipeline_eval.py {detector,attribute,joint} --json-out …` |
| Cell-type classifier | `eval_cellType.py --weights … --split test --json-out …` |
| Differential / blast-% error | `eval_pipeline_differential.py --mode {joint,decoupled} --det-weights … --json-out …` |
| Perception table (aggregates the below) | `collect_perception.py --det-dir <dir> --out <dir>` |

| Paper result | Command |
|---|---|
| Detector, per class × fold | `detection_perClass_folds.py` → `per_class_{summary,by_fold}.csv` |
| Cell-type, pooled over folds | `pool_celltype_folds.py` → `pooled_summary.json` |
| Attribute probes, camera × camera | `compare_attr_probes.py` → `attribute_probe_comparison.json` |
| Subtype classification | `eval_leukemia_classifiers_dino.py` (`_eff.py` for the ablation) |
| Cross-validated subtype results | `eval_cv.py` |
| External validation (cAItomorph) | `eval_caitomorph.py`, `eval_caitomorph_ensemble.py` |
| Routing | `eval_router.py` |
| Numeric containment | `check_numeric_containment.py` |
| Reflection ablation | `reflection_evaluation_results.py`, `analyze_agent_trace.py` |
| Zero-shot report baseline | `llm_impression_baseline.py` |
| Report scoring | `judge_reports.py`, `report/scripts/08_eval_llm_reports.py` |
| Metrics printout | `report/scripts/09_print_metrics.py` |
| Result aggregation | `collect_results.py`, `collect_external_extras.py`, `summarize_batch_eval.py`, `summarize_patient_clf_cameras.py` |

> **TODO.** The three script↔table mappings I inferred from filenames rather
> than contents — Table 1a ← `check_numeric_containment.py`, Table 2 ←
> `reflection_evaluation_results.py`, Table 1b's zero-shot column ←
> `llm_impression_baseline.py` — need confirming, along with each one's
> arguments.

`eval_pipeline_differential.py` is the one that justifies the decoupled design:
it holds the detector boxes fixed and compares the YOLO class head (`--mode
joint`) against a separate crop classifier (`--mode decoupled`) on the
per-patient differential percentages that feed the subtype classifier,
including blast-pool error against the 20 % threshold. Both scripts accept
`--compare a.json b.json …` to print a side-by-side table.

`collect_perception.py` hardcodes
`--out /home/roba.majzoub/research/AgenticHematology/outputs/paper_results`.
Change the default to a relative path before release.

---

## 8. Outputs

Each case directory contains:

```
case_<id>_detections.json       cells, boxes, classes, attributes
case_<id>_classification.json   subtype + confidence
case_<id>_validation.json       containment + consistency results
case_<id>_agent_trace.json      reflector decisions and stated reasons
```

A case flagged for review has a review banner appended to the end of the report
markdown, so the flag survives downstream.

> **TODO.** Confirm the report filename. The pipeline appends the banner to
> `state.report.markdown`; the orchestrator's on-disk name needs checking
> against `report.markdown` / `case_<id>_report.md`.

---

## 9. Known issues

Open before the repo is usable by anyone else:

- **`infer.py` appears stale.** It calls
  `build_attribute_model(num_attrs=…)`, while `train_attributes.py` and
  `evaluate.py` call `build_attribute_model(num_binary=…, num_size=…)`. It also
  treats the model output as a single logits tensor and sigmoids all seven
  attributes, whereas the trained model returns `(bin_logits, size_logits)`
  with a 3-class categorical `Cell_Size`. Its `torch.load` omits
  `weights_only=False`, which fails on torch ≥ 2.6. Both
  `pipeline/lib_stage1.sh` and `pipeline/run_stage1_infer.sh` call it.
- **`eval_cellType.py` has no `--fold`.** It evaluates the original manifest
  split, not a CV fold, unlike `pipeline_eval_dinobloom.py`, which rewrites the
  manifest's `split` column per fold. Per-fold cell-type numbers cannot come
  from this script as written.
- **Hardcoded absolute paths**, including other users' home and scratch
  directories, in `pipeline/run.sh`, `pipeline/lib_stage1.sh`,
  `pipeline/wipe_checkpoints.sh`, `pipeline/sbatch_*.sh`,
  `collect_perception.py` and `data/prepare_dataset.py`. These leak internal
  cluster layout and must be replaced with env vars or relative paths.
- **`#SBATCH --exclude=gpu-62,gpu-45`** in the sbatch wrappers is site-specific.
- **Two `configs/` directories**: `cv/configs/` (dataset yamls) and
  `cv/data/configs/` (`cv_folds.json`). Add a note in each so neither is
  cleaned up by mistake.
- `breakpoint()` in `patient_info_NoOverlap.py`; commented ones in
  `pipeline.py`.
- Hardcoded API key in `LLM_report_generator.py`. Rotate it — removing the line
  does not remove it from git history.
- **Two stray directories** in `wbc_unified/`: `'trainer.val_before_train=True'`
  and `'trainer.val_only=True'`. These are mistyped verl CLI arguments that
  became directories. Delete them.
- **`API_report _generator.py` has a space in its filename.** Rename it.
- `pipeline/run_stage1_infer.sh` and `report/scripts/run_stage1_infer.sh` are
  two copies. Keep one.
- Three copies of `yolo11m.pt` / `yolo26n.pt` (repo root, `agentic_hematology/`,
  `cv/`, `wbc_unified/`) plus `trained_weights.tar.gz`, `outputs.tar.gz` and
  `LLM_reports.tar.gz`. None belong in git.
- Scratch files at the repo root: `new.py`, `compare_1.py`, `qualitative.py`,
  `best.png`, `README_original.md`. Decide what ships.
- `mean_attrs_binaty.py` — typo in the filename.
- `.gitignore` needs `third_party/`, `runs/`, `output{,s}/`, `logs/`,
  `generated*/`, `__pycache__/`, `nltk_data/`, `*.pt`, `*.pkl`, `*.tar.gz`.
- Dead code: `compare()` in `eval_cellType.py` contains a `for … pass` loop;
  `aggregator.py` computes `cohort_type`, `_blast_morphology()` and a first
  `diagnostic_flags` dict that are all overwritten by `case_record["report_ready"]`.
- `detection_agent.py`, `detection_agent_v2.py` and
  `detection_agent_dinobloom.py` all coexist. Only the third is on the reported
  path; say so or remove the others.

---

## 10. Data and weights

> **TODO — required before publication.**
>
> - LLD (LeukemiaAttri): where to request it, and whether its licence permits
>   redistributing `reference_reports/`, which is derived from its annotations.
> - cAItomorph: same.
> - DinoBloom-L: download link (third-party weights, not redistributed here).
> - YOLOv11 pretrained `yolo11m.pt`: Ultralytics, note the licence.
> - Whether this repo ships trained checkpoints (detector, probe, random forest,
>   LoRA adapter) or reproduction requires a full retrain.
> - `LICENSE` for the code, and a separate statement for the reference reports
>   if their terms differ.

---

<!-- ## Citation

```bibtex
@inproceedings{abacus2027,
  title     = {ABACUS: Traceable Patient-Level Leukemia Reporting from
               Peripheral Blood Smears},
  author    = {TODO: full author list},
  booktitle = {IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2027}
} -->
```