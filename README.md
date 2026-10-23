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

