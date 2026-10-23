#!/usr/bin/env bash
# End-to-end WBC pipeline (from scratch, no resume):
#   CV detect -> eval | attributes -> eval | joint | infer
#   -> data build -> SFT -> eval | GRPO -> deploy eval
#
#   cd wbc_unified
#   export CONFIRM_FULL_RETRAIN=1
#   bash pipeline/run.sh
#   sbatch pipeline/sbatch_full.sh

set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CV="$PROJECT/cv"
REPORT="$PROJECT/report"
LOG_DIR="$PROJECT/logs"
mkdir -p "$LOG_DIR"

export DATA_ROOT="${DATA_ROOT:-/nfs-stor/zongyan/datasets/medical/LeukemiaDataset_Organized}"
export STAGE1_NGPUS="${STAGE1_NGPUS:-1}"
export LLM_NGPUS="${LLM_NGPUS:-${NGPUS:-4}}"
export CONFIRM_FULL_RETRAIN="${CONFIRM_FULL_RETRAIN:-0}"

ARTIFACT="${REPORT_LLM_ARTIFACT_ROOT:-/nfs-stor/zongyan/wbc_medical/rao.anwer/report_llm}"
JOB_TAG="${SLURM_JOB_ID:-local_$$}"
PIPELINE_EVAL_DIR="${PIPELINE_EVAL_DIR:-$LOG_DIR/pipeline_eval_${JOB_TAG}_$(date +%Y%m%d_%H%M%S)}"

_wipe_checkpoints() {
  export WIPE_STAGE1="${WIPE_STAGE1:-$([[ "${SKIP_STAGE1:-0}" == "1" ]] && echo 0 || echo 1)}"
  if [[ "${SKIP_SFT:-0}" == "1" && "${SKIP_GRPO:-0}" == "1" ]]; then
    export WIPE_LLM="${WIPE_LLM:-0}"
  else
    export WIPE_LLM="${WIPE_LLM:-1}"
  fi
  CONFIRM_WIPE=1 bash "$PROJECT/pipeline/wipe_checkpoints.sh"
}

if [[ "$CONFIRM_FULL_RETRAIN" != "1" ]]; then
  echo "ERROR: This will delete checkpoints and retrain from scratch." >&2
  echo "Preview wipe targets:" >&2
  _wipe_checkpoints
  echo "" >&2
  echo "  export CONFIRM_FULL_RETRAIN=1" >&2
  echo "  bash pipeline/run.sh" >&2
  exit 2
fi

echo "======== WBC unified pipeline (from scratch) ========"
_wipe_checkpoints

# shellcheck source=pipeline/lib_eval.sh
source "$PROJECT/pipeline/lib_eval.sh"
pipeline_eval_init "$PIPELINE_EVAL_DIR"
echo "PIPELINE_EVAL_DIR=$PIPELINE_EVAL_DIR"

_activate_stage1_env() {
  for d in "${CONDA_ROOT:-}" "$HOME/anaconda3" "$HOME/miniconda3" /apps/local/anaconda3; do
    [ -n "$d" ] || continue
    if [[ -f "$d/etc/profile.d/conda.sh" ]]; then
      # shellcheck source=/dev/null
      source "$d/etc/profile.d/conda.sh"
      module load nvidia/cuda/11.8 2>/dev/null || true
      conda activate "${CONDA_ENV:-SLA_Det}"
      return 0
    fi
  done
}

_activate_llm_env() {
  module unload nvidia/cuda 2>/dev/null || true
  # shellcheck source=/dev/null
  source "$PROJECT/verl_scripts/env.sh"
}

if [[ "${SKIP_STAGE1:-0}" != "1" ]]; then
  _activate_stage1_env
  # shellcheck source=pipeline/lib_stage1.sh
  source "$PROJECT/pipeline/lib_stage1.sh"
  stage1_train_and_infer "$CV" "$PIPELINE_EVAL_DIR"
fi

_activate_llm_env
cd "$REPORT"

export GRPO_FRESH_START="${FORCE_GRPO_FRESH:-1}"
export REWARD_FN="${REWARD_FN:-$PROJECT/verl_scripts/reward_morph.py}"
export TRAIN_FILE="${ARTIFACT}/data/verl/grpo_e2e/train.parquet"
export VAL_FILE="${ARTIFACT}/data/verl/grpo_e2e/val.parquet"
export EXPORT_STEP="${EXPORT_STEP:-50}"

SFT_VAL="${ARTIFACT}/data/verl/sft/val.parquet"
GRPO_VAL="${ARTIFACT}/data/verl/grpo_e2e/val.parquet"
MET_SFT="${PIPELINE_EVAL_DIR}/metrics_sft_val.json"
MET_GRPO_VAL="${PIPELINE_EVAL_DIR}/metrics_grpo_val.json"
MET_DEPLOY="${ARTIFACT}/data/eval/metrics_llm_pred.json"
MET_SUMMARIES="${PIPELINE_EVAL_DIR}/metrics_pred_summaries.json"

echo "======== Build LLM data ========"
python scripts/01_aggregate_gt.py
python scripts/02_aggregate_pred.py
python scripts/10_build_grpo_e2e.py --rebuild-sft
python scripts/06_prepare_verl_data.py --mode sft
python scripts/11_verify_e2e_setup.py

python scripts/13_eval_pred_summaries.py --out "$MET_SUMMARIES"
pipeline_eval_record "data_build" "pred_summaries_metrics" "$MET_SUMMARIES"
pipeline_eval_record "data_build" "grpo_e2e_train" "$TRAIN_FILE"
pipeline_eval_record "data_build" "grpo_e2e_val" "$GRPO_VAL"
pipeline_eval_record "data_build" "sft_val" "$SFT_VAL"

export NGPUS="$LLM_NGPUS"
echo "LLM GPUs: NGPUS=$NGPUS (Stage-1 used STAGE1_NGPUS=$STAGE1_NGPUS)"

if [[ "${SKIP_SFT:-0}" != "1" ]]; then
  echo "======== SFT training ========"
  bash "$PROJECT/verl_scripts/train.sh" sft "$NGPUS"
  bash "$PROJECT/verl_scripts/export_lora.sh" "$EXPORT_STEP"

  echo "======== SFT eval ========"
  python scripts/12_eval_lora_val.py \
    --parquet "$SFT_VAL" \
    --adapter "$SFT_LORA_ADAPTER" \
    --stage sft \
    --out "$MET_SFT"
  pipeline_eval_record "sft" "metrics" "$MET_SFT"
  pipeline_eval_record "sft" "adapter" "$SFT_LORA_ADAPTER"
  pipeline_eval_record "sft" "checkpoint" "${SFT_SAVE_DIR}/global_step_${EXPORT_STEP}"
fi

if [[ "${SKIP_GRPO:-0}" != "1" ]]; then
  echo "======== GRPO training ========"
  bash "$PROJECT/verl_scripts/cleanup_ray.sh" || true
  RAY_CPUS="${RAY_CPUS:-16}" bash "$PROJECT/verl_scripts/train.sh" grpo "$NGPUS"

  GRPO_STEP="${GRPO_LORA_STEP:-30}"
  GRPO_ACTOR="${GRPO_SAVE_DIR}/global_step_${GRPO_STEP}/actor"
  bash "$PROJECT/verl_scripts/export_lora.sh" "$GRPO_STEP" "$GRPO_ACTOR"
  GRPO_ADAPTER="${GRPO_ACTOR}/lora_adapter_lm"

  echo "======== GRPO eval ========"
  python scripts/12_eval_lora_val.py \
    --parquet "$GRPO_VAL" \
    --adapter "$GRPO_ADAPTER" \
    --stage grpo \
    --out "$MET_GRPO_VAL"
  pipeline_eval_record "grpo" "metrics_val" "$MET_GRPO_VAL"
  pipeline_eval_record "grpo" "adapter" "$GRPO_ADAPTER"
  pipeline_eval_record "grpo" "checkpoint" "$GRPO_ACTOR"

  echo "======== Deploy eval ========"
  export LORA_ADAPTER="$GRPO_ADAPTER"
  export SKIP_AGGREGATE=1
  bash "$PROJECT/pipeline/run_validate_pred.sh"
  pipeline_eval_record "deploy" "metrics" "$MET_DEPLOY"
  pipeline_eval_record "deploy" "reports_dir" "${ARTIFACT}/data/reports_llm_pred"
fi

pipeline_eval_write_summary "wbc_unified full pipeline from scratch"

echo ""
echo "======== Done ========"
echo "PIPELINE_EVAL_DIR=$PIPELINE_EVAL_DIR"
echo "PIPELINE_SUMMARY_JSON=$PIPELINE_SUMMARY_JSON"
date
