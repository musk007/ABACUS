#!/usr/bin/env bash
# LoRA SFT | Qwen3-VL-4B-Instruct | morphology paragraph | verl FSDP
# Usage: bash verl_scripts/run_sft.sh <nproc_per_node> [hydra overrides...]
# Small datasets: prefer nproc=1 to avoid multi-GPU checkpoint OOM

set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT="$PROJECT/report"
cd "$REPORT"
# shellcheck source=env.sh
source "$PROJECT/verl_scripts/env.sh"

[ "$#" -ge 1 ] || { echo "Usage: $0 <nproc_per_node> [overrides...]"; exit 1; }

NPROC="$1"
shift

TRAIN_FILE="${TRAIN_FILE:-$REPORT_LLM_ARTIFACT_ROOT/data/verl/sft/train.parquet}"
VAL_FILE="${VAL_FILE:-$REPORT_LLM_ARTIFACT_ROOT/data/verl/sft/val.parquet}"
SAVE_PATH="${SAVE_PATH:-$SFT_SAVE_DIR}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-$REPORT_LLM_ARTIFACT_ROOT/outputs/sft/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$(dirname "$HYDRA_RUN_DIR")" "$SAVE_PATH"

[ -f "$MODEL_PATH/config.json" ] || { echo "Missing base model: $MODEL_PATH"; exit 1; }
[ -f "$TRAIN_FILE" ] || { echo "Missing $TRAIN_FILE — run: bash verl_scripts/train.sh data"; exit 1; }

# 33 train samples / batch 4 ≈ 8 steps/epoch; epochs is only an upper bound, training stops at TOTAL_STEPS
TOTAL_EPOCHS="${TOTAL_EPOCHS:-8}"
TOTAL_STEPS="${TOTAL_TRAINING_STEPS:-50}"
SAVE_FREQ="${SAVE_FREQ:-$TOTAL_STEPS}"
MAX_CKPT="${MAX_CKPT_TO_KEEP:-1}"
TEST_FREQ="${TEST_FREQ:-25}"
LR="${LR:-1e-4}"
MAX_LEN="${MAX_LENGTH:-4096}"
TRUNCATION="${TRUNCATION:-error}"  # left-truncation would cut the GT system prompt
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

echo "LoRA SFT: base=${MODEL_PATH} rank=${LORA_RANK} alpha=${LORA_ALPHA} epochs=${TOTAL_EPOCHS} steps=${TOTAL_STEPS} -> ${SAVE_PATH}"

python -c "import verl.trainer.sft_trainer" 2>/dev/null || {
  echo "verl not installed: bash verl_scripts/install.sh"
  exit 1
}

torchrun --nnodes=1 --nproc_per_node="${NPROC}" \
  -m verl.trainer.sft_trainer \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.messages_key=messages \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-4}" \
  data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU:-1}" \
  data.max_length="${MAX_LEN}" \
  data.truncation="${TRUNCATION}" \
  data.ignore_input_ids_mismatch=true \
  data.num_workers="${DATALOADER_WORKERS:-2}" \
  model.path="${MODEL_PATH}" \
  model.lora_rank="${LORA_RANK}" \
  model.lora_alpha="${LORA_ALPHA}" \
  model.target_modules="${LORA_TARGETS}" \
  model.use_remove_padding=true \
  model.enable_gradient_checkpointing=true \
  "+model.override_config.attn_implementation=${ATTN_IMPLEMENTATION}" \
  optim.lr="${LR}" \
  engine=fsdp \
  engine.model_dtype="${MODEL_DTYPE}" \
  engine.param_offload="${FSDP_PARAM_OFFLOAD:-true}" \
  trainer.default_local_dir="${SAVE_PATH}" \
  trainer.project_name=leukemia_report_sft \
  trainer.experiment_name="${VERL_RUN_TAG}" \
  trainer.logger='["console"]' \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.max_ckpt_to_keep="${MAX_CKPT}" \
  trainer.test_freq="${TEST_FREQ}" \
  checkpoint.save_contents='[model]' \
  hydra.run.dir="${HYDRA_RUN_DIR}" \
  "$@"
