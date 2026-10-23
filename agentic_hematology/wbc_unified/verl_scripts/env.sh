# verl training environment (sourced by run_*.sh / train.sh / sbatch)
LOCAL_MODEL_ROOT="/home/roba.majzoub/research/AgenticHematology/report_llm/pretrained_models"
VERL_MODEL_DIR="${VERL_MODEL_DIR:-Qwen3-VL-4B-Instruct}"
# VERL_MODEL_DIR="${VERL_MODEL_DIR:-Qwen3.5-2B}"
MODEL_PATH="${MODEL_PATH:-${LOCAL_MODEL_ROOT}/${VERL_MODEL_DIR}}"
VERL_RUN_TAG="${VERL_RUN_TAG:-qwen3_4b_instruct_lora}"

LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_TARGETS="${LORA_TARGETS:-all-linear}"

export FOLD="${FOLD:-}"
if [[ -n "$FOLD" ]]; then
  export REPORT_LLM_ARTIFACT_ROOT="/home/roba.majzoub/research/AgenticHematology/report_llm/fold${FOLD}"
else
  export REPORT_LLM_ARTIFACT_ROOT="${REPORT_LLM_ARTIFACT_ROOT:-/home/roba.majzoub/research/AgenticHematology/report_llm}"
fi
mkdir -p "${REPORT_LLM_ARTIFACT_ROOT}"/{data,runs,outputs,cache,logs}

export CUDA_HOME=/usr/local/cuda-12.2
export CUDA_PATH=/usr/local/cuda-12.2
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export SFT_SAVE_DIR="${SFT_SAVE_DIR:-${REPORT_LLM_ARTIFACT_ROOT}/runs/verl/sft_${VERL_RUN_TAG}}"
export GRPO_SAVE_DIR="${GRPO_SAVE_DIR:-${REPORT_LLM_ARTIFACT_ROOT}/runs/verl/grpo_${VERL_RUN_TAG}}"
export GRPO_LORA_STEP="${GRPO_LORA_STEP:-30}"
export GRPO_LORA_ADAPTER="${GRPO_LORA_ADAPTER:-${GRPO_SAVE_DIR}/global_step_${GRPO_LORA_STEP}/actor/lora_adapter_lm}"
_sft_step="$(cat "${SFT_SAVE_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null || echo 50)"
export SFT_LORA_ADAPTER="${SFT_LORA_ADAPTER:-${SFT_SAVE_DIR}/global_step_${_sft_step}/lora_adapter}"
# vLLM language_model_only rejects visual.* LoRA keys; FSDP actor uses _lm, vLLM rollout uses _vllm
SFT_LORA_ADAPTER_LM="${SFT_LORA_ADAPTER%/}_lm"
if [ -f "${SFT_LORA_ADAPTER_LM}/adapter_model.safetensors" ]; then
  export SFT_LORA_ADAPTER="${SFT_LORA_ADAPTER_LM}"
fi
export VLLM_LORA_ADAPTER="${VLLM_LORA_ADAPTER:-${SFT_LORA_ADAPTER%/}_vllm}"
if [ ! -f "${VLLM_LORA_ADAPTER}/adapter_model.safetensors" ]; then
  VLLM_LORA_ADAPTER="${SFT_LORA_ADAPTER}"
fi
export VLLM_LORA_ADAPTER
export CV_FOLDS_JSON="${CV_FOLDS_JSON:-/home/roba.majzoub/research/AgenticHematology/agentic_hematology/wbc_unified/cv/configs/cv_folds.json}"

export HF_HOME="${HF_HOME:-${REPORT_LLM_ARTIFACT_ROOT}/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${REPORT_LLM_ARTIFACT_ROOT}/cache/pip}"
# echo "${HF_HOME}"
mkdir -p "${HF_HOME}" "${PIP_CACHE_DIR}"

export LOCAL_MODEL_ROOT VERL_MODEL_DIR MODEL_PATH VERL_RUN_TAG CV_FOLDS_JSON
export LORA_RANK LORA_ALPHA LORA_TARGETS
export REPORT_LLM_ARTIFACT_ROOT SFT_SAVE_DIR GRPO_SAVE_DIR GRPO_LORA_STEP GRPO_LORA_ADAPTER SFT_LORA_ADAPTER

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
  unset ROCR_VISIBLE_DEVICES
fi

# Drop login-node polluted LD paths; keep system libnvidia-ml for NVML/Ray/nvidia-smi
strip_bad_ld_paths() {
  if [ -z "${LD_LIBRARY_PATH:-}" ]; then
    return 0
  fi
  local out="" p
  IFS=':' read -ra _ld_parts <<< "${LD_LIBRARY_PATH}"
  for p in "${_ld_parts[@]}"; do
    [ -n "$p" ] || continue
    case "$p" in
      /apps/local/anaconda3/*|*cuda_version/cuda-12.*|*cuda_version/cuda-13.*) continue ;;
    esac
    out="${out}${out:+:}${p}"
  done
  unset _ld_parts
  if [ -n "$out" ]; then
    export LD_LIBRARY_PATH="$out"
  else
    unset LD_LIBRARY_PATH
  fi
}

# Prefer pip cu129 torch libs + system libnvidia-ml; do not module load cuda (old libcudart)
setup_vllm_cuda_path() {
  local sp dirs=() sysdirs=() clean="" p
  sp="$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)" || return 0
  for d in \
    "$sp/torch/lib" \
    "$sp/nvidia/cuda_runtime/lib" \
    "$sp/nvidia/cu13/lib"; do
    [ -d "$d" ] && dirs+=("$d")
  done
  for d in \
    /usr/lib/x86_64-linux-gnu \
    /usr/lib64/nvidia \
    /usr/local/cuda/lib64; do
    [ -d "$d" ] && sysdirs+=("$d")
  done
  [ "${#dirs[@]}" -eq 0 ] && [ "${#sysdirs[@]}" -eq 0 ] && return 0

  for d in "${dirs[@]}"; do
    clean="${clean}${clean:+:}${d}"
  done
  if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    IFS=':' read -ra _vllm_ld_parts <<< "${LD_LIBRARY_PATH}"
    for p in "${_vllm_ld_parts[@]}"; do
      [ -n "$p" ] || continue
      case "$p" in
        *cuda_version/cuda-*|*/cuda-12.*|*/cuda-13.*) continue ;;
        /apps/local/anaconda3/*) continue ;;
      esac
      [[ ":${clean}:" != *":${p}:"* ]] && clean="${clean}:${p}"
    done
    unset _vllm_ld_parts
  fi
  for d in "${sysdirs[@]}"; do
    [[ ":${clean}:" != *":${d}:"* ]] && clean="${clean}:${d}"
  done
  export VLLM_CUDA_LIB_PATH="$(IFS=:; echo "${dirs[*]}")"
  export LD_LIBRARY_PATH="${clean}"
  export VERL_LD_LIBRARY_PATH="${clean}"
}

if command -v python3 >/dev/null 2>&1; then
  setup_vllm_cuda_path
fi
