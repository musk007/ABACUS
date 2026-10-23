#!/usr/bin/env bash
# Data prep / LoRA SFT / export adapter / GRPO
# Usage:
#   bash verl_scripts/train.sh data
#   bash verl_scripts/train.sh sft 4
#   bash verl_scripts/train.sh export [step]
#   bash verl_scripts/train.sh grpo
#   bash verl_scripts/train.sh all 4
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT="$PROJECT/report"
cd "$REPORT"
# shellcheck source=env.sh
source "$PROJECT/verl_scripts/env.sh"

cmd="${1:-}"
shift || true

prepare_data() {
  python scripts/10_build_grpo_e2e.py ${GT_STATS_JSON:+--stats-json "$GT_STATS_JSON"} ${FOLD:+--fold "$FOLD"}
  python scripts/06_prepare_verl_data.py ${FOLD:+--fold "$FOLD"}

}

case "$cmd" in
  data) prepare_data ;;
  sft)
    prepare_data
    set +e
    bash "$PROJECT/verl_scripts/run_sft.sh" "${1:?Usage: train.sh sft <nproc>}"
    sft_rc=$?
    set -e
    step="${EXPORT_STEP:-50}"
    ckpt="${SFT_SAVE_DIR}/global_step_${step}"
    if compgen -G "${ckpt}/model_world_size_*_rank_0.pt" > /dev/null; then
      echo "SFT exit=$sft_rc but model shards exist — exporting LoRA"
      bash "$PROJECT/verl_scripts/export_lora.sh" "$step"
      [ "$sft_rc" -eq 0 ] || echo "Note: training exited $sft_rc; adapter export may still succeed."
    else
      exit "$sft_rc"
    fi
    ;;
  export) bash "$PROJECT/verl_scripts/export_lora.sh" "${1:-50}" ;;
  grpo) bash "$PROJECT/verl_scripts/run_grpo_4B.sh" "$@" ;;
  # grpo) bash "$PROJECT/verl_scripts/run_grpo.sh" "$@" ;;
  all)
    prepare_data
    bash "$PROJECT/verl_scripts/run_sft.sh" "${1:?Usage: train.sh all <nproc>}"
    bash "$PROJECT/verl_scripts/export_lora.sh" "${EXPORT_STEP:-50}"
    bash "$PROJECT/verl_scripts/run_grpo_4B.sh" "${@:2}"
    # bash "$PROJECT/verl_scripts/run_grpo.sh" "${@:2}"
    ;;
  *)
    echo "Usage: $0 {data|sft|export|grpo|all} [args...]"
    exit 1
    ;;
esac
