#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
: "${DATASET_ROOT:?Set DATASET_ROOT to your prepared dataset}"
: "${MODEL_ROOT:?Set MODEL_ROOT to the base model directory}"
CONFIG_NAME=${CONFIG_NAME:-fold_clothes_train}
EXPECTED_LEN=${EXPECTED_LEN:-2846}
: "${SAVE_ROOT:?Set SAVE_ROOT to the checkpoint output directory}"
MASTER_PORT=${MASTER_PORT:-29621}

if [[ ! -d "${CODE_ROOT}" ]]; then echo "Missing CODE_ROOT: ${CODE_ROOT}" >&2; exit 2; fi
if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then echo "Missing dataset meta/info.json: ${DATASET_ROOT}" >&2; exit 3; fi
if [[ ! -d "${MODEL_ROOT}/transformer" ]]; then echo "Missing LingBot/Wan model transformer dir: ${MODEL_ROOT}/transformer" >&2; exit 4; fi

cd "${CODE_ROOT}"
if [[ -z "${PY:-}" ]]; then
  if [[ -x /opt/venv/bin/python ]]; then PY=/opt/venv/bin/python
  elif [[ -x /opt/venv/bin/python3 ]]; then PY=/opt/venv/bin/python3
  elif command -v python3 >/dev/null 2>&1; then PY=$(command -v python3)
  elif command -v python >/dev/null 2>&1; then PY=$(command -v python)
  else echo "No Python interpreter found." >&2; exit 6
  fi
fi

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai}
export WANDB_PROJECT=${WANDB_PROJECT:-lingbot_va_fold_clothes}
export WANDB_TEAM_NAME=${WANDB_TEAM_NAME:-}
export DATASET_ROOT MODEL_ROOT
export PYTHONPATH="${CODE_ROOT}:${CODE_ROOT}/wan_va${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${WANDB_API_KEY:-}" && -z "${WANDB_TEAM_NAME}" ]]; then echo "Set WANDB_TEAM_NAME when using W&B" >&2; exit 2; fi

VISIBLE_GPU_COUNT=$(${PY} - <<'COUNTGPUS'
import torch
print(torch.cuda.device_count())
COUNTGPUS
)
if [[ "${VISIBLE_GPU_COUNT}" -lt 1 ]]; then echo "No CUDA GPU is visible to PyTorch inside this container." >&2; exit 7; fi
if [[ -z "${NGPU:-}" ]]; then NGPU=${VISIBLE_GPU_COUNT}
elif [[ "${NGPU}" -gt "${VISIBLE_GPU_COUNT}" ]]; then
  echo "Requested NGPU=${NGPU}, but PyTorch only sees ${VISIBLE_GPU_COUNT} CUDA device(s)." >&2
  exit 8
fi

mkdir -p "${SAVE_ROOT}"
echo "CODE_ROOT=${CODE_ROOT}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "MODEL_ROOT=${MODEL_ROOT}"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "EXPECTED_LEN=${EXPECTED_LEN}"
echo "SAVE_ROOT=${SAVE_ROOT}"
echo "NGPU=${NGPU}"
echo "VISIBLE_GPU_COUNT=${VISIBLE_GPU_COUNT}"
echo "PY=${PY}"
echo "WANDB_ENABLED=$([[ -n "${WANDB_API_KEY:-}" ]] && echo 1 || echo 0)"
echo "WANDB_TEAM_NAME=${WANDB_TEAM_NAME:-}"
echo "WANDB_PROJECT=${WANDB_PROJECT:-}"
echo "WANDB_NAME=${WANDB_NAME:-}"
echo "LINGBOT_BATCH_SIZE=${LINGBOT_BATCH_SIZE:-default}"
echo "LINGBOT_GRAD_ACCUM=${LINGBOT_GRAD_ACCUM:-default}"
echo "LINGBOT_NUM_STEPS=${LINGBOT_NUM_STEPS:-default}"
echo "LINGBOT_LR=${LINGBOT_LR:-default}"
echo "LINGBOT_LOAD_WORKER=${LINGBOT_LOAD_WORKER:-default}"
echo "LINGBOT_DATASET_INIT_WORKER=${LINGBOT_DATASET_INIT_WORKER:-default}"

"${PY}" tools/validate_fold_lingbot_dataset.py --config-name "${CONFIG_NAME}" --expected-len "${EXPECTED_LEN}" --dataset-init-worker 1

if [[ "${LINGBOT_VALIDATE_ONLY:-0}" == "1" ]]; then
  echo "LINGBOT_VALIDATE_ONLY=1, validation passed; exiting before training."
  exit 0
fi

"${PY}" -m torch.distributed.run   --nproc_per_node="${NGPU}"   --master_port "${MASTER_PORT}"   --tee 3   -m wan_va.train   --config-name "${CONFIG_NAME}"   --save-root "${SAVE_ROOT}"   ${RUN_MODE_ARGS:-}
