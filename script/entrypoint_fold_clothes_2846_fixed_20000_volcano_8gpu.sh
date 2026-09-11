#!/usr/bin/env bash
set -euo pipefail

export PY=${PY:-/opt/venv/bin/python}
export CODE_ROOT=${CODE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
: "${DATASET_ROOT:?Set DATASET_ROOT}"
export DATASET_ROOT
: "${MODEL_ROOT:?Set MODEL_ROOT}"
export MODEL_ROOT
export CONFIG_NAME=${CONFIG_NAME:-fold_clothes_train}
export EXPECTED_LEN=${EXPECTED_LEN:-2846}

export NGPU=${NGPU:-8}
export LINGBOT_BATCH_SIZE=${LINGBOT_BATCH_SIZE:-1}
export LINGBOT_GRAD_ACCUM=${LINGBOT_GRAD_ACCUM:-2}
export LINGBOT_NUM_STEPS=${LINGBOT_NUM_STEPS:-20000}
export LINGBOT_SAVE_INTERVAL=${LINGBOT_SAVE_INTERVAL:-2000}
export LINGBOT_LR=${LINGBOT_LR:-2.5e-5}
export LINGBOT_LOAD_WORKER=${LINGBOT_LOAD_WORKER:-4}
export LINGBOT_DATASET_INIT_WORKER=${LINGBOT_DATASET_INIT_WORKER:-1}
export LINGBOT_DEBUG_PROGRESS=${LINGBOT_DEBUG_PROGRESS:-1}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

export WANDB_PROJECT=${WANDB_PROJECT:-lingbot_va_fold_clothes}
export WANDB_TEAM_NAME=${WANDB_TEAM_NAME:-}
export WANDB_NAME=${WANDB_NAME:-fold_clothes_2846_fixed_volcano_8gpu_20000}

exec bash "${CODE_ROOT}/script/entrypoint_fold_clothes_train_platform.sh"
