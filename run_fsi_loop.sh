#!/usr/bin/env bash
set -euo pipefail

# -------- SETTINGS --------
PROJECT="fractal_neurons"
VENV=".venv"
PYTHON="${VENV}/bin/python"
RUN_DIR="runs/fsi_en_v1"
DISTILL_DIR="data/self_distilled_v1"
DISTILL_JSONL="${DISTILL_DIR}/distill.jsonl"
TOKENIZER_DIR="${RUN_DIR}/tokenizer"
CONFIG_EVOL_YAML="configs/fsi_distill_v1.yaml"
EVAL_DIR="${RUN_DIR}/eval"
SWARM_LOG="${RUN_DIR}/swarm_logs.jsonl"
SWARM_PROMPT="Summarise the current capabilities of the latest Fractal-MoE checkpoint."
DEVICE="cuda"
RESUME_FLAG="${LOOP_RESUME:-1}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
export CUDA_LAUNCH_BLOCKING=0

if [ ! -d "${VENV}" ]; then
  python3 -m venv "${VENV}"
fi
# shellcheck disable=SC1090
source "${VENV}/bin/activate"

pip install --upgrade pip >/dev/null
pip install -e . --quiet || true
pip install "torch>=2.2.0" "numpy>=1.24" "tqdm" "tokenizers>=0.15" \
            "datasets>=3.0.0" "huggingface_hub" "safetensors" \
            "einops" "transformers>=4.40" "pyyaml" "ujson" "orjson" --quiet

mkdir -p "${DISTILL_DIR}" "${EVAL_DIR}"

if [ ! -d "${RUN_DIR}" ]; then
  echo "FATAL: base run directory '${RUN_DIR}' not found. Train a base model first." >&2
  exit 1
fi

BASE_CKPT=$(RUN_BASE_DIR="${RUN_DIR}" ${PYTHON} <<'PY'
import glob
import os

run_dir = os.environ.get("RUN_BASE_DIR", "")
run_dir = os.path.abspath(os.path.expanduser(run_dir))
candidates = sorted(glob.glob(os.path.join(run_dir, "ema_final_*.pt")))
if not candidates:
    candidates = sorted(glob.glob(os.path.join(run_dir, "final_*.pt")))
if candidates:
    ckpt = candidates[-1]
elif os.path.isfile(os.path.join(run_dir, "latest.pt")):
    ckpt = os.path.join(run_dir, "latest.pt")
else:
    ckpt = ""
print(ckpt)
PY
)
BASE_CKPT=${BASE_CKPT//$'\n'/}

if [ -z "${BASE_CKPT}" ] || [ ! -f "${BASE_CKPT}" ]; then
  echo "FATAL: No checkpoint found in ${RUN_DIR}." >&2
  exit 1
fi

if [ "${USE_EXISTING_TOKENIZER:-0}" = "1" ]; then
  if [ -d "${TOKENIZER_DIR}" ]; then
    echo "[tokenizer] Using existing tokenizer at ${TOKENIZER_DIR} (USE_EXISTING_TOKENIZER=1)"
  else
    echo "[tokenizer] USE_EXISTING_TOKENIZER=1 but ${TOKENIZER_DIR} missing; consider building it manually"
  fi
fi

echo "[distill] Generating distilled dataset from ${BASE_CKPT}"
rm -f "${DISTILL_JSONL}"
${PYTHON} -m ${PROJECT}.distill \
  --ckpt "${BASE_CKPT}" \
  --output "${DISTILL_JSONL}" \
  --limit 10000 \
  --max_new_tokens 192 \
  --temperature 0.8 \
  --top_k 50 \
  --seq_len 512 \
  --steps 8 \
  --device "${DEVICE}" \
  --min_len 64 \
  --max_len 1024 \
  --accept-all

echo "[swarm] Orchestrating multi-agent debate"
${PYTHON} -m ${PROJECT}.orchestrator \
  --ckpt "${BASE_CKPT}" \
  --prompt "${SWARM_PROMPT}" \
  --rounds 4 \
  --device "${DEVICE}" \
  --log "${SWARM_LOG}" || true

echo "[eval] Running evaluation harness"
${PYTHON} -m ${PROJECT}.eval_harness \
  --ckpt "${BASE_CKPT}" \
  --device "${DEVICE}" \
  --output "${EVAL_DIR}/report.json"

echo "[evolve] Finetuning on distilled data"
if [ "${RESUME_FLAG}" != "0" ]; then
  RESUME_ARG="--resume"
else
  RESUME_ARG=""
fi
${PYTHON} -m ${PROJECT}.train \
  --config "${CONFIG_EVOL_YAML}" \
  --device "${DEVICE}" \
  ${RESUME_ARG}

echo "✅ Loop complete. Distilled data in ${DISTILL_JSONL}"
