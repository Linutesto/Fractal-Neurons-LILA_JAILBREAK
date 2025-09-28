#!/usr/bin/env bash
set -euo pipefail

# -------- SETTINGS --------
PROJECT="fractal_neurons"
VENV=".venv"
PYTHON="${VENV}/bin/python"
DATA_JSONL="data/english_corpus/english.jsonl"
RUN_DIR="runs/fsi_en_v1"
TOKENIZER_DIR="${RUN_DIR}/tokenizer"
CONFIG_YAML="configs/fsi_en_v1.yaml"
CONFIG_EVOL_YAML="configs/fsi_distill_v1.yaml"
LOG_DIR="${RUN_DIR}/logs"
EVAL_DIR="${RUN_DIR}/eval"
DISTILL_DIR="data/self_distilled_v1"
DISTILL_JSONL="${DISTILL_DIR}/distill.jsonl"
SWARM_LOG="${RUN_DIR}/swarm_logs.jsonl"
SWARM_PROMPT="Summarise the current capabilities of the latest Fractal-MoE checkpoint."
DEVICE="cuda"
RESUME_FLAG="${PIPELINE_RESUME:-0}"

# -------- ENV & DEPENDENCIES --------
if [ ! -d "${VENV}" ]; then
  python3 -m venv "${VENV}"
fi
# shellcheck disable=SC1090
source "${VENV}/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
export CUDA_LAUNCH_BLOCKING=0

pip install --upgrade pip
pip install -e . --quiet || true
pip install "torch>=2.2.0" "numpy>=1.24" "tqdm" "tokenizers>=0.15" \
            "datasets>=3.0.0" "huggingface_hub" "safetensors" \
            "einops" "transformers>=4.40" "pyyaml" "ujson" "orjson" --quiet

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${EVAL_DIR}" "${TOKENIZER_DIR}" "${DISTILL_DIR}"

if [ ! -f "${DATA_JSONL}" ]; then
  echo "FATAL: '${DATA_JSONL}' not found. Please prepare english.jsonl first." >&2
  exit 1
fi

# -------- 1) TOKENIZER --------
if [ "${USE_EXISTING_TOKENIZER:-0}" = "1" ]; then
  if [ -d "${TOKENIZER_DIR}" ]; then
    echo "[1/6] Skipping tokenizer build (USE_EXISTING_TOKENIZER=1); using existing ${TOKENIZER_DIR}"
  else
    echo "[1/6] USE_EXISTING_TOKENIZER=1 but ${TOKENIZER_DIR} missing; building tokenizer"
    ${PYTHON} -m ${PROJECT}.build_tokenizer \
      --corpus "${DATA_JSONL}" \
      --out "${TOKENIZER_DIR}" \
      --vocab-size 65536
  fi
else
  echo "[1/6] Building tokenizer → ${TOKENIZER_DIR}"
  ${PYTHON} -m ${PROJECT}.build_tokenizer \
    --corpus "${DATA_JSONL}" \
    --out "${TOKENIZER_DIR}" \
    --vocab-size 65536
fi

# -------- 2) TRAIN BASE MODEL --------
echo "[2/6] Training Fractal-MoE (config: ${CONFIG_YAML})"
if [ "${RESUME_FLAG}" != "0" ]; then
  RESUME_ARG="--resume"
else
  RESUME_ARG=""
fi

${PYTHON} -m ${PROJECT}.train --config "${CONFIG_YAML}" --device "${DEVICE}" ${RESUME_ARG}

RUN_BASE_DIR="${RUN_DIR}"
BASE_CKPT=$(RUN_BASE_DIR="${RUN_BASE_DIR}" ${PYTHON} <<'PY'
import glob
import os

run_dir = os.environ.get("RUN_BASE_DIR", "")
run_dir = os.path.abspath(os.path.expanduser(run_dir))
paths = sorted(glob.glob(os.path.join(run_dir, "final_*.pt")))
if not paths:
    paths = sorted(glob.glob(os.path.join(run_dir, "ema_final_*.pt")))
if paths:
    candidate = paths[-1]
elif os.path.isfile(os.path.join(run_dir, "latest.pt")):
    candidate = os.path.join(run_dir, "latest.pt")
else:
    candidate = ""
print(candidate)
PY
)
BASE_CKPT=${BASE_CKPT//$'\n'/}

if [ -z "${BASE_CKPT}" ] || [ ! -f "${BASE_CKPT}" ]; then
  echo "WARN: No checkpoints produced; aborting pipeline." >&2
  exit 1
fi
echo "[OK] Using checkpoint: ${BASE_CKPT}"

# -------- 3) SELF-DISTILLATION --------
echo "[3/6] Self-distillation → ${DISTILL_JSONL}"
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

# -------- 4) SWARM ORCHESTRATION --------
echo "[4/6] Swarm orchestration"
${PYTHON} -m ${PROJECT}.orchestrator \
  --ckpt "${BASE_CKPT}" \
  --prompt "${SWARM_PROMPT}" \
  --rounds 4 \
  --device "${DEVICE}" \
  --log "${SWARM_LOG}" || true

# -------- 5) EVALUATION --------
echo "[5/6] Evaluation harness"
${PYTHON} -m ${PROJECT}.eval_harness \
  --ckpt "${BASE_CKPT}" \
  --device "${DEVICE}" \
  --output "${EVAL_DIR}/report.json"

# -------- 6) EVOLUTION / FINETUNE ROUND --------
echo "[6/6] Evolution finetune round (config: ${CONFIG_EVOL_YAML})"
${PYTHON} -m ${PROJECT}.train \
  --config "${CONFIG_EVOL_YAML}" \
  --device "${DEVICE}" \
  --resume

echo "✅ Pipeline complete. Outputs in ${RUN_DIR}"
