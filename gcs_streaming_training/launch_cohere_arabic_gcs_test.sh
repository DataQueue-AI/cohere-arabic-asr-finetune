#!/bin/bash
set -euo pipefail

PROJECT_DIR="/workspace/STT-Training-Data-sep2026/gcs_streaming_training"
FINETUNE_SCRIPT="${PROJECT_DIR}/train_cohere_full.py"
MODEL_PATH="/workspace/STT-Training-Data-sep2026/finetune-cohere/models/cohere-transcribe-arabic-07-2026"
TRAIN_CSV="/workspace/STT-Training-Data-sep2026/org/stt-data/Dana/Data/CSVs_4_training/digitsinWords/15kbatches_1278silence_10kcs_15crtvai_ABBatches_ArabBankCallsBatches_emirates-dialect-speech_speed_aug_1.1xFew_gcs.csv"
EVAL_CSV=""

GPU_IDS="${GPU_IDS:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MASTER_PORT="${MASTER_PORT:-29620}"
OPTIMIZER="${OPTIMIZER:-adamw_bnb_8bit}"
RUN_NAME="cohere_arabic_gcs_streaming_test"
TS=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT="/workspace/cache"
OUT_DIR="${OUTPUT_ROOT}/results_${RUN_NAME}_${TS}"
LOG_DIR="${OUTPUT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_${TS}.log"
RESUME_CKPT="${RESUME_CKPT:-}"
USE_BF16="${USE_BF16:-1}"
GCS_DEST="gs://speech-annotation-source/stt-data/${RUN_NAME}"

# how many optimizer steps to run; -1 = full num_train_epochs (use a small value
# like 50 for a quick smoke test of the GCS streaming path).
MAX_STEPS="${MAX_STEPS:--1}"

# Streaming cache for any gs:// audio paths in TRAIN_CSV/EVAL_CSV (see gcs_cache.py).
# Not the whole dataset needs to fit here -- only the working set; oldest-accessed
# files are evicted once the cache exceeds GCS_CACHE_MAX_GB.
GCS_CACHE_DIR="${GCS_CACHE_DIR:-/workspace/gcs_audio_cache}"
GCS_CACHE_MAX_GB="${GCS_CACHE_MAX_GB:-80}"
GCS_KEY_FILE="${GCS_KEY_FILE:-${PROJECT_DIR}/google.json}"
NUM_WORKERS="${NUM_WORKERS:-8}"

NVENV_DIR="/workspace/nvenv"
if [ -d "${NVENV_DIR}" ]; then
  source "${NVENV_DIR}/bin/activate"
fi

if [ ! -f "${TRAIN_CSV}" ]; then
  echo "Train CSV not found: ${TRAIN_CSV}"
  exit 1
fi

if [ -n "${EVAL_CSV}" ] && [ ! -f "${EVAL_CSV}" ]; then
  echo "Eval CSV not found: ${EVAL_CSV}"
  exit 1
fi

if [ ! -f "${FINETUNE_SCRIPT}" ]; then
  echo "finetune.py not found: ${FINETUNE_SCRIPT}"
  exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
  echo "Model path not found: ${MODEL_PATH}"
  exit 1
fi

if [ -n "${RESUME_CKPT}" ] && [ ! -d "${RESUME_CKPT}" ]; then
  echo "Resume checkpoint not found: ${RESUME_CKPT}"
  exit 1
fi

while ss -ltn | awk '{print $4}' | grep -q ":${MASTER_PORT}$"; do
  MASTER_PORT=$((MASTER_PORT + 1))
done

mkdir -p "${OUTPUT_ROOT}" "${OUT_DIR}" "${LOG_DIR}"
cd "${PROJECT_DIR}"

# Start checkpoint uploader in background
UPLOADER_LOG="${LOG_DIR}/${RUN_NAME}_${TS}_uploader.log"
setsid nohup bash "${PROJECT_DIR}/ckpt_uploader.sh" "${OUT_DIR}" "${GCS_DEST}" \
  > "${UPLOADER_LOG}" 2>&1 < /dev/null &
UPLOADER_PID=$!
echo "Uploader PID=${UPLOADER_PID} log=${UPLOADER_LOG}"

echo "Starting ${RUN_NAME}"
echo "GPU_IDS=${GPU_IDS}"
echo "TRAIN_CSV=${TRAIN_CSV}"
echo "EVAL_CSV=${EVAL_CSV:-<none>}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "OUT_DIR=${OUT_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "OPTIMIZER=${OPTIMIZER}"
echo "MAX_STEPS=${MAX_STEPS}"
if [ -n "${RESUME_CKPT}" ]; then
  echo "RESUME_CKPT=${RESUME_CKPT}"
fi

EXTRA_ARGS=()
if [ -n "${RESUME_CKPT}" ]; then
  EXTRA_ARGS+=(--resume_from_checkpoint "${RESUME_CKPT}")
fi
if [ -n "${EVAL_CSV}" ]; then
  EXTRA_ARGS+=(--eval_csv_file "${EVAL_CSV}")
fi

PRECISION_ARGS=()
if [ "${USE_BF16}" = "1" ]; then
  PRECISION_ARGS+=(--bf16)
fi

torchrun --nproc_per_node=4 --master_port=${MASTER_PORT} "${FINETUNE_SCRIPT}" \
  --model_id "${MODEL_PATH}" \
  --language ar \
  --csv_file "${TRAIN_CSV}" \
  --num_train_epochs 2 \
  --max_steps "${MAX_STEPS}" \
  --output_dir "${OUT_DIR}" \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2.5e-6 \
  --warmup_steps 50 \
  --eval_steps 500 \
  --num_workers "${NUM_WORKERS}" \
  --min_audio_seconds 0.1 \
  --max_audio_seconds 30.0 \
  --save_total_limit 3 \
  --save_only_model \
  --final_eval_clips 0 \
  --gcs_cache_dir "${GCS_CACHE_DIR}" \
  --gcs_cache_max_gb "${GCS_CACHE_MAX_GB}" \
  --gcs_key_file "${GCS_KEY_FILE}" \
  "${EXTRA_ARGS[@]}" 2>&1 \
  | while IFS= read -r line; do printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"; done \
  | tee "${LOG_FILE}"

# chmod +x /workspace/STT-Training-Data-sep2026/gcs_streaming_training/launch_cohere_arabic_gcs_test.sh
# MAX_STEPS=50 setsid nohup bash /workspace/STT-Training-Data-sep2026/gcs_streaming_training/launch_cohere_arabic_gcs_test.sh \
# > /workspace/STT-Training-Data-sep2026/gcs_streaming_training/cohere_arabic_gcs_streaming_test.log 2>&1 < /dev/null &
# echo "PID: $!"
