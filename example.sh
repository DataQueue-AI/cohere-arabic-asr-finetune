#!/bin/bash
set -euo pipefail

PROJECT_DIR="/workspace/Dana/finetune-cohere"
FINETUNE_SCRIPT="/workspace/Dana/finetune-cohere/train_cohere_full.py"
MODEL_PATH="/workspace/Dana/finetune-cohere/models/cohere-transcribe-arabic-07-2026"
TRAIN_CSV="/workspace/Dana/org/stt-data/Dana/Data/CSVs_4_training/digitsinWords/15kbatches_1278silence_10kcs_15crtvai_ABBatches_ArabBankCallsBatches_emirates-dialect-speech_speed_aug_1.1xFew_kejue-common-voice.csv"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MASTER_PORT="${MASTER_PORT:-29620}"
OPTIMIZER="${OPTIMIZER:-adamw_bnb_8bit}"
RUN_NAME="cohere-transcribe-arabic-15kbatches_1278silence_10kcs_15crtvai_ABBatches_ArabBankCallsBatches_emirates-dialect-speech_speed_aug_1.1xFew_kejue-common-voice"
TS=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT="/root/cache"
OUT_DIR="${OUTPUT_ROOT}/results_${RUN_NAME}_${TS}"
LOG_DIR="${OUTPUT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_${TS}.log"
RESUME_CKPT="${RESUME_CKPT:-}"
USE_BF16="${USE_BF16:-1}"
GCS_DEST="gs://speech-annotation-source/stt-data/${RUN_NAME}"

if [ -d "/workspace/Dana/nvenv" ]; then
  source /workspace/Dana/nvenv/bin/activate
elif [ -d "/workspace/stt-data/Dana/teslavenv" ]; then
  source /workspace/stt-data/Dana/teslavenv/bin/activate
fi

if [ ! -f "${TRAIN_CSV}" ]; then
  echo "Train CSV not found: ${TRAIN_CSV}"
  exit 1
fi

if [ ! -f "${FINETUNE_SCRIPT}" ]; then
  echo "finetune.py not found: ${FINETUNE_SCRIPT}"
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

# Start checkpoint uploader in background, if present
UPLOADER_SCRIPT="/workspace/Dana/finetune-cohere/ckpt_uploader.sh"
if [ -f "${UPLOADER_SCRIPT}" ]; then
  UPLOADER_LOG="${LOG_DIR}/${RUN_NAME}_${TS}_uploader.log"
  setsid nohup bash "${UPLOADER_SCRIPT}" "${OUT_DIR}" "${GCS_DEST}" \
    > "${UPLOADER_LOG}" 2>&1 < /dev/null &
  UPLOADER_PID=$!
  echo "Uploader PID=${UPLOADER_PID} log=${UPLOADER_LOG}"
else
  echo "Uploader script not found, skipping checkpoint upload: ${UPLOADER_SCRIPT}"
fi

echo "Starting ${RUN_NAME}"
echo "GPU_IDS=${GPU_IDS}"
echo "TRAIN_CSV=${TRAIN_CSV}"
echo "OUT_DIR=${OUT_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "OPTIMIZER=${OPTIMIZER}"
if [ -n "${RESUME_CKPT}" ]; then
  echo "RESUME_CKPT=${RESUME_CKPT}"
fi

EXTRA_ARGS=()
if [ -n "${RESUME_CKPT}" ]; then
  EXTRA_ARGS+=(--resume_from_checkpoint "${RESUME_CKPT}")
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
  --output_dir "${OUT_DIR}" \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2.5e-6 \
  --warmup_steps 50 \
  --num_workers 0 \
  --min_audio_seconds 0.1 \
  --max_audio_seconds 30.0 \
  --save_total_limit 3 \
  --save_only_model \
  --final_eval_clips 0 \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"

# chmod +x /workspace/Dana/finetune-cohere/launch_cohere_arabic_15kbatches_1278silence_10kcs_15crtvai_ABBatches_ArabBankCallsBatches_emirates-dialect-speech_speed_aug_1.1xFew_kejue-common-voice.sh
# setsid nohup bash /workspace/Dana/finetune-cohere/launch_cohere_arabic_15kbatches_1278silence_10kcs_15crtvai_ABBatches_ArabBankCallsBatches_emirates-dialect-speech_speed_aug_1.1xFew_kejue-common-voice.sh \
# > /workspace/cache/cohere_arabic_15kbatches_1278silence_10kcs_15crtvai_ABBatches_ArabBankCallsBatches_emirates-dialect-speech_speed_aug_1.1xFew_kejue-common-voice.log 2>&1 < /dev/null &
# echo "PID: $!"
