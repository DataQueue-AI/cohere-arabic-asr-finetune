#!/bin/bash
# Watches OUT_DIR for new checkpoint-* dirs and syncs each to GCS. Run in background alongside training.
set -uo pipefail

OUT_DIR="${1:?Usage: $0 <out_dir> <gcs_dest>}"
GCS_DEST="${2:?Usage: $0 <out_dir> <gcs_dest>}"
POLL_INTERVAL=60
UPLOADED_LOG="${OUT_DIR}/.uploaded_ckpts"
LOG_FILE="${OUT_DIR}/.uploader.log"

export GOOGLE_APPLICATION_CREDENTIALS="/workspace/google.json"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"; }

log "Uploader started. Watching: ${OUT_DIR} -> ${GCS_DEST}"
touch "${UPLOADED_LOG}"

while true; do
  for ckpt_dir in "${OUT_DIR}"/checkpoint-*/; do
    [ -d "${ckpt_dir}" ] || continue
    ckpt_name=$(basename "${ckpt_dir}")
    if grep -qxF "${ckpt_name}" "${UPLOADED_LOG}"; then
      continue
    fi
    # Wait until the checkpoint write is complete (no files modified in last 10s)
    last_mod=$(find "${ckpt_dir}" -newer "${ckpt_dir}/trainer_state.json" 2>/dev/null | wc -l)
    if [ ! -f "${ckpt_dir}/trainer_state.json" ]; then
      continue  # still writing
    fi
    log "Uploading ${ckpt_name} (background) ..."
    # mark before launch so the loop doesn't re-trigger; remove on failure
    echo "${ckpt_name}" >> "${UPLOADED_LOG}"
    (
      if gsutil -m rsync -r "${ckpt_dir}" "${GCS_DEST}/${ckpt_name}/" >> "${LOG_FILE}" 2>&1; then
        log "Done: ${ckpt_name}"
      else
        log "ERROR uploading ${ckpt_name}; removing from uploaded list so it retries"
        sed -i "/^${ckpt_name}$/d" "${UPLOADED_LOG}"
      fi
    ) &
  done
  sleep "${POLL_INTERVAL}"
done
