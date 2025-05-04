#!/bin/bash

set -euo pipefail

# Error handling
error_handler() {
    echo "❌ Error at line $1: $2"
    exit 1
}
trap 'error_handler $LINENO "$BASH_COMMAND"' ERR

# Validate required variables
: "${TRAIN_JOB_NAME:?TRAIN_JOB_NAME is not set}"

# B2 settings
B2_BUCKET="less-paper"
B2_BASE_PATH="artem"
MODEL_FOLDER="../out/$TRAIN_JOB_NAME"

echo "📁 Syncing model directory: $MODEL_FOLDER"
b2 sync "$MODEL_FOLDER" "b2://${B2_BUCKET}/${B2_BASE_PATH}/$(basename "${TRAIN_JOB_NAME}_${ITERATION}")"

echo "📁 Syncing selected_data directory: ../selected_data/"
b2 sync ../selected_data "b2://${B2_BUCKET}/${B2_BASE_PATH}/selected_data"

echo "✅ Upload to B2 complete."
