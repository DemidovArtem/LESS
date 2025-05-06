#!/bin/bash

set -euo pipefail

# Error handling function
error_handler() {
    echo "❌ Error occurred at line $1: $2"
    exit 1
}

# Trap to catch and report errors
trap 'error_handler $LINENO "$BASH_COMMAND"' ERR

# Parameter handling
if [ "${SET_PARAMS:-}" = "1" ]; then
    source set_params.sh
else
    echo "⚠️  SET_PARAMS is not 1 — skipping set_params.sh"
fi

# Training step
echo "🧠 Starting training with LoRA..."
./less/scripts/train/lora_train.sh \
    "$TRAIN_FILES" \
    "$TRAIN_MODEL_PATH" \
    "$TRAIN_JOB_NAME" \
    "$TARGET_TASK_NAME" \
    "$SCORE_SCALING" \
    "$SCORE_FILE" \
    "$NUM_ITERATIONS" \
    "$ITERATION" | tee log.txt

echo "✅ train.sh completed successfully."
