#!/bin/bash

set -euo pipefail

# Error handling function
error_handler() {
    echo "❌ Error occurred at line $1: $2"
    exit 1
}

# Trap for catching errors
trap 'error_handler $LINENO "$BASH_COMMAND"' ERR

# Parameter handling
if [ "${SET_PARAMS:-}" = "1" ]; then
    source set_params.sh
else
    echo "⚠️  SET_PARAMS is not 1 — skipping set_params.sh"
fi

# Step 1: Gradients step
echo "🔍 Step 1: Running grad extraction..."
for CKPT in $CKPTS; do
    export SELECT_MODEL_PATH=../out/llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}/checkpoint-${CKPT}
    export SELECT_OUTPUT_PATH=../grads/llama2-7b-p${PERCENTAGE}-${EXPERIMENT_POSTFIX}-i${NUM_ITERATIONS}-${ITERATION}-lora-seed${DATA_SEED}/${TASK}-ckpt${CKPT}-sgd

    echo "🔍 Grad extraction for CKPT=${CKPT}..."
    echo "SELECT_MODEL_PATH=${SELECT_MODEL_PATH}"
    echo "SELECT_OUTPUT_PATH=${SELECT_OUTPUT_PATH}"
    ./less/scripts/get_info/grad/get_eval_lora_grads.sh "$TASK" "$DATA_DIR" "$SELECT_MODEL_PATH" "$SELECT_OUTPUT_PATH" "$SELECT_DIMS" | tee log.txt
done


# Step 2: Matching step
echo "🔗 Step 2: Running matching..."
./less/scripts/data_selection/matching.sh \
    "$MATCHING_GRADIENT_PATH" "$TRAIN_FILE_NAMES" "$CKPTS" "$CHECKPOINT_WEIGHTS" \
    "$VALIDATION_GRADIENT_PATH" "$TARGET_TASK_NAME" "$SELECTED_DATA_OUTPUT_PATH" "$NUM_ITERATIONS"

# Step 3: Top-k selection step
echo "🎯 Step 3: Selecting top-k data..."
python3 -m less.data_selection.write_selected_data \
    --target_task_names "${TARGET_TASK_NAME}" \
    --train_file_names $TRAIN_FILE_NAMES \
    --train_files $(for name in ${TRAIN_FILE_NAMES}; do echo -n "../data/train/processed/${name}/${name}_data.jsonl "; done) \
    --output_path "$SELECTED_DATA_OUTPUT_PATH" \
    --percentage "$PERCENTAGE" \
    --num_iterations "$NUM_ITERATIONS" \
    --iteration "$ITERATION"

echo "✅ select.sh completed successfully."
