#!/bin/bash

set -euo pipefail

# Function to print an error message and exit
error_handler() {
    echo "❌ Error occurred at line $1: $2"
    exit 1
}

# Trap any error, call error_handler with line number and last command
trap 'error_handler $LINENO "$BASH_COMMAND"' ERR

source set_params.sh

export NUM_ITERATIONS=10
export SET_PARAMS=0

for ((i=0; i<NUM_ITERATIONS; i++)); do
    echo "▶️ Starting iteration $i..."
    export ITERATION=${i}
    if [ "$i" -ne 0 ]; then
        export TRAIN_MODEL_PATH="../out/${TRAIN_JOB_NAME}"
        export SELECT_MODEL_PATH=${TRAIN_MODEL_PATH}
    fi

    export SELECT_OUTPUT_PATH=../grads/llama2-7b-p${PERCENTAGE}-i${NUM_ITERATIONS}-${i}-lora-seed${DATA_SEED}/${TASK}-ckpt${CKPT}-sgd

    echo "🔍 Running select.sh..."
    ./select.sh

    export SCORE_FILE="sorted_p${PERCENTAGE}_i${NUM_ITERATIONS}_${i}.csv"
    export TRAIN_FILES="../selected_data/mmlu/top_p${PERCENTAGE}_i${NUM_ITERATIONS}_${i}.jsonl"
    echo "Set training params SCORE_FILE='$SCORE_FILE', TRAIN_FILES='$TRAIN_FILES'"
    echo "🧠 Running train.sh..."
    ./train.sh

#    rm -rf $SELECT_OUTPUT_PATH
    echo "✅ Iteration $i completed successfully."
done

export SET_PARAMS=1
echo "🎉 All iterations completed."
