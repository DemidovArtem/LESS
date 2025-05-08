#!/bin/bash

set -euo pipefail

# Function to print an error message and exit
error_handler() {
    echo "❌ Error occurred at line $1: $2"
    exit 1
}

# Trap any error, call error_handler with line number and last command
trap 'error_handler $LINENO "$BASH_COMMAND"' ERR

cd /workspace/LESS || return 1

source set_params.sh

export NUM_ITERATIONS=1


export SET_PARAMS=0

for ((i=0; i<NUM_ITERATIONS; i++)); do
    echo "▶️ Starting iteration $i..."
    export ITERATION=${i}
    export SKIP_SELECT_MODEL_CHOICE=0
    if [ "$i" -ne "0" ]; then
        export TRAIN_MODEL_PATH="../out/${TRAIN_JOB_NAME}"
        export SELECT_MODEL_PATH=${TRAIN_MODEL_PATH}
        export SKIP_SELECT_MODEL_CHOICE=1
    fi

    echo "🔍 Running select.sh..."
    ./select.sh


    export SCORE_FILE="sorted_p${PERCENTAGE}_i${NUM_ITERATIONS}_${i}.csv"
    export TRAIN_FILES="/workspace/selected_data/mmlu/top_p${PERCENTAGE}_i${NUM_ITERATIONS}_${i}.jsonl"
    echo "Set training params SCORE_FILE='$SCORE_FILE', TRAIN_FILES='$TRAIN_FILES'"
    echo "🧠 Running train.sh..."
    ./train.sh

    echo "Running evaluate.sh..."
    ./evaluate.sh
    echo "Uploading data to backblaze..."
    ./upload_to_b2.sh
    echo "✅ Iteration $i completed successfully."
done

export SET_PARAMS=1
export SKIP_SELECT_MODEL_CHOICE=0
echo "🎉 All iterations completed."
