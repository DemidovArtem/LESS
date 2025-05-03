#!/bin/bash

set -e  # Exit immediately if any command fails

source set_params.sh

export NUM_ITERATIONS=10
export SET_PARAMS=0

for ((i=0; i<NUM_ITERATIONS; i++)); do
    if [ "$i" -ne 0 ]; then
        export TRAIN_MODEL_PATH="../out/${TRAIN_JOB_NAME}"
    fi
    export SELECT_MODEL_PATH=${TRAIN_MODEL_PATH}
    export SELECT_OUTPUT_PATH=../grads/llama2-7b-p${PERCENTAGE}-i${NUM_ITERATIONS}-${i}-lora-seed${DATA_SEED}/${TASK}-ckpt${CKPT}-sgd

    ./select.sh
    export SCORE_FILE=sorted_p0.05_i${NUM_ITERATIONS}_${i}.csv
    export TRAIN_FILES=../selected_data/mmlu/top_p0.05_i${NUM_ITERATIONS}_${i}.jsonl
    ./train.sh
#    rm -rf $SELECT_OUTPUT_PATH
done

export SET_PARAMS=1
