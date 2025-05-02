#!/bin/bash

source set_params.sh

export NUM_ITERATIONS=10

export SELECT_OUTPUT_PATH=../grads/llama2-7b-p${PERCENTAGE}-i${NUM_ITERATIONS}-lora-seed${DATA_SEED}/${TASK}-ckpt${CKPT}-sgd
export SET_PARAMS=0

for ((i=0; i<NUM_ITERATIONS; i++)); do
    if [ "$i" -ne 0 ]; then
        export TRAIN_MODEL_PATH="../out/${TRAIN_JOB_NAME}"
    fi
    export SELECT_MODEL_PATH=${TRAIN_MODEL_PATH}
    ./select.sh
    ./train.sh

done

export SET_PARAMS=1