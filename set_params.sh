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

export TARGET_TASK_NAME="mmlu"

export EXPERIMENT_POSTFIX="first-checkpoint-iterative"
export CKPTS="422" # checkpointing index
export CHECKPOINT_WEIGHTS="5.005931e-01" # average lr of the epoch

export SKIP_SELECT_MODEL_CHOICE=0
export TASK=${TARGET_TASK_NAME}

# warmup

export DATA_DIR=/workspace/data
export PERCENTAGE=0.05 # percentage of the full data to train, you can specify the training file you want to use in the script
export DATA_SEED=3
export MODEL_PATH=meta-llama/Llama-2-7b-hf
export WARMUP_JOB_NAME=llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}

#export CKPT=422 - need to iterate over all available checkpoints,
# so all the steps of build datastore and select data should be wrapped in a for loop

# build datastore

#export TRAINING_DATA_NAME=dolly - need to iterate over it too
#export TRAINING_DATA_FILE=/workspace/data/train/processed/${TRAINING_DATA_NAME}/${TRAINING_DATA_NAME}_data.jsonl # when changing data name, change the data path accordingly
export GRADIENT_TYPE="adam"
#export DATASTORE_MODEL_PATH=/workspace/out/llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}/checkpoint-${CKPT}
#export BUILD_OUTPUT_PATH=/workspace/grads/llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}
export BUILD_DIMS="8192"

# select data
## grads step
#export SELECT_MODEL_PATH=${DATASTORE_MODEL_PATH}
#export SELECT_OUTPUT_PATH=/workspace/grads/llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}/${TASK}-ckpt${CKPT}-sgd # for validation data, we always use sgd
export SELECT_DIMS="8192" # We use 8192 as our default projection dimension
## matching step
export MATCHING_DIM=8192
export MATCHING_GRADIENT_PATH=/workspace/grads/llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}/{}-ckpt{}-${GRADIENT_TYPE}/dim${MATCHING_DIM}
export TRAIN_FILE_NAMES="flan_v2 cot dolly oasst1"
export VALIDATION_GRADIENT_PATH=/workspace/grads/llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}/{}-ckpt{}-sgd/dim${MATCHING_DIM}
export SELECTED_DATA_OUTPUT_PATH="/workspace/selected_data"


# train
export SCORE_SCALING='none'
export TRAIN_FILES=/workspace/selected_data/${TARGET_TASK_NAME}/top_p${PERCENTAGE}.jsonl
export TRAIN_JOB_NAME=llama2-7b-less-p${PERCENTAGE}-lora-${EXPERIMENT_POSTFIX}-${TASK}
export TRAIN_MODEL_PATH=${MODEL_PATH}


