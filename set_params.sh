export TARGET_TASK_NAME="mmlu"
export EXPERIMENT_POSTFIX="iterative-top-k"

# warmup

export DATA_DIR=../data
export PERCENTAGE=0.05 # percentage of the full data to train, you can specify the training file you want to use in the script
export DATA_SEED=3
export MODEL_PATH=meta-llama/Llama-2-7b-hf
export WARMUP_JOB_NAME=llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}

# build datastore

export CKPT=422
export TRAINING_DATA_NAME=dolly
export TRAINING_DATA_FILE=../data/train/processed/${TRAINING_DATA_NAME}/${TRAINING_DATA_NAME}_data.jsonl # when changing data name, change the data path accordingly
export GRADIENT_TYPE="adam"
export DATASTORE_MODEL_PATH=../out/llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}/checkpoint-${CKPT}
export BUILD_OUTPUT_PATH=../grads/llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}
export BUILD_DIMS="8192"

# select data
## grads step
export SELECT_MODEL_PATH=${MODEL_PATH}
export TASK=mmlu
export SELECT_OUTPUT_PATH=../../grads/llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}/${TASK}-ckpt${CKPT}-sgd # for validation data, we always use sgd
export SELECT_DIMS="4096 8192" # We use 8192 as our default projection dimension
## matching step
export MATCHING_DIM=8192
export MATCHING_GRADIENT_PATH=../../grads/llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}/dim${MATCHING_DIM}
export TRAIN_FILE_NAMES="flan_v2 cot dolly oasst1"
export CKPTS="422 845 1268 1688" # checkpointing index
export CHECKPOINT_WEIGHTS="5.005931e-01 3.333333e-01 1.660735e-01 0.000000e+00" # average lr of the epoch
export VALIDATION_GRADIENT_PATH=../../grads/llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}/${TASK}-ckpt${CKPT}-sgd/dim${MATCHING_DIM}
export SELECTED_DATA_OUTPUT_PATH="../selected_data"


# train
export SCORE_SCALING='none'
export TRAIN_FILES=../data/selected_data/${TARGET_TASK_NAME}/top_p${PERCENTAGE}.jsonl
export TRAIN_JOB_NAME=llama2-7b-less-p${PERCENTAGE}-lora-${EXPERIMENT_POSTFIX}
export TRAIN_MODEL_PATH=${MODEL_PATH}


