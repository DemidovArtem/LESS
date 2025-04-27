# warmup

export DATA_DIR=../data
export PERCENTAGE=0.05 # percentage of the full data to train, you can specify the training file you want to use in the script
export DATA_SEED=3
export MODEL_PATH=meta-llama/Llama-2-7b-hf
export WARMUP_JOB_NAME=llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}


# build datastore

export CKPT=25
export TRAINING_DATA_NAME=dolly
export TRAINING_DATA_FILE=../data/train/processed/${TRAINING_DATA_NAME}/${TRAINING_DATA_NAME}_data.jsonl # when changing data name, change the data path accordingly
export GRADIENT_TYPE="adam"
export DATASTORE_MODEL_PATH=../out/llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}/checkpoint-${CKPT}
export BUILD_OUTPUT_PATH=../grads/llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}
export BUILD_DIMS="8192"

# select data
export TASK=mmlu
export SELECT_OUTPUT_PATH=../grads/llama2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}/${TASK}-ckpt${CKPT}-sgd # for validation data, we always use sgd
export SELECT_DIMS="4096 8192" # We use 8192 as our default projection dimension

# train

export TARGET_TASK_NAME="tydiqa"
export TRAIN_FILES=../data/selected_data/${TARGET_TASK_NAME}/top_p0.05.jsonl
export TRAIN_JOB_NAME=llama2-7b-less-p${PERCENTAGE}-lora-weighted


