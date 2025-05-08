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
if [[ "${SET_PARAMS:-}" == "1" ]]; then
    source set_params.sh
else
    echo "⚠️  SET_PARAMS is not 1 — skipping set_params.sh"
fi

# Check required env variables
: "${TASK:?TASK variable is not set}"
: "${TRAIN_MODEL_PATH:?TRAIN_MODEL_PATH variable is not set}"


cd /workspace/LESS/evaluation || return 1
#export PYTHONPATH=/workspace/open-instruct:$PYTHONPATH

export EVAL_PATH=$TRAIN_MODEL_PATH
[[ "${TRAIN_MODEL_PATH}" != /* ]] && export EVAL_PATH="result_${TRAIN_MODEL_PATH}"

# Build command
cmd="python -m eval.${TASK}.run_eval \
    --data_dir /workspace/data/eval/${TASK} \
    --save_dir ${TRAIN_MODEL_PATH}/eval/${ITERATION} \
    --model ${TRAIN_MODEL_PATH} \
    --tokenizer ${TRAIN_MODEL_PATH} \
    --eval_batch_size 1 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format"

if [ "$TASK" != "mmlu" ]; then
    cmd+=" \
    --n_shot 1 \
    --max_num_examples_per_lang 200 \
    --max_context_length 1024 \
    --convert_to_bf16"
fi

# Run the evaluation
echo "🧪 Running evaluation for task: $TASK"
eval "$cmd"
