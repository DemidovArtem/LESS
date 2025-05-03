set -e
if [ "$SET_PARAMS" = "1" ]; then
    source set_params.sh
else
    echo "Don't use set_params as source for train"
fi

./less/scripts/train/lora_train.sh "$TRAIN_FILES" "$TRAIN_MODEL_PATH" "$TRAIN_JOB_NAME" "$TARGET_TASK_NAME" "$SCORE_SCALING" "$SCORE_FILE" | tee log.txt
