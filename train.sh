source set_params.sh
./less/scripts/train/lora_train.sh "$TRAIN_FILES" "$MODEL_PATH" "$TRAIN_JOB_NAME" "$TARGET_TASK_NAME" "$SCORE_SCALING" | tee log.txt
