source set_params.sh
./less/scripts/train/warmup_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$WARMUP_JOB_NAME" | tee log.txt
