if [ "$SET_PARAMS" = "1" ]; then
    source set_params.sh
else
    echo "Don't use set_params as source for select"
fi


# grads step
./less/scripts/get_info/grad/get_eval_lora_grads.sh "$TASK" "$DATA_DIR" "$SELECT_MODEL_PATH" "$SELECT_OUTPUT_PATH" "$SELECT_DIMS" | tee log.txt

# matching step
./less/scripts/data_selection/matching.sh "$MATCHING_GRADIENT_PATH" "$TRAIN_FILE_NAMES" "$CKPTS" "$CHECKPOINT_WEIGHTS" "$VALIDATION_GRADIENT_PATH" "$TARGET_TASK_NAMES" "$SELECTED_DATA_OUTPUT_PATH" "$NUM_ITERATIONS"

# Top-k choice step
python3 -m less.data_selection.write_selected_data \
--target_task_names ${TARGET_TASK_NAMES} \
--train_file_names ${TRAIN_FILE_NAMES} \
--train_files $(for name in ${TRAIN_FILE_NAMES}; do echo -n "../data/train/processed/${name}/${name}_data.jsonl "; done) \
--output_path $SELECTED_DATA_OUTPUT_PATH \
--percentage $PERCENTAGE \
--num_iterations ${NUM_ITERATIONS}