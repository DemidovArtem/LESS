#!/usr/bin/env python
# coding=utf-8
import logging
import os
import random
import sys

import datasets
import torch.distributed as dist
import transformers
from less.data_selection.get_training_dataset import get_training_dataset
from less.train.data_arguments import DataArguments, get_data_statistics
from less.train.model_arguments import ModelArguments, add_padding_to_tokenizer
from less.train.training_arguments import TrainingArguments
from less.train.weighting_strategy import add_weights
# from instruction_tuning.train.lora_trainer import LoRAFSDPTrainer, Trainer
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.optim.lr_scheduler import LambdaLR
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, HfArgumentParser, Trainer,
                          set_seed)

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WeightedTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        from transformers.modeling_utils import unwrap_model
        from transformers.utils.import_utils import is_peft_available
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        weights = None
        if 'weights' in inputs:
            weights = inputs.pop('weights')
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if weights is not None:
            weights = weights.squeeze()
            loss *= weights
        else:
            print('Weight is not defined')
        return (loss, outputs) if return_outputs else loss

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        optimizer = optimizer or self.optimizer

        # Total across all iterations (virtual total steps)
        num_iterations = getattr(self.args, "num_iterations", None)
        current_iteration = getattr(self.args, "iteration", 0)

        if num_iterations is None:
            # Use default Hugging Face behavior
            return super().create_scheduler(num_training_steps, optimizer)

        # Compute total "virtual" steps
        total_virtual_steps = num_training_steps * num_iterations
        current_virtual_step_offset = num_training_steps * current_iteration

        warmup_steps = int(self.args.warmup_ratio * total_virtual_steps)

        def lr_lambda(current_step):
            # Convert local step into global virtual step
            global_step = current_virtual_step_offset + current_step

            if global_step < warmup_steps:
                return float(global_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_virtual_steps - global_step) / float(max(1, total_virtual_steps - warmup_steps))
            )

        self.lr_scheduler = LambdaLR(optimizer, lr_lambda)
        return self.lr_scheduler


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # Load training dataset
    train_dataset = get_training_dataset(data_args.train_files,
                                         tokenizer=tokenizer,
                                         max_seq_length=data_args.max_seq_length,
                                         sample_percentage=data_args.percentage,
                                         seed=data_args.sample_data_seed)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, torch_dtype=model_args.torch_dtype)
    add_padding_to_tokenizer(tokenizer)

    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        # if you load lora model and resize the token embeddings, the requires_grad flag is set to True for embeddings
        if isinstance(model, PeftModel):
            model.get_input_embeddings().weight.requires_grad = False
            model.get_output_embeddings().weight.requires_grad = False

    if not isinstance(model, PeftModel) and model_args.lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        logger.info(
            f"Applied LoRA to model."
        )
        model.print_trainable_parameters()

        # for checkpointing
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    get_data_statistics(train_dataset)
    weights_map = {}
    if data_args.target_task_name:
        train_dataset, weights_map = add_weights(
            train_dataset=train_dataset,
            data_args=data_args,
        )
    if "dataset" in train_dataset.features:
        train_dataset = train_dataset.remove_columns(
            ["dataset", "id", "messages"])

    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(
            f"Sample {index} of the training set: {train_dataset[index]}.")

    model_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable model_params: {model_params}")

    analysis_dataset = None
    if training_args.analysis_mode:
        from less.data_selection.get_validation_dataset import get_dataset
        analysis_dataset = get_dataset(training_args.analysis_dataset,
                                       data_dir=data_args.data_dir,
                                       tokenizer=tokenizer,
                                       max_length=data_args.max_seq_length)

        if data_args.target_task_name:
            weights_list = [
                weights_map[
                    (
                        f"{analysis_dataset[index]['dataset']}_influence_score.pt",
                        str(analysis_dataset[index]['id'].split('_')[-1])
                    )
                ]
                for index in range(len(analysis_dataset))
            ]
            analysis_dataset = analysis_dataset.add_column("weights", weights_list)
            print('Weights were added to analysis_dataset!')
            print(analysis_dataset[0])

    if dist.is_initialized() and dist.get_rank() == 0:
        print(model)
    elif not dist.is_initialized():
        print(model)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=analysis_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding="longest",
        )
    )

    # Training
    resume_from_checkpoint = None
    if training_args.num_iterations is not None:
        assert training_args.iteration is not None
        if training_args.iteration > 0:
            resume_from_checkpoint = True
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # remove the full model in the end to save space, only adapter is needed
    if isinstance(model, PeftModel):
        pytorch_model_path = os.path.join(
            training_args.output_dir, "pytorch_model_fsdp.bin")
        os.remove(pytorch_model_path) if os.path.exists(
            pytorch_model_path) else None


if __name__ == "__main__":
    main()
