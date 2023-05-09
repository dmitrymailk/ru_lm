import os
import sys


import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    PrefixTuningConfig,
    TaskType,
    prepare_model_for_int8_training,
)
import transformers
from datasets import load_dataset

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from utils.data.data_utils import create_prompt_dataset_v2
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--data_path",
        nargs="*",
        default=["Dahoas/rm-static"],
        help="Path to the training dataset. Accepted format:"
        "1) a single data path, 2) multiple datasets in the"
        "form: dataset1-path dataset2-path ...",
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )

    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the model."
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "rulm_self_instruct"
    args = parse_args()

    model_name = "facebook/xglm-7.5B"
    # model_name = "facebook/xglm-1.7B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = prepare_model_for_int8_training(model)

    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        num_virtual_tokens=20,
        num_attention_heads=12,
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    train_dataset, eval_dataset = create_prompt_dataset_v2(
        datasets_names=args.data_path,
        tokenizer=tokenizer,
        max_seq_len=1024,
        output_path="./datasets/",
        seed=42,
    )

    data_collator_pad = transformers.DataCollatorWithPadding(
        tokenizer=tokenizer,
    )

    def collator(x):
        features_map = {key: [] for key in x[0].keys()}
        for item in x:
            for key in item.keys():
                features_map[key].append(item[key])

        del features_map["labels"]
        features_map = data_collator_pad(features_map)
        features_map["labels"] = features_map["input_ids"]
        return features_map

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            warmup_steps=0,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir=args.output_dir,
            optim="adamw_bnb_8bit",
            # optim="adamw_torch_fused",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            # в половину ускоряет тренировку
            tf32=True,
            report_to="wandb",
            num_train_epochs=args.num_train_epochs,
        ),
        data_collator=collator,
    )
    model.config.use_cache = False
    trainer.train()
    # slow trainer?
