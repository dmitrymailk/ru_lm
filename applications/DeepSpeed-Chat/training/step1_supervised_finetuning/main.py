#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import deepspeed
from transformers import XGLMForCausalLM, XGLMConfig, TrainingArguments
from transformers.deepspeed import HfDeepSpeedConfig

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    DataCollatorWithPadding,
    XGLMForCausalLM,
    DataCollatorForSeq2Seq,
)
import deepspeed
from deepspeed.ops.adam import FusedAdam
import wandb


# from deepspeed.

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from utils.data.data_utils import create_prompt_dataset, create_prompt_dataset_v2
from utils.utils import (
    print_rank_0,
    to_device,
    save_hf_format,
    set_random_seed,
    get_all_reduce_mean,
    get_optimizer_grouped_parameters,
    save_zero_three_model,
)
from utils.ds_utils import get_train_ds_config
from utils.module.lora import (
    convert_linear_layer_to_lora,
    convert_lora_to_linear_layer,
    only_optimize_lora_parameters,
)
from utils.model.model_utils import create_hf_model


from typing import List, Optional, Tuple

import torch


def find_all_linear_names(args, model):
    cls = (
        bnb.nn.Linear4bit
        if args.bits == 4
        else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


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
        "--sft_only_data_path",
        nargs="*",
        default=[],
        help="Path to the dataset for only using in SFT phase.",
    )
    parser.add_argument(
        "--data_output_path",
        type=str,
        default="/tmp/data_files/",
        help="Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help="Tokenizer path",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the model."
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable HF gradient checkpointing for model.",
    )
    # deepspeed features
    parser.add_argument(
        "--zero_plus_plus", action="store_true", help="Enable ZeRO++ techniques."
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Actor model (and clones).",
    )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = {
        "fp16": {
            "enabled": True,
            "auto_cast": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 3e-6,
                # "lr": 3e-10,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.0,
            },
        },
        # "scheduler": {
        #     "type": "WarmupDecayLR",
        #     "params": {
        #         "warmup_min_lr": 0,
        #         "warmup_max_lr": 3e-5,
        #         "warmup_num_steps": 2000,
        #     },
        # },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "none", "pin_memory": True},
            "offload_param": {"device": "none", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_fp16_weights_on_model_save": True,
        },
        # "zero_optimization": {
        #     "stage": 0,
        # },
        "gradient_accumulation_steps": 32,
        "gradient_clipping": 1.0,
        "steps_per_print": 100,
        "train_batch_size": 16,
        "train_micro_batch_size_per_gpu": 1,
        "wall_clock_breakdown": False,
        "wandb": {
            "enabled": True,
            "project": "rulm_self_instruct",
        },
    }
    ds_config["train_batch_size"] = (
        ds_config["train_micro_batch_size_per_gpu"]
        * torch.cuda.device_count()
        * ds_config["gradient_accumulation_steps"]
    )

    if args.zero_plus_plus:
        print("Enable Zero++")
        zero_plus_plus_config = {
            "zero_optimization": {
                "stage": 3,
                "reduce_bucket_size": 10000000,
                "reduce_scatter": True,
                "zero_quantized_weights": True,
                "zero_hpz_partition_size": 1,
                "zero_quantized_gradients": True,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "offload_optimizer": {"device": "none", "pin_memory": True},
                "offload_param": {"device": "none", "pin_memory": True},
                "overlap_comm": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_fp16_weights_on_model_save": True,
            }
        }
        ds_config["zero_optimization"] = zero_plus_plus_config["zero_optimization"]

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    # https://pytorch.org/docs/stable/distributed.html#torch.distributed.barrier
    # синхронизирует все процессы
    torch.distributed.barrier()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        # "./models/tokenizers/xglm_4.5B_fix_v1",
        # "./models/tokenizers/xglm_1.7B_fix_v1",
        fast_tokenizer=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = None

    if ds_config["zero_optimization"]["stage"] == 3:
        with deepspeed.zero.Init(
            config_dict_or_path=ds_config,
        ):
            training_args = TrainingArguments(
                deepspeed=ds_config, output_dir="./models/"
            )
            model = XGLMForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.float16,
            )
    else:
        training_args = TrainingArguments(deepspeed=ds_config, output_dir="./models/")
        model = XGLMForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16,
        )

    # model = torch.compile(model)

    if args.local_rank == 0:
        create_prompt_dataset_v2(
            datasets_names=args.data_path,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            output_path="./datasets/",
            seed=args.seed,
            local_rank=args.local_rank,
        )
    torch.distributed.barrier()

    # load in every process
    train_dataset, eval_dataset = create_prompt_dataset_v2(
        datasets_names=args.data_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        output_path="./datasets/",
        seed=args.seed,
    )
    data_collator_pad = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=2048,
        return_tensors="pt",
    )

    def collator(x):
        features_map = data_collator_pad(x)
        return features_map

    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collator,
        sampler=train_sampler,
        batch_size=ds_config["train_micro_batch_size_per_gpu"],
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=collator,
        sampler=eval_sampler,
        batch_size=ds_config["train_micro_batch_size_per_gpu"],
    )

    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            # deepspeed.accelerator.get_accelerator().empty_cache()
            batch = to_device(batch, device)
            print_rank_0(f"***Evaluation {step}/{len(eval_dataloader)}***")
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        config=ds_config,
        dist_init_required=True,
    )

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    print_rank_0(
        f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank,
    )
    perplexity = evaluation(model, eval_dataloader)

    print_rank_0(f"ppl: {perplexity}", args.global_rank)

    if args.global_rank == 0:
        wandb.log({"eval_ppl": perplexity})
    torch.distributed.barrier()
    zero_stage = ds_config["zero_optimization"]["stage"]
    checkpoint_steps = len(train_dataloader) // 5
    print_rank_0("***** Checkpoint save steps *****", checkpoint_steps)
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank,
        )
        model.train()
        for step, batch in enumerate(train_dataloader):
            # deepspeed.accelerator.get_accelerator().empty_cache()
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            # if args.global_rank == 0:
            #     print(loss.item())
            model.backward(loss)
            model.step()
            if (step + 1) % checkpoint_steps == 0:
                torch.distributed.barrier()
                print_rank_0(
                    f"Save model epoch={epoch}_step={step}",
                    args.global_rank,
                )
                sub_folder = f"epoch={epoch}_step={step}"
                output_dir = os.path.join(args.output_dir, sub_folder)
                if args.global_rank == 0:
                    save_hf_format(model, tokenizer, args, sub_folder=sub_folder)

                if zero_stage == 3:
                    # For zero stage 3, each gpu only has a part of the model, so we need a special save function
                    os.makedirs(output_dir, exist_ok=True)

                    save_zero_three_model(
                        model, args.global_rank, output_dir, zero_stage=zero_stage
                    )

        torch.distributed.barrier()
        print_rank_0(
            f"Save model epoch={epoch}",
            args.global_rank,
        )
        sub_folder = f"epoch={epoch}"
        output_dir = os.path.join(args.output_dir, sub_folder)
        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args, sub_folder=sub_folder)

        if zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            os.makedirs(output_dir, exist_ok=True)
            save_zero_three_model(
                model, args.global_rank, output_dir, zero_stage=zero_stage
            )

        # Evaluate perplexity on the validation set.
        print_rank_0(
            f"***** Evaluating perplexity, Epoch {epoch+1}/{args.num_train_epochs} *****",
            args.global_rank,
        )
        perplexity = evaluation(model, eval_dataloader)
        print_rank_0(f"ppl: {perplexity}", args.global_rank)
        if args.global_rank == 0:
            wandb.log({"eval_ppl": perplexity})
        model.tput_timer.update_epoch_count()


if __name__ == "__main__":
    main()
