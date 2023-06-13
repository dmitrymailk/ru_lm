# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset, Dataset
import evaluate

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

import hashlib
from datasets import load_from_disk
from datasets import load_dataset, load_from_disk
from torch.utils.data import Subset
import re
from datasets import concatenate_datasets

torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="tiiuae/falcon-7b",
    )
    trust_remote_code: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
        },
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."},
    )


@dataclass
class DataArguments:
    source_max_len: int = field(
        default=2048,
        metadata={
            "help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    target_max_len: int = field(
        default=2048,
        metadata={
            "help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    datasets: list[str] = field(
        metadata={"help": "Which dataset to finetune on. See datamodule for options."},
        default_factory=list[str],
    )


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)

    adam8bit: bool = field(default=True, metadata={"help": "Use 8-bit adam."})
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=4, metadata={"help": "How many bits to use."})
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    max_memory_MB: int = field(default=40000, metadata={"help": "Free memory per gpu."})
    report_to: str = field(
        default="wandb",
        metadata={"help": "To use wandb or something else for reporting."},
    )
    output_dir: str = field(
        default="./output",
        metadata={"help": "The output dir for logs and checkpoints"},
    )
    optim: str = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to be used"},
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={
            "help": "The training batch size per GPU. Increase for better speed."
        },
    )
    gradient_accumulation_steps: int = field(
        default=16,
        metadata={
            "help": "How many gradients to accumulate before to perform an optimizer step"
        },
    )
    max_steps: int = field(
        default=10000, metadata={"help": "How many optimizer update steps to take"}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "The L2 weight decay rate of AdamW"}
    )  # use lora dropout instead for regularization if needed
    learning_rate: float = field(
        default=0.0002,
        metadata={"help": "The learnign rate"},
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Removed unused columns. Needed to make this codebase work."},
    )
    max_grad_norm: float = field(
        default=0.3,
        metadata={
            "help": "Gradient clipping max norm. This is tuned and works well for all models tested."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing. You want to use this."},
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "To train or not to train, that is the question?"},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Fraction of steps to do a warmup for"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "The frequency of update steps after which to log the loss"},
    )
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_strategy: str = field(
        default="steps", metadata={"help": "When to save checkpoints"}
    )
    save_steps: int = field(
        default=5000, metadata={"help": "How often to save a model"}
    )
    save_total_limit: int = field(
        default=40,
        metadata={
            "help": "How many checkpoints to save before the oldest is overwritten"
        },
    )


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


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print("Saving PEFT checkpoint...")
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(
                state.best_model_checkpoint, "adapter_model"
            )
        else:
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, "a"):
                os.utime(fname, times)

        touch(join(args.output_dir, "completed"))
        self.save_model(args, state, kwargs)


def get_accelerate_model(args, checkpoint_dir):
    n_gpus = torch.cuda.device_count()
    max_memory = f"{args.max_memory_MB}MB"
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"
    # device_map = {"": 0}

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}
        max_memory = {"": max_memory[local_rank]}

    print(f"loading base model {args.model_name_or_path}...")
    compute_dtype = (
        torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        torch_dtype=(
            torch.float32
            if args.fp16
            else (torch.bfloat16 if args.bf16 else torch.float32)
        ),
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token,
    )
    if compute_dtype == torch.float16 and args.bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print(
                "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
            )
            print("=" * 80)

    setattr(model, "model_parallel", True)
    setattr(model, "is_parallelizable", True)

    model.config.torch_dtype = (
        torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    )

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if checkpoint_dir is not None:
        print("Loading adapters from checkpoint.")
        model = PeftModel.from_pretrained(
            model, join(checkpoint_dir, "adapter_model"), is_trainable=True
        )
    else:
        print(f"adding LoRA modules...")
        modules = find_all_linear_names(args, model)
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model


def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4:
        trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):
    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return


# russian dataset
class RuInstructTranslated(PromptRawDataset):
    def __init__(
        self,
    ):
        self.dataset_name = "self_instruct_translated"
        self.dataset_name_clean = "self_instruct_translated"
        self.raw_datasets = load_from_disk(
            "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/self_instruct_translated/"
        )

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return f"Human: {sample['prompt_translated']} Assistant:"

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return f" {sample['completion_translated']}"

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return self.get_prompt(sample) + self.get_chosen(sample)

    def get_prompt_and_rejected(self, sample):
        return


# russian dataset
class RuDollyInstructTranslated(PromptRawDataset):
    def __init__(self):
        self.dataset_name = "dolly_translated_prompt"
        self.dataset_name_clean = "dolly_translated_prompt"
        self.raw_datasets = load_from_disk(
            "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/prompt_datasets/dolly_translated_prompt"
        )
        self.raw_datasets = self.raw_datasets.train_test_split(test_size=100, seed=42)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"]


# russian dataset
class RuDollyInstructTranslatedV2(PromptRawDataset):
    def __init__(self):
        self.dataset_name = "dolly_translated_prompt_v2"
        self.dataset_name_clean = "dolly_translated_prompt_v2"
        self.raw_datasets = load_from_disk(
            "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/prompt_datasets/dolly_translated_prompt_v2_clean_v1"
        )
        self.raw_datasets = self.raw_datasets.train_test_split(test_size=100, seed=42)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"]


# russian dataset
class RuChip2Translated(PromptRawDataset):
    def __init__(self):
        self.dataset_name = "chip2_instruct_alpha_prompt_ru"
        self.dataset_name_clean = "chip2_instruct_alpha_prompt_ru"
        self.raw_datasets = load_from_disk(
            "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/prompt_datasets/chip2_instruct_alpha_prompt_ru"
        )
        self.raw_datasets = self.raw_datasets.train_test_split(test_size=100, seed=42)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"]


# russian dataset
class RuChip2TranslatedV2(PromptRawDataset):
    def __init__(self):
        self.dataset_name = "chip2_instruct_alpha_prompt_ru_v2"
        self.dataset_name_clean = "chip2_instruct_alpha_prompt_ru_v2"
        self.raw_datasets = load_from_disk(
            "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/prompt_datasets/chip2_instruct_alpha_prompt_ru_v2_clean_v1"
        )
        self.raw_datasets = self.raw_datasets.train_test_split(test_size=100, seed=42)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"]


# russian dataset
class RuOpenAssTranslated(PromptRawDataset):
    def __init__(self):
        self.dataset_name = "openass_prompt_dataset_ru"
        self.dataset_name_clean = "openass_prompt_dataset_ru"
        self.raw_datasets = load_from_disk(
            "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/prompt_datasets/openass_prompt_dataset_ru"
        )
        self.raw_datasets = self.raw_datasets.train_test_split(test_size=100, seed=42)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"]


class RuOpenAssTranslatedV2(PromptRawDataset):
    def __init__(self):
        self.dataset_name = "openass_prompt_dataset_ru_v2"
        self.dataset_name_clean = "openass_prompt_dataset_ru_v2"
        self.raw_datasets = load_from_disk(
            "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/prompt_datasets/openass_prompt_dataset_ru_v2_clean_v1"
        )
        self.raw_datasets = self.raw_datasets.train_test_split(test_size=100, seed=42)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"]


# english dataset
class EnDollyInstructTranslated(PromptRawDataset):
    def __init__(self):
        self.dataset_name = "dolly_original_prompt"
        self.dataset_name_clean = "dolly_original_prompt"
        self.raw_datasets = load_from_disk(
            "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/prompt_datasets/dolly_original_prompt"
        )
        self.raw_datasets = self.raw_datasets.train_test_split(test_size=100, seed=42)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"]


# english dataset
class EnDollyInstructTranslatedV2(PromptRawDataset):
    def __init__(self):
        self.dataset_name = "dolly_original_prompt_v2"
        self.dataset_name_clean = "dolly_original_prompt_v2"
        self.raw_datasets = load_from_disk(
            "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/prompt_datasets/dolly_original_prompt_v2"
        )
        self.raw_datasets = self.raw_datasets.train_test_split(test_size=100, seed=42)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"]


# english dataset
class EnChip2Translated(PromptRawDataset):
    def __init__(self):
        self.dataset_name = "chip2_instruct_alpha_prompt_en"
        self.dataset_name_clean = "chip2_instruct_alpha_prompt_en"
        self.raw_datasets = load_from_disk(
            "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/prompt_datasets/chip2_instruct_alpha_prompt_en"
        )
        self.raw_datasets = self.raw_datasets.train_test_split(test_size=100, seed=42)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"]


class EnChip2TranslatedV2(PromptRawDataset):
    def __init__(self):
        self.dataset_name = "chip2_instruct_alpha_prompt_en_v2"
        self.dataset_name_clean = "chip2_instruct_alpha_prompt_en_v2"
        self.raw_datasets = load_from_disk(
            "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/prompt_datasets/chip2_instruct_alpha_prompt_en_v2_clean_v1"
        )
        self.raw_datasets = self.raw_datasets.train_test_split(test_size=100, seed=42)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"]


# english dataset
class EnOpenAssTranslated(PromptRawDataset):
    def __init__(self):
        self.dataset_name = "openass_prompt_dataset_en"
        self.dataset_name_clean = "openass_prompt_dataset_en"
        self.raw_datasets = load_from_disk(
            "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/prompt_datasets/openass_prompt_dataset_en"
        )
        self.raw_datasets = self.raw_datasets.train_test_split(test_size=100, seed=42)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"]


class EnOpenAssTranslatedV2(PromptRawDataset):
    def __init__(self):
        self.dataset_name = "openass_prompt_dataset_en_v2"
        self.dataset_name_clean = "openass_prompt_dataset_en_v2"
        self.raw_datasets = load_from_disk(
            "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/prompt_datasets/openass_prompt_dataset_en_v2_clean_v1"
        )
        self.raw_datasets = self.raw_datasets.train_test_split(test_size=100, seed=42)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"]


# english dataset
class EnInstructTranslated(PromptRawDataset):
    def __init__(self):
        self.dataset_name = "self_instruct_en"
        self.dataset_name_clean = "self_instruct_en"
        self.raw_datasets = load_from_disk(
            "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/self_instruct_translated/"
        )

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return f"Human: {sample['prompt']} Assistant:"

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return f" {sample['completion']}"

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return self.get_prompt(sample) + self.get_chosen(sample)

    def get_prompt_and_rejected(self, sample):
        return


def get_raw_dataset(dataset_name):
    if dataset_name == "dolly_original_prompt":
        return EnDollyInstructTranslated()
    elif dataset_name == "dolly_original_prompt_v2":
        return EnDollyInstructTranslatedV2()
    elif dataset_name == "dolly_translated_prompt":
        return RuDollyInstructTranslated()
    elif dataset_name == "dolly_translated_prompt_v2_clean_v1":
        return RuDollyInstructTranslatedV2()
    elif dataset_name == "chip2_instruct_alpha_prompt_ru":
        return RuChip2Translated()
    elif dataset_name == "chip2_instruct_alpha_prompt_ru_v2_clean_v1":
        return RuChip2TranslatedV2()
    elif dataset_name == "chip2_instruct_alpha_prompt_en":
        return EnChip2Translated()
    elif dataset_name == "chip2_instruct_alpha_prompt_en_v2_clean_v1":
        return EnChip2TranslatedV2()
    elif dataset_name == "openass_prompt_dataset_ru":
        return RuOpenAssTranslated()
    elif dataset_name == "openass_prompt_dataset_ru_v2_clean_v1":
        return RuOpenAssTranslatedV2()
    elif dataset_name == "openass_prompt_dataset_en":
        return EnOpenAssTranslated()
    elif dataset_name == "openass_prompt_dataset_en_v2_clean_v1":
        return EnOpenAssTranslatedV2()
    else:
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
        )


def get_hash_filename(
    tokenizer=None,
    output_path="./",
    data_path: list = None,
    max_seq_len: int = 512,
    seed: int = 1234,
):
    os.makedirs(output_path, exist_ok=True)
    data_path.sort()
    f_name = "_".join(data_path)
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    f_name = f"{f_name}_tokenizer{tokenizer_name}_seqlen{max_seq_len}_seed{seed}"
    f_name = "_".join(f_name.split("/"))
    # default hash generates hash depends on process
    f_name = hashlib.sha256(f_name.encode("utf-8")).hexdigest()
    return f_name


def prepare_dataset_v3(
    prompt_func,
    example,
    tokenizer,
    max_seq_len=2048,
):
    # print(example)
    formated_prompt = prompt_func(example)
    chosen_token = tokenizer(
        formated_prompt,
        max_length=max_seq_len,
        # padding="max_length",
        # padding=False,
        truncation=True,
        # return_tensors="pt",
    )

    return {
        "input_ids": chosen_token["input_ids"],
        "attention_mask": chosen_token["attention_mask"],
        "labels": chosen_token["input_ids"],
    }


def create_prompt_dataset_v2(
    datasets_names: list[str] = None,
    tokenizer=None,
    max_seq_len: int = 2048,
    output_path: str = "./datasets",
    seed: int = 1234,
):
    os.makedirs(output_path, exist_ok=True)
    fname = get_hash_filename(
        tokenizer=tokenizer,
        output_path=output_path,
        data_path=datasets_names,
        max_seq_len=max_seq_len,
        seed=seed,
    )
    print("fname", fname)
    train_fname = f"{output_path}/traindata_{fname}/"
    eval_fname = f"{output_path}/evaldata_{fname}/"
    cache_found = os.path.isdir(train_fname) and os.path.isdir(eval_fname)
    print("cache_found", cache_found)

    if cache_found:
        train_data = load_from_disk(train_fname)
        eval_data = load_from_disk(eval_fname)
        return train_data, eval_data

    if not cache_found:
        train_datasets = []
        eval_datasets = []
        for name in datasets_names:
            dataset = get_raw_dataset(
                dataset_name=name,
            )
            # print(dataset)
            for stage in ["train", "test"]:
                if stage == "train":
                    train_data = dataset.get_train_data().map(
                        lambda x: prepare_dataset_v3(
                            prompt_func=dataset.get_prompt_and_chosen,
                            example=x,
                            tokenizer=tokenizer,
                            max_seq_len=max_seq_len,
                        ),
                        num_proc=32,
                        load_from_cache_file=False,
                    )
                    train_data = train_data.select_columns(
                        ["input_ids", "attention_mask", "labels"]
                    )
                    train_datasets.append(train_data)
                else:
                    eval_data = dataset.get_eval_data().map(
                        lambda x: prepare_dataset_v3(
                            prompt_func=dataset.get_prompt_and_chosen,
                            example=x,
                            tokenizer=tokenizer,
                            max_seq_len=max_seq_len,
                        ),
                        num_proc=32,
                        load_from_cache_file=False,
                    )
                    eval_data = eval_data.select_columns(
                        ["input_ids", "attention_mask", "labels"]
                    )
                    eval_datasets.append(eval_data)

        # print(train_datasets)
        train_datasets = concatenate_datasets(train_datasets)
        eval_datasets = concatenate_datasets(eval_datasets)
        train_datasets.save_to_disk(train_fname)
        eval_datasets.save_to_disk(eval_fname)
        return train_datasets, eval_datasets


def make_data_module_v2(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=2048,
        return_tensors="pt",
    )

    def collator(x):
        # print(x)
        # if "input_embeds" in x.keys():
        #     del x["inputs_embeds"]
        features_map = data_collator(x)
        return features_map

    train_dataset, eval_dataset = create_prompt_dataset_v2(
        datasets_names=args.datasets,
        tokenizer=tokenizer,
        max_seq_len=args.source_max_len,
    )

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )


def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, "completed"))
        if is_completed:
            return None, True  # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith(
                "checkpoint"
            ):
                max_step = max(max_step, int(filename.replace("checkpoint-", "")))
        if max_step == 0:
            return None, is_completed  # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f"checkpoint-{max_step}")
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed  # checkpoint found!
    return None, False  # first training


def train():
    hfparser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        extra_args,
    ) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    args = argparse.Namespace(
        **vars(model_args),
        **vars(data_args),
        **vars(training_args),
    )

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print("Detected that training was already completed!")

    model = get_accelerate_model(args, checkpoint_dir)

    model.config.use_cache = False
    print_trainable_parameters(args, model)
    print("loaded model")
    set_seed(args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
    )

    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    data_module = make_data_module_v2(tokenizer=tokenizer, args=args)
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k: v for k, v in data_module.items() if k != "predict_dataset"},
    )

    # Callbacks
    trainer.add_callback(SavePeftModelCallback)

    # Verifying the datatypes.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        logger.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    # Prediction

    if args.do_train or args.do_eval or args.do_predict:
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))


if __name__ == "__main__":
    train()