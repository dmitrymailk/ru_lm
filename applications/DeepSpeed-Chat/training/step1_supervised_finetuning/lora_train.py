import logging
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"
import sys


import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
import accelerate

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0.dev0")

logger = logging.getLogger(__name__)


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from utils.data.data_utils import create_prompt_dataset_v2
import argparse

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
import torch
import bitsandbytes as bnb

torch.backends.cuda.matmul.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["dolly_translated_prompt_v2_clean_v1"],
        help="Path to the training dataset. Accepted format:"
        "1) a single data path, 2) multiple datasets in the"
        "form: dataset1-path dataset2-path ...",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        # required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/models/test",
        help="Where to store the model.",
    )

    parser.add_argument(
        "--seed", type=int, default=1234, help="A seed for reproducible training."
    )

    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
    )

    args = parser.parse_args()

    return args


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


def main():
    args = parse_args()

    os.environ["WANDB_PROJECT"] = "lora_self_instruct"
    # model_name = "facebook/xglm-4.5B"
    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        device_map="auto",
        # torch_dtype=torch.float16,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Fixing broken tokenizers
    tokenizer.pad_token = tokenizer.eos_token

    data_path = args.datasets
    max_seq_len = args.max_seq_len
    seed = args.seed
    dataset_output_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets"
    output_dir = args.output_dir

    train_dataset, eval_dataset = create_prompt_dataset_v2(
        datasets_names=data_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        output_path=dataset_output_path,
        seed=seed,
    )
    data_collator_pad = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )

    def collator(x):
        features_map = data_collator_pad(x)
        return features_map

    # prepare for lora int8 train
    model = prepare_model_for_kbit_training(model)
    lora_r = 16
    lora_alpha = 16
    lora_target_modules = [
        # "q_proj",
        # "v_proj",
        # # "k_proj",
        # # "out_proj",
        # # "fc1",
        # # "fc2",
        "q_proj",
        "gate_proj",
        "up_proj",
        "o_proj",
        "v_proj",
        "k_proj",
        "down_proj",
    ]
    # lora_target_modules = find_all_linear_names(args=args, model=model)
    lora_dropout = 0.05

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    if torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    # model = torch.compile(model, backend='inductor')
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            bf16=True,
            # fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="no",
            save_strategy="steps",
            max_steps=args.max_steps,
            save_steps=args.save_steps,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
            group_by_length=True,
            report_to="wandb",
        ),
        data_collator=collator,
    )
    # https://discuss.huggingface.co/t/what-is-the-purpose-of-use-cache-in-decoder/958
    # везде можно заметить данную строчку, но согласно данному ответу этот параметр используется
    # только для генерации
    model.config.use_cache = False
    # if torch.__version__ >= "2" and sys.platform != "win32":

    print("Start train")
    trainer.train()


if __name__ == "__main__":
    main()
