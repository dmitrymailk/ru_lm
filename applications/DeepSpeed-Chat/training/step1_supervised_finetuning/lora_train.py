"""
Fine-Tune Falcon LLM models
"""

import argparse
from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    Trainer,
)
from peft import LoraConfig
import transformers
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from trl import SFTTrainer
from accelerate import Accelerator
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_int8_training,
)
from peft.tuners.lora import LoraLayer
import sys
import os
import bitsandbytes as bnb

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from utils.data.data_utils import create_prompt_dataset_v2




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="facebook/xglm-4.5B")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--quantize_mode",
        type=str,
        default="16bit",
        choices=["4bit", "8bit", "16bit"],
    )
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--save_steps", type=int, default=20000)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--batch_size_per_device", type=int, default=16)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Path to the training dataset. Accepted format:"
        "1) a single data path, 2) multiple datasets in the"
        "form: dataset1-path dataset2-path ...",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable HF gradient checkpointing for model.",
    )

    return parser.parse_args()


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

        touch(os.join(args.output_dir, "completed"))
        self.save_model(args, state, kwargs)


def find_all_linear_names(args, model):
    cls = (
        bnb.nn.Linear4bit
        if args.quantize_mode == "4int"
        else (bnb.nn.Linear8bitLt if args.quantize_mode == "8int" else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def run_training(args):
    model_name = args.model_name

    if args.quantize_mode == "4bit":
        print("4bit")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif args.quantize_mode == "8bit":
        print("8bit")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif args.quantize_mode == "16bit":
        print("16bit")
        bnb_config = BitsAndBytesConfig()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        # device_map={"": Accelerator().process_index},
        device_map="auto",
        trust_remote_code=True,
        # torch_dtype=torch.float16
    )
    # model.config.use_cache = False

    if getattr(model, "is_loaded_in_8bit", False) or getattr(
        model, "is_loaded_in_4bit", False
    ):
        print("prepare_model_for_kbit_training")
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset, eval_dataset = create_prompt_dataset_v2(
        datasets_names=args.datasets,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        output_path="./datasets/",
        seed=1234,
    )
    data_collator_pad = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=args.max_seq_len,
        return_tensors="pt",
    )

    def collator(x):
        features_map = data_collator_pad(x)
        return features_map

    lora_alpha = 16
    lora_dropout = 0.1
    lora_r = 64

    modules = find_all_linear_names(args, model)

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules=[
        #     # "k_proj",
        #     # "v_proj",
        #     # "q_proj",
        #     # "out_proj",
        #     # "fc1",
        #     # "fc2",
        # ],
        target_modules=modules,
    )
    model = get_peft_model(model, peft_config)

    for name, module in model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    per_device_train_batch_size = args.batch_size_per_device
    gradient_accumulation_steps = 1
    gradient_checkpointing = args.gradient_checkpointing
    optim = "adamw_bnb_8bit"
    save_steps = args.save_steps
    logging_steps = 10
    learning_rate = 2e-4
    max_grad_norm = 0.3
    max_steps = args.max_steps
    warmup_ratio = 0.03
    lr_scheduler_type = "constant"

    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=True,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        report_to="wandb",
    )
    print(training_arguments)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=training_arguments,
        data_collator=collator,
    )
    trainer.add_callback(SavePeftModelCallback)

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    trainer.train()


def main(args):
    run_training(args)


if __name__ == "__main__":
    args = get_args()
    main(args)
