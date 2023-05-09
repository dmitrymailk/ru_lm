import logging
import os
import sys


import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0.dev0")

logger = logging.getLogger(__name__)

import os
import sys


from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

import transformers

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


def main():
    os.environ["WANDB_PROJECT"] = "rulm_self_instruct"
    args = parse_args()
    model_name = "facebook/xglm-7.5B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ds_config = {
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },

        "optimizer": {
            "type": "OnebitAdam",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto",
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
            },
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
            "cpu_offload": True,
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 2000,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
    }

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
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_steps=0,
            learning_rate=2e-4,
            # fp16=True,
            logging_steps=1,
            output_dir=args.output_dir,
            # optim="adamw_bnb_8bit",
            optim="adamw_torch_fused",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            # в половину ускоряет тренировку
            tf32=True,
            bf16=True,
            report_to="wandb",
            num_train_epochs=args.num_train_epochs,
            deepspeed=ds_config,
        ),
        data_collator=collator,
    )
    model.config.use_cache = False
    trainer.train()


if __name__ == "__main__":
    main()
