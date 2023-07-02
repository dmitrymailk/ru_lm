import logging
import os
import sys


import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
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


def main():
    os.environ["WANDB_PROJECT"] = "lora_self_instruct"
    model_name = "facebook/xglm-564M"
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    data_path = ["dolly_translated_prompt_v2_clean_v1"]
    max_seq_len = 2048
    seed = 1234
    dataset_output_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets"
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
        max_length=2048,
        return_tensors="pt",
    )

    def collator(x):
        features_map = data_collator_pad(x)
        return features_map


if __name__ == "__main__":
    main()
