import os
import math
import sys
import os
import psutil


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from transformers import AutoTokenizer

from utils.data.data_utils import get_raw_dataset, create_prompt_dataset_v2
from datasets import concatenate_datasets


if __name__ == "__main__":
    print("start")
    # datasets_names = "self_instruct_translated databricks_dolly_15k_translated_fixed self_instruct_en databricks_dolly_15k_fixed_en Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets openai/webgpt_comparisons stanfordnlp/SHP"
    datasets_names = "Dahoas/synthetic-instruct-gptj-pairwise stanfordnlp/SHP"
    datasets_names = datasets_names.split()
    print(datasets_names)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    create_prompt_dataset_v2(
        datasets_names=datasets_names,
        tokenizer=tokenizer,
        max_seq_len=512
    )
    print("ok")
    hash
