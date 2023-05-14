import os
import sys
import argparse

import torch
import transformers
import wandb
from datasets import load_dataset
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BloomTokenizerFast, get_scheduler, DataCollatorWithPadding

from petals import DistributedBloomForCausalLM
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from utils.data.data_utils import create_prompt_dataset, create_prompt_dataset_v2


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
        "--data_split",
        type=str,
        default="6,2,2",
        help="Comma-separated list of proportions for training"
        "phase 1, 2, and 3 data. For example the split `2,4,4`"
        "will use 60% of data for phase 1, 20% for phase 2"
        "and 20% for phase 3.",
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
        "--offload", action="store_true", help="Enable ZeRO Offload techniques."
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Actor model (and clones).",
    )
    ## LoRA for efficient training setting
    parser.add_argument(
        "--lora_dim",
        type=int,
        default=0,
        help="If > 0, use LoRA for efficient training.",
    )
    parser.add_argument(
        "--lora_module_name",
        type=str,
        default="decoder.layers.",
        help="The scope of LoRA.",
    )
    parser.add_argument(
        "--only_optimize_lora",
        action="store_true",
        help="Only optimize the LoRA parameters.",
    )
    args = parser.parse_args()

    # Validate settings
    if args.gradient_checkpointing and args.lora_dim > 0:
        assert (
            not args.only_optimize_lora
        ), "--gradient_checkpointing and --only_optimizer_lora cannot be enabled at the same time."

    return args


def collator(x):
    features_map = {key: [] for key in x[0].keys()}
    for item in x:
        for key in item.keys():
            features_map[key].append(item[key])

    del features_map["labels"]
    features_map = data_collator_pad(features_map)
    features_map["labels"] = features_map["input_ids"]
    return features_map


if __name__ == "__main__":
    args = parse_args()
    # Choose a model you'd like to prompt-tune. We recommend starting with
    # the smaller 7.1B version of BLOOM (bigscience/bloom-7b1-petals) for faster prototyping.
    # Once your code is ready, you can switch to full-scale
    # 176B-parameter BLOOM (bigscience/bloom-petals) or BLOOMZ (bigscience/bloomz-petals).
    # MODEL_NAME = "bigscience/bloom-7b1-petals"
    # MODEL_NAME = "bigscience/bloomz-petals"
    MODEL_NAME = "bigscience/bloom-petals"

    # Choose a prompt-tuning mode ('ptune' or 'deep_ptune').
    # The latter fine-tunes separate prefixes for each transformer block,
    # so prompt-tuning will take more time but yield better results.
    # See this paper for details of how it works: https://arxiv.org/pdf/2110.07602.pdf
    TUNING_MODE = "ptune"

    NUM_PREFIX_TOKENS = 16
    device = "cuda:0"
    BATCH_SIZE = 16
    LR = 1e-2
    WEIGHT_DECAY = 0.0
    SEED = args.seed
    MODEL_MAX_LENGTH = args.max_seq_len

    tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = MODEL_MAX_LENGTH
    model = DistributedBloomForCausalLM.from_pretrained(
        MODEL_NAME, pre_seq_len=NUM_PREFIX_TOKENS, tuning_mode=TUNING_MODE
    ).to(device)

    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n, p.requires_grad, p.device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    create_prompt_dataset_v2(
        datasets_names=args.data_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        output_path="./datasets/",
        seed=args.seed,
        local_rank=args.local_rank,
    )

    train_dataset, eval_dataset = create_prompt_dataset_v2(
        datasets_names=args.data_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        output_path="./datasets/",
        seed=args.seed,
    )
    data_collator_pad = DataCollatorWithPadding(
        tokenizer=tokenizer,
    )

    train_sampler = RandomSampler(train_dataset)
    eval_sampler = SequentialSampler(eval_dataset)
    print(f"args.per_device_train_batch_size={args.per_device_train_batch_size}")

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collator,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=collator,
        sampler=eval_sampler,
        batch_size=args.per_device_train_batch_size,
    )

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataset),
    )

    # def evaluation(model, eval_dataloader):
    #     model.eval()
    #     losses = 0
    #     for step, batch in enumerate(eval_dataloader):
    #         batch = batch.to(device)
    #         print(f"***Evaluation {step}/{len(eval_dataloader)}***")
    #         with torch.no_grad():
    #             outputs = model(**batch)

    #         loss = outputs.loss
    #         losses += loss.float()
    #     losses = losses / (step + 1)
    #     try:
    #         perplexity = torch.exp(losses)
    #     except OverflowError:
    #         perplexity = float("inf")
    #     return perplexity

    wandb.init(
        project="rulm_self_instruct",
    )

    model.train()
    for i, batch in tqdm(enumerate(train_dataset)):
        batch = {k: torch.tensor(v).to(device).unsqueeze(0) for k, v in batch.items()}
        del batch["attention_mask"]
        # print(batch)

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        wandb.log({"Train/Samples/train_loss": loss})
        print(f"Step {i} loss={loss}")

        if (i + 1) % 1000 == 0:
            sub_folder = f"step={i}"
            output_dir = os.path.join(args.output_dir, sub_folder)
            model.save_pretrained(save_directory=output_dir)
