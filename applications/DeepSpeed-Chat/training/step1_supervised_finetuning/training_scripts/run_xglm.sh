#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# export CUDA_VISIBLE_DEVICES=1,2,3
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_PROJECT="lora_self_instruct"
# DeepSpeed Team
OUTPUT=$1
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
    exit
fi

ZERO_STAGE=2
mkdir -p ./models/$OUTPUT
export WANDB_NAME=$OUTPUT

#    --data_path self_instruct_translated databricks_dolly_15k_translated_fixed \
#    --data_path self_instruct_translated databricks_dolly_15k_translated_fixed Dahoas/rm-static Dahoas/full-hh-rlhf \
#    --data_path self_instruct_translated databricks_dolly_15k_translated_fixed self_instruct_en databricks_dolly_15k_fixed_en \
#    --data_path self_instruct_translated databricks_dolly_15k_translated_fixed self_instruct_en databricks_dolly_15k_fixed_en \
#    --model_name_or_path facebook/xglm-4.5B \
#    --tokenizer_path facebook/xglm-4.5B \
nohup deepspeed main.py \
   --data_path chip2_instruct_alpha_prompt_en_v2_clean_v2 chip2_instruct_alpha_prompt_ru_v2_clean_v1 dolly_original_prompt_v2_clean_v1 dolly_translated_prompt_v2_clean_v1 openass_prompt_dataset_en_v2_clean_v2 openass_prompt_dataset_ru_v2_clean_v1 \
   --model_name_or_path facebook/xglm-4.5B \
   --tokenizer_path facebook/xglm-4.5B \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 2048 \
   --learning_rate 3.0e-4 \
   --weight_decay 0.1 \
   --num_train_epochs 8  \
   --gradient_accumulation_steps 8 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --gradient_checkpointing \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir ./models/$OUTPUT > ./models/$OUTPUT/training.log &
#    --zero_plus_plus \
