#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# export CUDA_VISIBLE_DEVICES=0,1,3

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
    exit
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p ./models/$OUTPUT

#    --data_path self_instruct_translated databricks_dolly_15k_translated_fixed \
#    --data_path self_instruct_translated databricks_dolly_15k_translated_fixed Dahoas/rm-static Dahoas/full-hh-rlhf \
#    --data_path self_instruct_translated databricks_dolly_15k_translated_fixed self_instruct_en databricks_dolly_15k_fixed_en \
#    --data_path self_instruct_translated databricks_dolly_15k_translated_fixed self_instruct_en databricks_dolly_15k_fixed_en \
nohup deepspeed main.py \
   --data_split 1,0,0 \
   --data_path chip2_instruct_alpha_prompt_en chip2_instruct_alpha_prompt_ru dolly_original_prompt dolly_translated_prompt openass_prompt_dataset_en openass_prompt_dataset_ru \
   --model_name_or_path facebook/xglm-4.5B \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 9.65e-6 \
   --weight_decay 0.1 \
   --num_train_epochs 4  \
   --gradient_accumulation_steps 8 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --gradient_checkpointing \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir ./models/$OUTPUT > ./models/$OUTPUT/training.log &
