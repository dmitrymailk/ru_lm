#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# export CUDA_VISIBLE_DEVICES=0,1

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
   --data_path chip2_instruct_alpha_prompt_en_v2_clean_v1 chip2_instruct_alpha_prompt_ru_v2_clean_v1 dolly_original_prompt_v2 dolly_translated_prompt_v2_clean_v1 openass_prompt_dataset_en_v2_clean_v1 openass_prompt_dataset_ru_v2_clean_v1 \
   --model_name_or_path facebook/xglm-4.5B \
   --tokenizer_path ./models/tokenizers/xglm_4.5B_fix_v1 \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 2048 \
   --learning_rate 9.65e-6 \
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
