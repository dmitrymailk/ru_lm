#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# export CUDA_VISIBLE_DEVICES=0,1,3
# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=output
    exit
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi
mkdir -p ./models/$OUTPUT
export CUDA_LAUNCH_BLOCKING=1

#    --data_path self_instruct_translated databricks_dolly_15k_translated_fixed \
nohup deepspeed main.py \
    --data_path self_instruct_translated \
   --data_split 1,0,0 \
   --model_name_or_path facebook/xglm-1.7B \
   --per_device_train_batch_size 12 \
   --per_device_eval_batch_size 12 \
   --max_seq_len 512 \
   --learning_rate 9.65e-6 \
   --weight_decay 0.1 \
   --num_train_epochs 2 \
   --gradient_accumulation_steps 8 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir ./models/$OUTPUT > ./models/$OUTPUT/training.log &
