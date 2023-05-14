#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
export CUDA_VISIBLE_DEVICES=1

# DeepSpeed Team
OUTPUT=$1
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
    exit
fi
mkdir -p ./models/$OUTPUT

# single node
nohup python -u petals_train.py \
   --data_path dolly_translated_prompt \
   --per_device_train_batch_size 8 \
   --max_seq_len 1024 \
   --num_train_epochs 4  \
   --seed 1234 \
   --output_dir ./models/$OUTPUT > ./models/$OUTPUT/training.log &