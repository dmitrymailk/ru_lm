#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# You can provide two models to compare the performance of the baseline and the finetuned model
export CUDA_VISIBLE_DEVICES=0
# model_name="xglm-4.5B_ru_v10"
# log_path="./models/$model_name/eval$(date +"%d.%m.%Y_%H:%M:%S").log"

model_name_qlora="red_pajama_chat_ru_v1/checkpoint-20000"
log_path="./models/$model_name_qlora/adapter_model/eval$(date +"%d.%m.%Y_%H:%M:%S").log"
nohup python -u eval_model.py > $log_path &
