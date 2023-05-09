#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# You can provide two models to compare the performance of the baseline and the finetuned model
# export CUDA_VISIBLE_DEVICES=0
model_name="xglm-1.7B_ru_v2"
log_path="./models/$model_name/eval.log"
nohup python -u eval_model.py > $log_path &
