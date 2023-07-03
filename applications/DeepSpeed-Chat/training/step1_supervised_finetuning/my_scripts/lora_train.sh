export WANDB_PROJECT=lora_self_instruct
# export CUDA_VISIBLE_DEVICES=1
OUTPUT=$1
if [ "$OUTPUT" == "" ]; then
    exit
fi

mkdir -p ./models/$OUTPUT
export WANDB_NAME=$OUTPUT

nohup python -u lora_train.py \
	--datasets chip2_instruct_alpha_prompt_en_v2_clean_v2 chip2_instruct_alpha_prompt_ru_v2_clean_v1 dolly_original_prompt_v2_clean_v1 dolly_translated_prompt_v2_clean_v1 openass_prompt_dataset_en_v2_clean_v2 openass_prompt_dataset_ru_v2_clean_v1 \
	--model_name facebook/xglm-7.5B \
	--max_steps -1 \
	--save_steps 5000 \
	--per_device_train_batch_size 1 \
	--gradient_accumulation_steps 8 \
	--bits 16 \
	--num_train_epochs 8 \
	--output_dir ./models/$OUTPUT > ./models/$OUTPUT/training.log &

