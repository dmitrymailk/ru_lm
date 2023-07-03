export WANDB_PROJECT=lora_self_instruct
export CUDA_VISIBLE_DEVICES=2
OUTPUT=$1
if [ "$OUTPUT" == "" ]; then
    exit
fi

mkdir -p ./models/$OUTPUT
export WANDB_NAME=$OUTPUT

nohup python -u lora_train.py \
	--datasets chip2_instruct_alpha_prompt_en_v2_clean_v2 chip2_instruct_alpha_prompt_ru_v2_clean_v1 dolly_original_prompt_v2_clean_v1 dolly_translated_prompt_v2_clean_v1 openass_prompt_dataset_en_v2_clean_v2 openass_prompt_dataset_ru_v2_clean_v1 \
	--model_name huggyllama/llama-7b \
	--max_steps 2000 \
	--save_steps 100 \
	--per_device_train_batch_size 4 \
	--gradient_accumulation_steps 32 \
	--bits 8 \
	--num_train_epochs 8 \
	--output_dir ./models/$OUTPUT > ./models/$OUTPUT/training.log &

