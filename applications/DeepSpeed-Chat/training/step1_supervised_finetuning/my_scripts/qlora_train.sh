export CUDA_VISIBLE_DEVICES=0,1,2
export WANDB_PROJECT=rulm_self_instruct

OUTPUT=$1
if [ "$OUTPUT" == "" ]; then
    exit
fi

mkdir -p ./models/$OUTPUT
export WANDB_NAME=$OUTPUT

# --model_name_or_path bs-la/bloomz-7b1-4b-ru \
# --model_name_or_path k0t1k/mosaicml-mpt-7b-instruct-lora \
nohup python -u qlora_train.py --datasets chip2_instruct_alpha_prompt_en_v2_clean_v1 chip2_instruct_alpha_prompt_ru_v2_clean_v1 dolly_original_prompt_v2 dolly_translated_prompt_v2_clean_v1 openass_prompt_dataset_en_v2_clean_v1 openass_prompt_dataset_ru_v2_clean_v1 \
	--source_max_len 2048 \
	--model_name_or_path tiiuae/falcon-7b \
	--max_steps 20000 --save_steps 5000 --gradient_checkpointing True \
	--output_dir ./models/$OUTPUT > ./models/$OUTPUT/training.log &