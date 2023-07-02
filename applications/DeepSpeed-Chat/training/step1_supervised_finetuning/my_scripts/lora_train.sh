export WANDB_PROJECT=rulm_self_instruct
export CUDA_VISIBLE_DEVICES=1
OUTPUT=$1
if [ "$OUTPUT" == "" ]; then
    exit
fi

mkdir -p ./models/$OUTPUT
export WANDB_NAME=$OUTPUT

# --model_name_or_path bs-la/bloomz-7b1-4b-ru \
# --model_name_or_path k0t1k/mosaicml-mpt-7b-instruct-lora \
	# --model_name_or_path tiiuae/falcon-7b \
# nohup python -u qlora_train.py --datasets chip2_instruct_alpha_prompt_en_v2_clean_v2 chip2_instruct_alpha_prompt_ru_v2_clean_v1 dolly_original_prompt_v2_clean_v1 dolly_translated_prompt_v2_clean_v1 openass_prompt_dataset_en_v2_clean_v2 openass_prompt_dataset_ru_v2_clean_v1 \
# 	--model_name_or_path facebook/xglm-4.5B \
# 	--max_steps 20000 --save_steps 5000 \
# 	--output_dir ./models/$OUTPUT > ./models/$OUTPUT/training.log &

	# --model_name facebook/xglm-4.5B \
# nohup python -u -m torch.distributed.run --nnodes 1 --nproc_per_node 1 lora_train.py \
nohup python -u lora_train.py \
	--datasets chip2_instruct_alpha_prompt_en_v2_clean_v2 chip2_instruct_alpha_prompt_ru_v2_clean_v1 dolly_original_prompt_v2_clean_v1 dolly_translated_prompt_v2_clean_v1 openass_prompt_dataset_en_v2_clean_v2 openass_prompt_dataset_ru_v2_clean_v1 \
	--model_name facebook/xglm-7.5B \
	--max_steps 20000 --save_steps 5000 \
	--batch_size_per_device 1 \
	--quantize_mode 8bit \
	--output_dir ./models/$OUTPUT > ./models/$OUTPUT/training.log &

