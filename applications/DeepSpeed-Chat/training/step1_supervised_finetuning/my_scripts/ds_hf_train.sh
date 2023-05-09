export CUDA_VISIBLE_DEVICES=0,1
OUTPUT=$1
mkdir -p ./models/$OUTPUT

nohup python ds_trainer.py \
	--data_path chip2_instruct_alpha_prompt_en chip2_instruct_alpha_prompt_ru dolly_original_prompt dolly_translated_prompt openass_prompt_dataset_en openass_prompt_dataset_ru \
	--num_train_epochs 4  \
	--output_dir ./models/$OUTPUT > ./models/$OUTPUT/training.log &