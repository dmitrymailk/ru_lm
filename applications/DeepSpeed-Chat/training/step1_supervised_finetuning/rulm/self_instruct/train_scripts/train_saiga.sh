export CUDA_VISIBLE_DEVICES=2,3
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_PROJECT="saiga_self_instruct"

nohup python -m src.train --config-file configs/saiga_7b.json \
	--train-file ./train.jsonl \
	--val-file valid.jsonl  \
	--output-dir models/saiga_7b_v1 \
	--omit-base-model-save > ./models/saiga_7b_v1/training.log &