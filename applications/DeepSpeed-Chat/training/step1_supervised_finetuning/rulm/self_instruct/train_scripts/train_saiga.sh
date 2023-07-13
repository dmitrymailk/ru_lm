export CUDA_VISIBLE_DEVICES=3
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_PROJECT="saiga_self_instruct"

OUTPUT=$1
if [ "$OUTPUT" == "" ]; then
    exit
fi

ZERO_STAGE=2
mkdir -p ./models/$OUTPUT

nohup python -u -m src.train --config-file configs/saiga_7b.json \
	--train-file ./train.jsonl \
	--val-file valid.jsonl  \
	--output-dir models/$OUTPUT \
	--omit-base-model-save > ./models/$OUTPUT/training.log &