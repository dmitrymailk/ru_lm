export CUDA_VISIBLE_DEVICES=2
export WANDB_BASE_URL="https://api.wandb.ai"

OUTPUT=$1
if [ "$OUTPUT" == "" ]; then
    exit
fi

mkdir -p ./models/$OUTPUT
export WANDB_NAME=$OUTPUT

# nohup python -u -m src.train --config-file configs/saiga_7b.json \
# nohup python -u -m src.train --config-file configs/ruGPT3.5-13B.json \
# nohup python -u -m src.train --config-file configs/saiga2_7b.json \
nohup python -u -m src.train --config-file configs/saiga2_13b.json \
	--train-file ./train.jsonl \
	--val-file valid.jsonl  \
	--output-dir models/$OUTPUT \
	--omit-base-model-save > ./models/$OUTPUT/training.log &