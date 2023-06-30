export CUDA_VISIBLE_DEVICES=0
model=facebook/wmt21-dense-24-wide-en-x
# model=dim/facebook_wmt21-dense-24-wide-en-x
num_shard=2
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

# docker run -u "$UID:$GID" --gpus all --shm-size 20g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:sha-ae466a8 --model-id $model --num-shard $num_shard
docker run -u "$UID:$GID" --gpus all --shm-size 20g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:sha-ae466a8 --model-id $model --sharded false
# docker run -u "$UID:$GID" --gpus all --shm-size 20g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:sha-ae466a8 --help