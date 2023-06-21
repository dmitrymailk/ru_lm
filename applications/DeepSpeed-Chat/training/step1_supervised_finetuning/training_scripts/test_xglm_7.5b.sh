OUTPUT=$1
nohup deepspeed pure_deepspeed.py  > ./models/$OUTPUT/training.log &
