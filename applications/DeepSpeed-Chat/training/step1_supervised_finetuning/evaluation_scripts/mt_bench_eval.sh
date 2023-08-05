export CUDA_VISIBLE_DEVICES=2

log_path="./datasets/final_evaluation_datasets/mt_bench/eval$(date +"%d.%m.%Y_%H:%M:%S").log"

nohup python -u mt_bench_evaluation.py > $log_path &
