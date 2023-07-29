export CUDA_VISIBLE_DEVICES=2

log_path="./eval_optimization$(date +"%d.%m.%Y_%H:%M:%S").log"

nohup python -u optuna_xglm.py > $log_path &