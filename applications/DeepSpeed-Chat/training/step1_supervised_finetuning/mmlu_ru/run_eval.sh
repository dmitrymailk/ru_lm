# export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=1
# export CUDA_VISIBLE_DEVICES=2
export CUDA_VISIBLE_DEVICES=3


# python -u mmlu_ru.py --hf_model_id "huggyllama/llama-7b" --k_shot 5 --lang "ru" --output_dir "results"
# model_name=/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/rulm/self_instruct/models/goral_xglm_4.5B
# model_name=/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/rulm/self_instruct/models/saiga_7b_v2
# model_name=/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/rulm/self_instruct/models/rugpt_v1
# model_name=IlyaGusev/saiga_7b_lora
# model_name=IlyaGusev/gigasaiga_lora
# model_name=/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/rulm/self_instruct/models/goral_xglm_v2
# model_name=dim/llama2_13b_dolly_oasst1_chip2
# model_name=IlyaGusev/saiga2_13b_lora
# model_name=/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/rulm/rulm2/rulm/self_instruct/models/saiga2_v2
model_name=/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/models/xglm-4.5B_ru_v10/epoch=6_step=41141
lang="en"
# lang="ru"
# results_folder="xglm_4.5B_lora_our_dataset_$lang"
# results_folder="llama_7b_our_dataset_$lang"
# results_folder="rugpt_13b_our_dataset_$lang"
# results_folder="saiga_7b_lora_$lang"
# results_folder="llama2_13b_lora_our_dataset_$lang"
# results_folder="llama2_13b_lora_saiga_dataset_$lang"
# results_folder="llama2_7b_lora_our_dataset_$lang"
results_folder="xglm_finetuned_our_dataset_$lang"
# python -u mmlu_ru.py --hf_model_id $model_name --k_shot 5 --lang "ru" --output_dir $results_folder
python -u mmlu_ru.py --hf_model_id $model_name --k_shot 5 --lang $lang --output_dir $results_folder
