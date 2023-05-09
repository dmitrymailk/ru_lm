from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

if __name__ == "__main__":
    zero_model_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/models/xglm-4.5B_ru_v4/epoch=0_step=7"
    state_dict = get_fp32_state_dict_from_zero_checkpoint(zero_model_path)
    model = AutoModelForCausalLM.from_pretrained("facebook/xglm-4.5B")
    tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-4.5B")
    # model = model.cpu() # move to cpu
    model.load_state_dict(state_dict)
    save_dict = model.state_dict()
    output_model_file = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/models/xglm-4.5B_ru_v4/epoch=0_step=7/pytorch_model.bin"
    output_config_file = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/models/xglm-4.5B_ru_v4/epoch=0_step=7/config.json"
    output_dir = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/models/xglm-4.5B_ru_v4/epoch=0_step=7/"
    torch.save(save_dict, output_model_file)
    model.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)
    print("test")
