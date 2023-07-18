import json

from datasets import load_dataset, Dataset
from tqdm import tqdm

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from instruct_models import SaigaConversation, generate, GoralConversation


def eval_saiga_based(
    weights_path=None,
    tokenizer_path=None,
    output_save_path=None,
    conversation_class=None,
):
    assert weights_path
    assert tokenizer_path
    assert output_save_path

    mt_bench_en = load_dataset("dim/mt_bench_en")
    mt_bench_en = mt_bench_en["train"]
    mt_bench_en = mt_bench_en.to_list()

    # weights_path = "IlyaGusev/saiga_7b_lora"
    # tokenizer_path = "IlyaGusev/saiga_7b_lora"

    config = PeftConfig.from_pretrained(weights_path)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        # load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, weights_path, torch_dtype=torch.float16)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    generation_config = GenerationConfig.from_pretrained(tokenizer_path)
    generation_config.do_sample = False

    # test generation
    print("***")
    print("Test generation")
    print("***")
    inp = "Почему трава зеленая?"
    conversation = conversation_class()
    conversation.add_user_message(inp)
    prompt = conversation.get_prompt(tokenizer)

    output = generate(model, tokenizer, prompt, generation_config)
    print(inp)
    print(output)

    # start evaluation

    for i in tqdm(range(len(mt_bench_en))):
        # print(item)
        item = mt_bench_en[i]
        conversation = conversation_class()
        mt_bench_en[i]["replies"] = []
        for turn in item["turns"]:
            print(turn)
            print("*" * 10)
            conversation.add_user_message(turn)
            prompt = conversation.get_prompt(tokenizer)

            output = generate(model, tokenizer, prompt, generation_config)
            conversation.add_bot_message(output)
            print(output)
            print("*" * 50)
            mt_bench_en[i]["replies"].append(output)
        print("=" * 100)

    with open(
        output_save_path,
        "w",
        encoding="utf-8",
    ) as outfile:
        json.dump(mt_bench_en, outfile)


if __name__ == "__main__":
    eval_saiga_based(
        weights_path="/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/rulm/self_instruct/models/saiga_7b_v2/checkpoint-4850/adapter_model",
        tokenizer_path="/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/rulm/self_instruct/models/saiga_7b_v2",
        output_save_path="/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/final_evaluation_datasets/mt_bench/mt_bench_en_saiga_7b_v2.json",
        conversation_class=GoralConversation,
    )
