import json

from datasets import load_dataset, Dataset
from tqdm import tqdm

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from instruct_models import SaigaConversation, generate


def eval_saiga7b():
    mt_bench_en = load_dataset("dim/mt_bench_en")
    mt_bench_en = mt_bench_en["train"]
    mt_bench_en = mt_bench_en.to_list()

    weights_path = "IlyaGusev/saiga_7b_lora"
    tokenizer_path = "IlyaGusev/saiga_7b_lora"

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
    conversation = SaigaConversation()
    conversation.add_user_message(inp)
    prompt = conversation.get_prompt(tokenizer)

    output = generate(model, tokenizer, prompt, generation_config)
    print(inp)
    print(output)

    # start evaluation

    for i in tqdm(range(len(mt_bench_en))):
        # print(item)
        if i > 2:
            break
        item = mt_bench_en[i]
        conversation = SaigaConversation()
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
        f"./datasets/final_evaluation_datasets/mt_bench/mt_bench_en_saiga_7b_en.json",
        "w",
        encoding="utf-8",
    ) as outfile:
        json.dump(mt_bench_en, outfile)


if __name__ == '__main__':
    eval_saiga7b()