import json

from datasets import load_dataset, Dataset
from tqdm import tqdm

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from instruct_models import (
    SaigaConversation,
    generate,
    GoralConversation,
    XGLMConversation,
    GigaChatConversationAPI,
    YandexGPTAPI,
    ChatGPTConversationAPI,
)


def eval_saiga_based(
    weights_path=None,
    tokenizer_path=None,
    output_save_path=None,
    conversation_class=None,
    start_token_id=None,
    bot_token_id=None,
):
    assert not weights_path is None
    assert not tokenizer_path is None
    assert not output_save_path is None
    assert not conversation_class is None
    assert not start_token_id is None
    assert not bot_token_id is None

    mt_bench_en = load_dataset("dim/mt_bench_en")
    mt_bench_en = mt_bench_en["train"]
    mt_bench_en = mt_bench_en.to_list()

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
    conversation = conversation_class(
        start_token_id=start_token_id,
        bot_token_id=bot_token_id,
    )
    conversation.add_user_message(inp)
    prompt = conversation.get_prompt(tokenizer)

    output = generate(model, tokenizer, prompt, generation_config)
    print(inp)
    print(output)

    # start evaluation

    for i in tqdm(range(len(mt_bench_en))):
        # print(item)
        item = mt_bench_en[i]
        conversation = conversation_class(
            start_token_id=start_token_id,
            bot_token_id=bot_token_id,
        )
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


def eval_ru_saiga_based(
    weights_path=None,
    tokenizer_path=None,
    output_save_path=None,
    conversation_class=None,
    start_token_id=None,
    bot_token_id=None,
):
    assert not weights_path is None
    assert not tokenizer_path is None
    assert not output_save_path is None
    assert not conversation_class is None
    assert not start_token_id is None
    assert not bot_token_id is None

    mt_bench_en = load_dataset("dim/mt_bench_ru")
    mt_bench_en = mt_bench_en["train"]
    mt_bench_en = mt_bench_en.to_list()

    config = PeftConfig.from_pretrained(weights_path)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        load_in_8bit=True,
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
    conversation = conversation_class(
        start_token_id=start_token_id,
        bot_token_id=bot_token_id,
    )
    conversation.add_user_message(inp)
    prompt = conversation.get_prompt(tokenizer)

    output = generate(model, tokenizer, prompt, generation_config)
    print(inp)
    print(output)

    # start evaluation

    for i in tqdm(range(len(mt_bench_en))):
        # print(item)
        item = mt_bench_en[i]
        conversation = conversation_class(
            start_token_id=start_token_id,
            bot_token_id=bot_token_id,
        )
        mt_bench_en[i]["replies"] = []
        for turn in item["turns_ru"]:
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


def eval_xglm_based(
    model_path=None,
    output_save_path=None,
    conversation_class=None,
):
    assert model_path
    assert output_save_path

    mt_bench_en = load_dataset("dim/mt_bench_en")
    mt_bench_en = mt_bench_en["train"]
    mt_bench_en = mt_bench_en.to_list()

    # weights_path = "IlyaGusev/saiga_7b_lora"
    # tokenizer_path = "IlyaGusev/saiga_7b_lora"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
    )
    model = model.eval()

    # test generation
    print("***")
    print("Test generation")
    print("***")
    inp = "Почему трава зеленая?"
    conversation = conversation_class(
        model=model,
        tokenizer=tokenizer,
        debug_status=1,
    )
    result = conversation.chat(inp)
    print(result)

    # start evaluation

    for i in tqdm(range(len(mt_bench_en))):
        # print(item)
        # if i > 2:
        #     break
        item = mt_bench_en[i]
        conversation = conversation_class(
            model=model,
            tokenizer=tokenizer,
            debug_status=1,
        )
        mt_bench_en[i]["replies"] = []
        for turn in item["turns"]:
            print(turn)
            print("*" * 10)
            result = conversation.chat(turn)
            print(result)
            print("*" * 50)
            mt_bench_en[i]["replies"].append(result)
        print("=" * 100)

    with open(
        output_save_path,
        "w",
        encoding="utf-8",
    ) as outfile:
        json.dump(mt_bench_en, outfile)


def eval_ru_xglm_based(
    model_path=None,
    output_save_path=None,
    conversation_class=None,
):
    assert model_path
    assert output_save_path

    mt_bench_en = load_dataset("dim/mt_bench_ru")
    mt_bench_en = mt_bench_en["train"]
    mt_bench_en = mt_bench_en.to_list()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
    )
    model = model.eval()

    # test generation
    print("***")
    print("Test generation")
    print("***")
    inp = "Почему трава зеленая?"
    conversation = conversation_class(
        model=model,
        tokenizer=tokenizer,
        debug_status=1,
    )
    result = conversation.chat(inp)
    print(result)

    # start evaluation

    for i in tqdm(range(len(mt_bench_en))):
        # print(item)
        # if i > 2:
        #     break
        item = mt_bench_en[i]
        conversation = conversation_class(
            model=model,
            tokenizer=tokenizer,
            debug_status=1,
        )
        mt_bench_en[i]["replies"] = []
        for turn in item["turns_ru"]:
            print(turn)
            print("*" * 10)
            result = conversation.chat(turn)
            print(result)
            print("*" * 50)
            mt_bench_en[i]["replies"].append(result)
        print("=" * 100)

    with open(
        output_save_path,
        "w",
        encoding="utf-8",
    ) as outfile:
        json.dump(mt_bench_en, outfile)


def gigachat_eval_based(
    lang="en",
    output_save_path=None,
):
    assert not output_save_path is None
    mt_bench = None
    if lang == "en":
        mt_bench = load_dataset("dim/mt_bench_en")
    elif lang == "ru":
        mt_bench = load_dataset("dim/mt_bench_ru")
    else:
        assert False, "Language is not supported"

    mt_bench = mt_bench["train"]
    mt_bench = mt_bench.to_list()

    # test generation
    print("***")
    print("Test generation")
    print("***")
    inp = "Почему трава зеленая?"
    conversation = GigaChatConversationAPI()
    result = conversation.send_message(inp)
    print(inp)
    print(result)

    for i in tqdm(range(len(mt_bench))):
        # print(item)
        item = mt_bench[i]
        mt_bench[i]["replies"] = []

        turns_field_name = "turns"
        if lang == "ru":
            turns_field_name = "turns_ru"
        conversation = GigaChatConversationAPI()
        for turn in item[turns_field_name]:
            print(turn)
            print("*" * 10)

            output = conversation.send_message(turn)
            print(output)
            print("*" * 50)
            mt_bench[i]["replies"].append(output)
        print("=" * 100)

    with open(
        output_save_path,
        "w",
        encoding="utf-8",
    ) as outfile:
        json.dump(mt_bench, outfile)


def yandexgpt_eval_based(
    lang="en",
    output_save_path=None,
):
    assert not output_save_path is None
    mt_bench = None
    if lang == "en":
        mt_bench = load_dataset("dim/mt_bench_en")
    elif lang == "ru":
        mt_bench = load_dataset("dim/mt_bench_ru")
    else:
        assert False, "Language is not supported"

    mt_bench = mt_bench["train"]
    mt_bench = mt_bench.to_list()

    # test generation
    print("***")
    print("Test generation")
    print("***")
    inp = "Почему трава зеленая?"
    conversation = YandexGPTAPI()
    result = conversation.send_chat(inp)
    print(inp)
    print(result)

    for i in tqdm(range(len(mt_bench))):
        # print(item)
        item = mt_bench[i]
        mt_bench[i]["replies"] = []

        turns_field_name = "turns"
        if lang == "ru":
            turns_field_name = "turns_ru"
        conversation = YandexGPTAPI()
        for turn in item[turns_field_name]:
            print(turn)
            print("*" * 10)

            output = conversation.send_chat(turn)
            print(output)
            print("*" * 50)
            mt_bench[i]["replies"].append(output)
        print("=" * 100)

    with open(
        output_save_path,
        "w",
        encoding="utf-8",
    ) as outfile:
        json.dump(mt_bench, outfile)


def chat_gpt_eval_based(lang="en", output_save_path=None):
    assert not output_save_path is None

    if lang == "en":
        mt_bench = load_dataset("dim/mt_bench_en")
    elif lang == "ru":
        mt_bench = load_dataset("dim/mt_bench_ru")
    else:
        raise ValueError("Language not supported")

    mt_bench = mt_bench["train"]
    mt_bench = mt_bench.to_list()

    print("Test generation")
    inp = "Почему трава зеленая?"
    api = ChatGPTConversationAPI()
    response = api.send_message(inp)
    print(inp)
    print(response)

    for i in tqdm(range(len(mt_bench))):
        item = mt_bench[i]
        mt_bench[i]["replies"] = []

        turns_field = "turns" if lang == "en" else "turns_ru"

        api = ChatGPTConversationAPI()
        for turn in item[turns_field]:
            print(turn)
            print("*" * 10)

            response = api.send_message(turn)
            print(response)
            print("*" * 50)

            mt_bench[i]["replies"].append(response)

        print("=" * 100)

    with open(output_save_path, "w", encoding="utf-8") as f:
        json.dump(mt_bench, f)


if __name__ == "__main__":
    # weights_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/rulm/self_instruct/models/rugpt_v1/checkpoint-4400/adapter_model"
    # weights_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/rulm/self_instruct/models/goral_xglm_v2/checkpoint-4700/adapter_model"
    # weights_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/rulm/self_instruct/models/goral_xglm_v2/checkpoint-5000/adapter_model/"
    # weights_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/rulm/rulm2/rulm/self_instruct/models/saiga2_v2/checkpoint-4900/adapter_model"
    # weights_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/rulm/rulm2/rulm/self_instruct/models/saiga2_v2"
    # weights_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/rulm/rulm2/rulm/self_instruct/models/saiga2_13b_v4"
    # weights_path = "IlyaGusev/saiga2_7b_lora"
    # weights_path = "IlyaGusev/saiga2_13b_lora"
    weights_path = "IlyaGusev/gigasaiga_lora"

    # tokenizer_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/rulm/self_instruct/models/rugpt_v1"
    # tokenizer_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/rulm/self_instruct/models/goral_xglm_v2/"
    # tokenizer_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/rulm/self_instruct/models/goral_xglm_v2/"
    # tokenizer_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/rulm/rulm2/rulm/self_instruct/models/saiga2_v2/"
    # tokenizer_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/rulm/rulm2/rulm/self_instruct/models/saiga2_v2"
    # tokenizer_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/rulm/rulm2/rulm/self_instruct/models/saiga2_13b_v4"
    # tokenizer_path = "IlyaGusev/saiga2_7b_lora"
    # tokenizer_path = "IlyaGusev/saiga2_13b_lora"
    tokenizer_path = "IlyaGusev/gigasaiga_lora"

    # output_save_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/final_evaluation_datasets/mt_bench/mt_bench_en_rugpt_13B_our_dataset.json"
    # output_save_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/final_evaluation_datasets/mt_bench/mt_bench_ru_xglm_4.5B_saiga_dataset.json"
    # output_save_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/final_evaluation_datasets/mt_bench/mt_bench_ru_xglm_4.5B_lora_saiga_dataset.json"
    # output_save_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/final_evaluation_datasets/mt_bench/mt_bench_en_saiga2_7b_our_dataset.json"
    # output_save_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/final_evaluation_datasets/mt_bench/mt_bench_ru_saiga2_7b.json"
    # output_save_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/final_evaluation_datasets/mt_bench/mt_bench_ru_gigasaiga_13b.json"
    # output_save_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/final_evaluation_datasets/mt_bench/mt_bench_ru_yandexgpt.json"
    output_save_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/final_evaluation_datasets/mt_bench/mt_bench_ru_chatgpt.json"
    # ----
    # rugpt
    start_token_id = 2
    # rugpt
    bot_token_id = 46787
    # # ----
    # # xglm
    # start_token_id = 0
    # # xglm
    # bot_token_id = 7425
    # # ----
    # # saiga
    # start_token_id = 1
    # # saiga
    # bot_token_id = 9225

    # conversation_class = GoralConversation
    conversation_class = SaigaConversation

    # eval_saiga_based(
    #     weights_path=weights_path,
    #     tokenizer_path=tokenizer_path,
    #     output_save_path=output_save_path,
    #     conversation_class=conversation_class,
    #     start_token_id=start_token_id,
    #     bot_token_id=bot_token_id,
    # )
    # eval_ru_saiga_based(
    #     weights_path=weights_path,
    #     tokenizer_path=tokenizer_path,
    #     output_save_path=output_save_path,
    #     conversation_class=conversation_class,
    #     start_token_id=start_token_id,
    #     bot_token_id=bot_token_id,
    # )
    # eval_xglm_based(
    #     model_path="/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/models/xglm-4.5B_ru_v10/epoch=6_step=41141",
    #     output_save_path="/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/final_evaluation_datasets/mt_bench/mt_bench_en_xglm_4.5b_v10_epoch_6_step_41141.json",
    #     conversation_class=XGLMConversation,
    # )
    # eval_ru_xglm_based(
    #     model_path="/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/models/xglm-4.5B_ru_v10/epoch=6_step=41141",
    #     output_save_path="/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/final_evaluation_datasets/mt_bench/mt_bench_ru_xglm_4.5b_v10_epoch_6_step_41141.json",
    #     conversation_class=XGLMConversation,
    # )
    # gigachat_eval_based(
    #     # lang="ru",
    #     lang="en",
    #     output_save_path=output_save_path,
    # )
    # yandexgpt_eval_based(
    #     lang="ru",
    #     output_save_path=output_save_path,
    # )
    chat_gpt_eval_based(lang="ru", output_save_path=output_save_path)
