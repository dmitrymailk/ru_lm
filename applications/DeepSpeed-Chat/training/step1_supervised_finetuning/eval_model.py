import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
)
from datasets import load_dataset, load_from_disk
import torch
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from pathlib import Path

from transformers import GenerationConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from pathlib import Path


def generate(model, tokenizer, inputs, device="cuda"):
    with torch.no_grad():
        generate_ids = model.generate(
            inputs.input_ids.to(device),
            penalty_alpha=0.25,
            top_k=4,
            repetition_penalty=1.1,
            max_new_tokens=512,
        )

        result = tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return result


def make_clean_name(name):
    return name.replace("/", "_")


def extract_answer(g_answer: str):
    search_str = "Assistant"
    search_index = g_answer.index(search_str) + len(search_str) + 1
    return g_answer[search_index:]


def compare_baseline_finetuned():
    # ru_instruct = load_from_disk(
    #     "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/databricks_dolly_15k_translated_fixed"
    # )
    ru_instruct = load_dataset("IlyaGusev/ru_turbo_alpaca")
    ru_instruct = ru_instruct["train"].filter(lambda x: x["label"] == "ok")
    ru_instruct = ru_instruct.train_test_split(test_size=500, seed=42)
    ru_instruct = ru_instruct["test"]
    baseline_model = "facebook/xglm-4.5B"
    finetuned_model = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/models/xglm-4.5B_ru_v5/"

    table = {
        "baseline_model": [],
        "finetuned_model": [],
        "prompt": [],
        "original_answer": [],
        "baseline_model_output": [],
        "finetuned_model_output": [],
    }

    def get_model(model_path, tokenizer, device):
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.to(device)

        tokenizer.pad_token = tokenizer.eos_token
        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

        return model

    tokenizer = AutoTokenizer.from_pretrained(baseline_model, fast_tokenizer=True)

    model_baseline = get_model(baseline_model, tokenizer, "cuda:0")
    model_finetuned = get_model(finetuned_model, tokenizer, "cuda:1")
    model_baseline.eval()
    model_finetuned.eval()
    total_steps = len(ru_instruct)
    print_results = True

    for i, example in enumerate(ru_instruct):
        print(f"{i+1}/{total_steps} - {(1+i)/total_steps*100}%")

        prompt = f"Human: {example['instruction']} {example['input']} Assistant:"

        answer = example["output"]
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        baseline_result = generate(
            model_baseline, tokenizer=tokenizer, inputs=inputs, device="cuda:0"
        )
        baseline_result = baseline_result[0]
        baseline_result = extract_answer(baseline_result)

        finetuned_result = generate(
            model_finetuned, tokenizer=tokenizer, inputs=inputs, device="cuda:1"
        )
        finetuned_result = finetuned_result[0]
        finetuned_result = extract_answer(finetuned_result)

        table["baseline_model"].append(baseline_model)
        table["finetuned_model"].append(finetuned_model.split("/")[-1])
        table["baseline_model_output"].append(baseline_result)
        table["finetuned_model_output"].append(finetuned_result)
        table["original_answer"].append(answer)
        table["prompt"].append(prompt)
        if print_results:
            print()
            print(prompt)
            print("Baseline: ", baseline_result)
            print("Finetuned: ", finetuned_result)
            print("Real: ", answer)
            print("=" * 100)
            print("=" * 100)
        if i > total_steps:
            break

    clean_name = make_clean_name(baseline_model)
    pd.DataFrame(data=table).to_csv(f"{finetuned_model}/{clean_name}.csv", index=False)


def compare_pretrained_models():
    # ru_instruct = load_from_disk(
    #     "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/databricks_dolly_15k_translated_fixed"
    # )
    ru_instruct = load_dataset("IlyaGusev/ru_turbo_alpaca")
    ru_instruct = ru_instruct["train"].filter(lambda x: x["label"] == "ok")
    ru_instruct = ru_instruct.train_test_split(test_size=500, seed=42)
    ru_instruct = ru_instruct["test"]
    baseline_model = "IlyaGusev/llama_7b_ru_turbo_alpaca_lora"
    finetuned_model = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/models/xglm-4.5B_ru_v5/"

    table = {
        "baseline_model": [],
        "finetuned_model": [],
        "finetuned_prompt": [],
        "baseline_prompt": [],
        "original_answer": [],
        "baseline_model_output": [],
        "finetuned_model_output": [],
    }

    def get_model(model_path, tokenizer, device):
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.to(device)

        tokenizer.pad_token = tokenizer.eos_token
        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

        return model

    tokenizer_baseline = AutoTokenizer.from_pretrained(baseline_model)
    tokenizer_finetuned = AutoTokenizer.from_pretrained(finetuned_model)

    config = PeftConfig.from_pretrained(baseline_model)
    model_base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    generation_config = GenerationConfig.from_pretrained(baseline_model)
    model_baseline = PeftModel.from_pretrained(model_base, baseline_model)
    model_baseline.to("cuda:0")

    model_finetuned = get_model(finetuned_model, tokenizer_finetuned, "cuda:1")

    model_baseline.eval()
    model_finetuned.eval()

    total_steps = len(ru_instruct)
    print_results = True

    for i, example in enumerate(ru_instruct):
        print(f"{i+1}/{total_steps} - {(1+i)/total_steps*100}%")

        baseline_prompt = (
            f"Задание: {example['instruction']}\nВход: {example['input']}\nВыход:"
        )
        finetune_prompt = (
            f"Human: {example['instruction']} {example['input']} Assistant:"
        )

        answer = example["output"]

        ## generate baseline result
        inputs_baseline = tokenizer_baseline(
            baseline_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to("cuda:0")
        inputs_baseline = {
            k: v
            for k, v in inputs_baseline.items()
            if k in ("input_ids", "attention_mask")
        }
        baseline_result = model_baseline.generate(
            **inputs_baseline, generation_config=generation_config
        )[0]
        baseline_result = tokenizer_baseline.decode(
            baseline_result,
            skip_special_tokens=True,
        )

        ## generate finetuned result
        inputs_finetuned = tokenizer_finetuned(
            finetune_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        finetuned_result = generate(
            model_finetuned,
            tokenizer=tokenizer_finetuned,
            inputs=inputs_finetuned,
            device="cuda:1",
        )[0]
        finetuned_result = extract_answer(finetuned_result)

        table["baseline_model"].append(baseline_model)
        table["finetuned_model"].append(finetuned_model.split("/")[-1])
        table["baseline_model_output"].append(baseline_result)
        table["finetuned_model_output"].append(finetuned_result)
        table["original_answer"].append(answer)
        table["baseline_prompt"].append(baseline_prompt)
        table["finetuned_prompt"].append(finetune_prompt)
        if print_results:
            print()
            print(baseline_prompt)
            print("Baseline: ", baseline_result)
            print(finetune_prompt)
            print("Finetuned: ", finetuned_result)
            print("Real: ", answer)
            print("=" * 100)
            print("=" * 100)
        if i > total_steps:
            break

    clean_name = make_clean_name(baseline_model)
    clean_name_fine = make_clean_name(finetuned_model.split("/")[-1])
    pd.DataFrame(data=table).to_csv(
        f"{finetuned_model}/{clean_name}vs{clean_name_fine}.csv",
        index=False,
    )


def self_instruct_predict_qlora(
    model_name: str = None,
    print_results=True,
):
    peft_model_id = f"/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/models/{model_name}/adapter_model/"
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        return_dict=True,
        device_map="auto",
        # device_map={"": 0},
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, peft_model_id)

    ru_instruct = load_dataset("IlyaGusev/ru_turbo_alpaca")
    ru_instruct = ru_instruct["train"].filter(lambda x: x["label"] == "ok")
    ru_instruct = ru_instruct.train_test_split(test_size=500, seed=42)
    ru_instruct = ru_instruct["test"]

    table = {
        "finetuned_model": [],
        "prompt": [],
        "original_answer": [],
        "finetuned_model_output_1": [],
        # "finetuned_model_output_2": [],
        "gen_config_1": [],
        # "gen_config_2": [],
    }

    gen_config_1 = GenerationConfig(
        max_new_tokens=512,
        repetition_penalty=1.1,
    )
    # gen_config_2 = GenerationConfig(
    #     max_new_tokens=1024,
    #     repetition_penalty=1.1,
    #     top_k=20,
    #     top_p=0.95,
    # )

    for i, example in tqdm(enumerate(ru_instruct)):
        print(f"{i+1}/{len(ru_instruct)} - {(1+i)/len(ru_instruct)}%")

        prompt = f"Human:\n{example['instruction']}\n{example['input']}\nAssistant:"

        answer = example["output"]
        batch = tokenizer(
            prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to("cuda")

        output_tokens_1 = model.generate(**batch, generation_config=gen_config_1)
        finetuned_result_1 = tokenizer.decode(
            output_tokens_1[0],
            skip_special_tokens=True,
        )

        # output_tokens_2 = model.generate(**batch, generation_config=gen_config_2)
        # finetuned_result_2 = tokenizer.decode(
        #     output_tokens_2[0],
        #     skip_special_tokens=True,
        # )

        table["prompt"].append(prompt)
        table["finetuned_model"].append(model_name)
        table["finetuned_model_output_1"].append(finetuned_result_1)
        # table["finetuned_model_output_2"].append(finetuned_result_2)
        table["gen_config_1"].append(gen_config_1.to_dict())
        # table["gen_config_2"].append(gen_config_2.to_dict())
        table["original_answer"].append(answer)

        if print_results:
            print()
            print(prompt)
            print("=" * 50)
            print("Finetuned 1: ", finetuned_result_1)
            print("=" * 50)
            # print("Finetuned 2: ", finetuned_result_2)
            # print("=" * 50)
            print("Real: ", answer)
            print("=" * 100)
            print("=" * 100)

        # break

    clean_name = make_clean_name(model_name)
    pd.DataFrame(data=table).to_csv(f"{peft_model_id}{clean_name}.csv", index=False)


def evaluate_all_deepspeed_xglm_models(
    base_path: str = None,
):
    def add_special_tokens_v2(string):
        string = string.replace("\n", "</s>")
        return string

    def remove_special_tokens_v2(string):
        string = string.replace("</s>", "\n")
        string = string.replace("\n ", "\n")
        string = string.replace("<|endoftext|>", "")
        return string

    def encode_v2(text: str, tokenizer, special_tokens=True):
        text = add_special_tokens_v2(text)
        text = tokenizer.encode(text, add_special_tokens=special_tokens)
        return text

    def decode_v2(tokens: list[int], tokenizer):
        tokens = tokenizer.decode(tokens)
        tokens = remove_special_tokens_v2(tokens)
        return tokens

    ru_instruct = load_dataset("IlyaGusev/ru_turbo_alpaca")
    ru_instruct = ru_instruct["train"].filter(lambda x: x["label"] == "ok")
    ru_instruct = ru_instruct.train_test_split(test_size=500, seed=42)
    ru_instruct = ru_instruct["test"]

    base_folder = f"/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/models/{base_path}"

    p = Path(base_folder)
    model_dirs = sorted([x for x in p.iterdir() if x.is_dir()])

    gen_config_1 = GenerationConfig(
        max_new_tokens=512,
        repetition_penalty=1.1,
    )

    class StoppingCriteriaSub(StoppingCriteria):
        def __init__(self, stops, tokenizer, prompt):
            super().__init__()
            self.stops = stops
            self.tokenizer = tokenizer
            self.prompt = add_special_tokens_v2(prompt)
            self.prompt = tokenizer.decode(tokenizer.encode(self.prompt))

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
            for stop in self.stops:
                generated_temp_ids = input_ids.tolist()[0]
                if stop in tokenizer.decode(generated_temp_ids)[len(self.prompt) :]:
                    return True

            return False

    stop_words = [
        "<|endoftext|>",
        "Human:",
    ]

    for finetuned_model in model_dirs:
        finetuned_model = str(finetuned_model)
        model = AutoModelForCausalLM.from_pretrained(
            finetuned_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/xglm-4.5B",
            padding_side="left",
        )
        model = model.eval()
        clean_name = make_clean_name(finetuned_model.split("/")[-2:])
        table = {
            "finetuned_model": [],
            "prompt": [],
            "original_answer": [],
            "finetuned_model_output": [],
        }

        for i, example in tqdm(enumerate(ru_instruct)):
            print(f"{i+1}/{len(ru_instruct)} - {(1+i)/len(ru_instruct)}%")

            prompt = f"Human:\n{example['instruction']}\n{example['input']}\nAssistant:"

            stopping_criteria = StoppingCriteriaList(
                [
                    StoppingCriteriaSub(
                        stops=stop_words,
                        tokenizer=tokenizer,
                        prompt=prompt,
                    )
                ]
            )

            answer = example["output"]
            input_text = encode_v2(
                prompt,
                tokenizer=tokenizer,
            )
            input_text = torch.tensor([input_text]).to("cuda")

            output_tokens_1 = model.generate(
                input_text,
                generation_config=gen_config_1,
                stopping_criteria=stopping_criteria,
            )
            finetuned_result_1 = decode_v2(output_tokens_1[0], tokenizer=tokenizer)

            table["prompt"].append(prompt)
            table["finetuned_model"].append(clean_name)
            table["finetuned_model_output_1"].append(finetuned_result_1)
            table["gen_config_1"].append(gen_config_1.to_dict())
            table["original_answer"].append(answer)

            print()
            print(prompt)
            print("=" * 50)
            print("Finetuned 1: ", finetuned_result_1)
            print("=" * 50)
            print("Real: ", answer)
            print("=" * 100)
            print("=" * 100)

        pd.DataFrame(data=table).to_csv(
            f"{finetuned_model}/{clean_name}.csv",
            index=False,
        )


if __name__ == "__main__":
    print("Starting evaluation.")
    # compare_baseline_finetuned()
    # compare_pretrained_models()
    # self_instruct_predict_qlora(
    #     model_name="adapter_xglm_7.5B_v1/checkpoint-20000",
    # )
    evaluate_all_deepspeed_xglm_models(
        base_path="xglm-4.5B_ru_v10",
    )
