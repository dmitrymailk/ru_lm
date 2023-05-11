import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import torch


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


if __name__ == "__main__":
    print("Starting evaluation.")
    compare_baseline_finetuned()
