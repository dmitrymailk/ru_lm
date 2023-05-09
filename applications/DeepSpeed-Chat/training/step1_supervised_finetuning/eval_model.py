import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import torch


def generate(
    model,
    tokenizer,
    inputs,
    num_beams=1,
    num_beam_groups=1,
    do_sample=False,
    num_return_sequences=1,
    max_new_tokens=100,
):
    with torch.no_grad():
        generate_ids = model.generate(
            inputs.input_ids.cuda(),
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
        )

        result = tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return result


def make_clean_name(name):
    return name.replace("/", "_")


def extract_answer(g_answer: str):
    search_str = "Assistant"
    search_index = g_answer.index(search_str) + len(search_str) + 1
    return g_answer[search_index:]


def compare_baseline_finetuned():
    ru_instruct = load_from_disk(
        "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/databricks_dolly_15k_translated_fixed"
    )
    ru_instruct = ru_instruct["test"]
    baseline_model = "facebook/xglm-1.7B"
    finetuned_model = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/models/xglm-1.7B_ru_v2"

    table = {
        "baseline_model": [],
        "finetuned_model": [],
        "prompt": [],
        "original_answer": [],
        "baseline_model_output": [],
        "finetuned_model_output": [],
    }

    def get_model(model_path, tokenizer):
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

        tokenizer.pad_token = tokenizer.eos_token
        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

        return model

    tokenizer = AutoTokenizer.from_pretrained(baseline_model, fast_tokenizer=True)

    model_baseline = get_model(baseline_model, tokenizer)
    model_finetuned = get_model(finetuned_model, tokenizer)
    model_baseline.eval()
    model_finetuned.eval()
    total_steps = 250
    print_results = True

    for i, example in enumerate(ru_instruct):
        print(f"{i+1}/{total_steps} - {(1+i)/total_steps*100}%")

        prompt = f"Human: {example['context_translated']} {example['instruction_translated']} Assistant: "

        answer = example["response_translated"]
        inputs = tokenizer(prompt, return_tensors="pt")
        baseline_result = generate(
            model_baseline,
            tokenizer=tokenizer,
            inputs=inputs,
        )
        baseline_result = baseline_result[0]
        baseline_result = extract_answer(baseline_result)

        finetuned_result = generate(
            model_finetuned,
            tokenizer=tokenizer,
            inputs=inputs,
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
            print(baseline_result)
            print(finetuned_result)
            print(answer)
            print("=" * 100)
            print("=" * 100)
        if i > total_steps:
            break

    clean_name = make_clean_name(baseline_model)
    pd.DataFrame(data=table).to_csv(f"{finetuned_model}/{clean_name}.csv", index=False)


if __name__ == "__main__":
    print("Starting evaluation.")
