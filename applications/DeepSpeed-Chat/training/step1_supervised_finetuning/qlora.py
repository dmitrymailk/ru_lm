from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        "timdettmers/guanaco-65b-merged",
        load_in_4bit=True,
        device_map="auto",
        # max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    input_text = "Напиши научную статью про реккурентную модель трансформера и предложи новые идеи"
    tokenizer = AutoTokenizer.from_pretrained("TheBloke/guanaco-65B-HF")
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    generated_result = model.generate(
        inputs.input_ids,
        max_new_tokens=2048,
        # repetition_penalty=1.1,
    )
    result = tokenizer.batch_decode(
        generated_result,
        skip_special_tokens=True,
    )
    print(result)
    # print(inputs)
