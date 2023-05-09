from transformers import AutoTokenizer, M2M100ForConditionalGeneration


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    device = "cuda"
    model.to(device)
    model.enable_xformers_memory_efficient_attention()
    inputs = tokenizer(
        "Studies have been shown that owning a dog is good for you", return_tensors="pt"
    ).to(device)

    result = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id("ru"))

    print(tokenizer.batch_decode(result)[0])
