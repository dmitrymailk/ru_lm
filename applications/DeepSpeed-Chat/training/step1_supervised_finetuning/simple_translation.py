import os
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from easynmt import EasyNMT
from optimum.bettertransformer import BetterTransformer
from datasets import load_from_disk, load_dataset
import os
from tqdm import tqdm
import json
import pandas as pd


class Translator:
    def __init__(self, model_name: str, device="cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.max_length = 2048
        self.init()

    def init(self):
        print(f"Init model. {self.device}")
        if self.model_name in ["facebook/nllb-200-3.3B", "facebook/wmt21-dense-24-wide-en-x",]:
            # device = int(self.device.split(":")[1])
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                use_auth_token=True,
                # load_in_8bit=True,
                # device_map={"": device}
            )
            # self.model = BetterTransformer.transform(self.model)
            self.model = self.model.half()
            self.model = self.model.eval()
            self.model = torch.compile(self.model, mode="reduce-overhead")
            self.model = self.model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_auth_token=True,
            )
        elif self.model_name == "opus-mt":
            self.model = EasyNMT(self.model_name)

        print("Model is initialized.")

    def translate(self, text: str):
        func_map = {
            "facebook/nllb-200-3.3B": self.nllb_translate,
            "opus-mt": self.opusmt_translate,
            "facebook/wmt21-dense-24-wide-en-x": self.wmt21_translate,
        }
        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        with torch.no_grad():
            return func_map[self.model_name](text)

    def __call__(self, text: str):
        return self.translate(text=text)

    def nllb_translate(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = self.to_device(inputs=inputs)
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id["rus_Cyrl"],
            max_new_tokens=self.max_length,
        )
        return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[
            0
        ]

    def opusmt_translate(self, text: str):
        return self.model.translate(text, source_lang="en", target_lang="ru")

    def wmt21_translate(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        inputs = self.to_device(inputs=inputs)
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.get_lang_id("ru"),
            max_new_tokens=self.max_length,
        )
        return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[
            0
        ]

    def to_device(self, inputs):
        for key in inputs.keys():
            inputs[key] = inputs[key].to(self.device)
        return inputs


def translate_dolly():
    base_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/"
    save_folder = "dolly_translated"
    full_path = f"{base_path}{save_folder}/"
    file_name = "dolly_translated.json"
    assert os.path.isdir(full_path)

    data = load_dataset("databricks/databricks-dolly-15k")
    data = data["train"]
    # data = data.select(range(5))

    fields = ["context", "instruction", "response"]

    model_name = "facebook/wmt21-dense-24-wide-en-x"
    translator = Translator(
        model_name=model_name,
        device="cuda:1",
    )

    translated_examples = []
    for example in tqdm(data):
        for field in fields:
            text = example[field]
            translated = translator(text=text)
            example[f"{field}_translated"] = translated
        translated_examples.append(example)

    with open(f"{full_path}{file_name}", "w", encoding="utf-8") as outfile:
        json.dump(translated_examples, outfile)


def translate_openass():
    base_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/"
    save_folder = "openass_translated_en2ru"
    full_path = f"{base_path}{save_folder}/"
    file_name = "openass_translated_en2ru.json"
    assert os.path.isdir(full_path)

    dataset = pd.read_json(
        path_or_buf="/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/pure_datasets/2023-04-12_oasst_all.messages.jsonl",
        lines=True,
    )
    dataset = dataset[dataset["lang"] == "en"]

    model_name = "facebook/wmt21-dense-24-wide-en-x"
    translator = Translator(
        model_name=model_name,
        device="cuda:2",
    )

    translated_examples = []
    for i in tqdm(range(len(dataset))):
        example = dataset.iloc[i].to_dict()
        text = example["text"]
        translated = translator(text=text)
        example[f"text_translated"] = translated
        translated_examples.append(example)
        # if i > 20:
        #     break

    with open(f"{full_path}{file_name}", "w", encoding="utf-8") as outfile:
        json.dump(translated_examples, outfile)
    print()


def translate_chip2():
    base_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/"
    save_folder = "chip2_instruct_alpha"
    full_path = f"{base_path}{save_folder}/"
    file_name = "chip2_instruct_alpha_v6a_4_translated.json"
    assert os.path.isdir(full_path)

    dataset = pd.read_json(
        path_or_buf="/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/pure_datasets/chip2_instruct_alpha/chip2_instruct_alpha_v6a_4_formatted.json",
    )

    model_name = "facebook/wmt21-dense-24-wide-en-x"
    translator = Translator(
        model_name=model_name,
        device="cuda:3",
    )

    translated_examples = []
    fields = ["user", "bot"]
    for i in tqdm(range(len(dataset))):
        example = dataset.iloc[i].to_dict()
        for field in fields:
            text = example[field]
            translated = translator(text=text)
            example[f"{field}_translated"] = translated
        translated_examples.append(example)
        # if i > 20:
        #     break

    with open(f"{full_path}{file_name}", "w", encoding="utf-8") as outfile:
        json.dump(translated_examples, outfile)
    print()


if __name__ == "__main__":
    print("Start translation")
    # translate_openass()
    translate_chip2()
