import os
import torch
import logging
import time
import os

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from easynmt import EasyNMT
from optimum.bettertransformer import BetterTransformer
from datasets import load_from_disk, load_dataset
import os
import pandas as pd
from multiprocessing.pool import Pool
import random
import itertools

logger = logging.getLogger(__name__)


class Translator:
    def __init__(self, model_name: str, device="cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.init()

    def init(self):
        logger.info(f"Init model. {self.device}")
        if self.model_name == "facebook/nllb-200-3.3B":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                use_auth_token=True,
            )
            self.model = BetterTransformer.transform(self.model)
            self.model.eval()
            self.model = torch.compile(self.model)
            self.model = self.model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_auth_token=True,
            )
        elif self.model_name == "facebook/wmt21-dense-24-wide-en-x":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                use_auth_token=True,
            )
            self.model = BetterTransformer.transform(self.model)
            self.model.eval()
            self.model = torch.compile(self.model)
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
        )
        return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[
            0
        ]

    def opusmt_translate(self, text: str):
        return self.model.translate(text, source_lang="en", target_lang="ru")

    def wmt21_translate(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = self.to_device(inputs=inputs)
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.get_lang_id("ru"),
        )
        return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[
            0
        ]

    def to_device(self, inputs):
        for key in inputs.keys():
            inputs[key] = inputs[key].to(self.device)
        return inputs


def translate(params):
    device, dataset_subset = params
    fields = ["context", "instruction", "response"]
    translated_examples = []
    model_name = "facebook/wmt21-dense-24-wide-en-x"

    translator = Translator(model_name=model_name, device=device)

    for i, example in enumerate(dataset_subset):
        for field in fields:
            text = example[field]
            translated = translator(text=text)
            example[f"{field}_translated"] = translated
        translated_examples.append(example)
    return translated_examples


if __name__ == "__main__":
    # really slow method, i don't know why
    print("Start translation")
    data = load_dataset("databricks/databricks-dolly-15k")
    data = data["train"]
    data = data.select(range(50))

    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    devices = [devices[0]]
    with Pool(
        processes=len(devices),
    ) as pool:
        my_tasks = []
        relu = lambda x: x if x < len(data) else len(data)
        chunk_size = len(data) // len(devices) + 1

        for device, start in zip(
            devices,
            range(0, len(data), chunk_size),
        ):
            subset = data.select(range(start, relu(start + chunk_size)))

            my_tasks.append([device, subset])
        start_time = time.time()
        result = pool.map(translate, my_tasks, chunksize=1)
        merged = list(itertools.chain(*result))
        total_time = time.time() - start_time

        # print(merged)
        print(len(merged))
        print(total_time)
