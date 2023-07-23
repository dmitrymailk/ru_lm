import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    AutoModelForSequenceClassification,
)

import numpy as np

import optuna


from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

DEFAULT_MESSAGE_TEMPLATE = " <s> {role}\n{content} </s>\n"
DEFAULT_SYSTEM_PROMPT = "Ты — Горал, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."


class Conversation:
    def __init__(
        self,
        message_template=DEFAULT_MESSAGE_TEMPLATE,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        start_token_id=0,
        bot_token_id=7425,
    ):
        self.message_template = message_template
        self.start_token_id = start_token_id
        self.bot_token_id = bot_token_id
        self.messages = [{"role": "system", "content": system_prompt}]

    def get_start_token_id(self):
        return self.start_token_id

    def get_bot_token_id(self):
        return self.bot_token_id

    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def add_bot_message(self, message):
        self.messages.append({"role": "bot", "content": message})

    def get_prompt(self, tokenizer):
        final_text = ""
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        final_text += tokenizer.decode(
            [
                self.start_token_id,
            ]
        )
        final_text += " "
        final_text += tokenizer.decode([self.bot_token_id])
        return final_text.strip()


def generate(model, tokenizer, prompt, generation_config):
    data = tokenizer(prompt, return_tensors="pt")
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(**data, generation_config=generation_config)[0]
    output_ids = output_ids[len(data["input_ids"][0]) :]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()


weights_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/rulm/self_instruct/models/goral_xglm_4.5B/checkpoint-2950/adapter_model"
tokenizer_path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/rulm/self_instruct/models/goral_xglm_4.5B"

config = PeftConfig.from_pretrained(weights_path)
gen_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    # load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
gen_model = PeftModel.from_pretrained(
    gen_model, weights_path, torch_dtype=torch.float16
)
gen_model.eval()
gen_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
rank_model = AutoModelForSequenceClassification.from_pretrained(reward_name)
rank_tokenizer = AutoTokenizer.from_pretrained(reward_name)
rank_model.cuda()

prompts = [
    "Explain nuclear fusion like I am five.",
    "I just came out of from jail, any suggestion of my future?",
    "What is Depreciation",
]


def objective(trial):
    # top_p = trial.suggest_float("top_p", 0.1, 1.0)
    # temperature = trial.suggest_float("temperature", 0.1, 2.0)
    # repetition_penalty = trial.suggest_float("repetition_penalty", 0.1, 1.5)
    # top_k = trial.suggest_int("top_k", 1, 300)
    # constructive search
    penalty_alpha = trial.suggest_float("penalty_alpha", 0.3, 0.9)
    top_k = trial.suggest_int("top_k", 1, 10)
    repetition_penalty = trial.suggest_float("repetition_penalty", 0.9, 1.1)
    evals = []

    for step in range(len(prompts)):
        initial_prompt = prompts[step]
        conversation = Conversation()
        conversation.add_user_message(initial_prompt)
        prompt = conversation.get_prompt(gen_tokenizer)
        # print("PROMPT", prompt)
        generation_config = GenerationConfig(
            max_new_tokens=512,
            # no_repeat_ngram_size=no_repeat_ngram_size,
            # repetition_penalty=repetition_penalty,
            # temperature=temperature,
            # top_k=top_k,
            # top_p=top_p,
            # -----  constructive search
            penalty_alpha=penalty_alpha,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        result = generate(gen_model, gen_tokenizer, prompt, generation_config)
        print(
            f"""
        PROMPT: {initial_prompt}
        RESULT: {result}
        """
        )
        rank_inputs = rank_tokenizer(initial_prompt, result, return_tensors="pt").to(
            "cuda"
        )
        rank_score = rank_model(**rank_inputs).logits[0].cpu().detach()
        rank_score = float(rank_score)
        print("SCORE", rank_score)
        trial.report(rank_score, step=step)
        evals.append(rank_score)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(evals)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=300)

print(study.best_params)
# обычный подбор 100 попыток
# {'top_p': 0.8462093887585331, 'temperature': 0.8854714157162881, 'repetition_penalty': 1.0390300295155908, 'top_k': 218}

# constructive
# {'penalty_alpha': 0.15179011343178722, 'top_k': 2, 'repetition_penalty': 1.2203762978220836}
# {'penalty_alpha': 0.45662610978236967, 'top_k': 4, 'repetition_penalty': 1.025161759659547}
# {'penalty_alpha': 0.46017593114309874, 'top_k': 6, 'repetition_penalty': 0.9919855327156304}
# {'penalty_alpha': 0.3116840503163154, 'top_k': 19, 'repetition_penalty': 1.04177069065127}