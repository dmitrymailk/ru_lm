import torch
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    AutoModelForSequenceClassification,
)

import numpy as np

import optuna

model_name = "gpt2"
gen_tokenizer = AutoTokenizer.from_pretrained(model_name)
gen_model = GPT2LMHeadModel.from_pretrained(model_name)
gen_model.cuda()

reward_name = "OpenAssistant/reward-model-deberta-v3-base"
rank_model = AutoModelForSequenceClassification.from_pretrained(reward_name)
rank_tokenizer = AutoTokenizer.from_pretrained(reward_name)
rank_model.cuda()

prompts = [
    "Explain nuclear fusion like I am five. Explanation:",
]


def objective(trial):
    top_p = trial.suggest_float("top_p", 0.1, 1.0)
    top_k = trial.suggest_int("top_k", 1, 300)

    evals = []

    for step in range(len(prompts)):
        prompt = prompts[step]
        input_ids = gen_tokenizer(prompt, return_tensors="pt").to("cuda").input_ids
        output = gen_model.generate(
            input_ids,
            do_sample=False,
            max_length=128,
            top_p=top_p,
            top_k=top_k,
        )
        result = gen_tokenizer.decode(output[0], skip_special_tokens=True)
        result = result[len(prompt) :]
        print(
            f"""
        PROMPT: {prompt}
        RESULT: {result}
        """
        )
        rank_inputs = rank_tokenizer(prompt, result, return_tensors="pt").to("cuda")
        rank_score = rank_model(**rank_inputs).logits[0].cpu().detach()
        trial.report(rank_score, step=step)
        evals.append(rank_score)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(evals)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print(study.best_params)
