import torch
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

import optuna

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_ids = tokenizer("DeepMind Company is", return_tensors="pt").input_ids
model = GPT2LMHeadModel.from_pretrained(model_name)

output = model.generate(input_ids, max_length=128)

prompts = [
    "DeepMind Company is",
    "Russia is",
]


def objective(trial):
    shuffle = trial.suggest_categorical("shuffle", [True, False])
    penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
    penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])

    for step in range(len(prompts)):
        output = model.generate(
            input_ids,
            do_sample=True,
            max_length=128,
            top_p=0.95,
            top_k=0,
        )
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        print(result)
        intermediate_value = 1
        trial.report(intermediate_value, step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return 2


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print(study.best_params)
