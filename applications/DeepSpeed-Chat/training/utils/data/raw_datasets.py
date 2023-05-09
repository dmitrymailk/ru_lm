# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from datasets import load_dataset, load_from_disk
from torch.utils.data import Subset
import re


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):
    def __init__(self, output_path, seed, local_rank):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return


# English dataset
class DahoasRmstaticDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "Dahoas/rm-static"
        self.dataset_name_clean = "Dahoas_rm_static"
        self.raw_datasets = load_dataset("Dahoas/rm-static")

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample["prompt"]

    def get_chosen(self, sample):
        return sample["chosen"]

    def get_rejected(self, sample):
        return sample["rejected"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"] + sample["chosen"]

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"] + sample["rejected"]


# English dataset
class DahoasFullhhrlhfDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "Dahoas/full-hh-rlhf"
        self.dataset_name_clean = "Dahoas_full_hh_rlhf"
        self.raw_datasets = load_dataset("Dahoas/full-hh-rlhf")

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample["prompt"]

    def get_chosen(self, sample):
        return sample["chosen"]

    def get_rejected(self, sample):
        return sample["rejected"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"] + sample["chosen"]

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"] + sample["rejected"]


# English dataset
class DahoasSyntheticinstructgptjpairwiseDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "Dahoas/synthetic-instruct-gptj-pairwise"
        self.dataset_name_clean = "Dahoas_synthetic_instruct_gptj_pairwise"
        self.raw_datasets = load_dataset("Dahoas/synthetic-instruct-gptj-pairwise")
        self.splitted_dataset = self.raw_datasets["train"].train_test_split(
            train_size=0.98,
            test_size=0.02,
            seed=seed,
        )

    def get_train_data(self):
        return self.splitted_dataset["train"]

    def get_eval_data(self):
        return self.splitted_dataset["test"]

    def get_prompt(self, sample):
        return " Human: " + sample["prompt"] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample["chosen"]

    def get_rejected(self, sample):
        return " " + sample["rejected"]

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample["prompt"] + " Assistant: " + sample["chosen"]

    def get_prompt_and_rejected(self, sample):
        return " Human: " + sample["prompt"] + " Assistant: " + sample["rejected"]


# English dataset
class YitingxieRlhfrewarddatasetsDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "yitingxie/rlhf-reward-datasets"
        self.dataset_name_clean = "yitingxie_rlhf_reward_datasets"
        self.raw_datasets = load_dataset("yitingxie/rlhf-reward-datasets")

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample["prompt"] + "Assistant:"

    def get_chosen(self, sample):
        return sample["chosen"].split("Assistant:")[-1]

    def get_rejected(self, sample):
        return sample["rejected"].split("Assistant:")[-1]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"] + sample["chosen"]

    def get_prompt_and_rejected(self, sample):
        return sample["prompt"] + sample["rejected"]


# English dataset
class OpenaiWebgptcomparisonsDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "openai/webgpt_comparisons"
        self.dataset_name_clean = "openai_webgpt_comparisons"
        self.raw_datasets = load_dataset("openai/webgpt_comparisons")
        self.splitted_dataset = self.raw_datasets["train"].train_test_split(
            train_size=0.98,
            test_size=0.02,
            seed=seed,
        )

    def get_train_data(self):
        self.splitted_dataset["train"]

    def get_eval_data(self):
        self.splitted_dataset["test"]

    def get_prompt(self, sample):
        return " Human: " + sample["question"]["full_text"] + " Assistant:"

    def get_chosen(self, sample):
        if float(sample["score_0"]) >= float(sample["score_1"]):
            response = sample["answer_0"]
        else:
            response = sample["answer_1"]
        # This data has citation square brackets and numbers (e.g., "[1]").
        # Right now we are not doing browser-assisted finetuning, thus we
        # remove these citations to avoid confusing the model.
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " " + response

    def get_rejected(self, sample):
        if float(sample["score_0"]) < float(sample["score_1"]):
            response = sample["answer_0"]
        else:
            response = sample["answer_1"]
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " " + response

    def get_prompt_and_chosen(self, sample):
        if float(sample["score_0"]) >= float(sample["score_1"]):
            response = sample["answer_0"]
        else:
            response = sample["answer_1"]
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " Human: " + sample["question"]["full_text"] + " Assistant: " + response

    def get_prompt_and_rejected(self, sample):
        if float(sample["score_0"]) < float(sample["score_1"]):
            response = sample["answer_0"]
        else:
            response = sample["answer_1"]
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " Human: " + sample["question"]["full_text"] + " Assistant: " + response


# English dataset
class StanfordnlpSHPDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "stanfordnlp/SHP"
        self.dataset_name_clean = "stanfordnlp_SHP"
        self.raw_datasets = load_dataset("stanfordnlp/SHP")

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return " Human: " + sample["history"] + " Assistant:"

    def get_chosen(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_A"]
        else:
            response = sample["human_ref_B"]
        return " " + response

    def get_rejected(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_B"]
        else:
            response = sample["human_ref_A"]
        return " " + response

    def get_prompt_and_chosen(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_A"]
        else:
            response = sample["human_ref_B"]
        return " Human: " + sample["history"] + " Assistant: " + response

    def get_prompt_and_rejected(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_B"]
        else:
            response = sample["human_ref_A"]
        return " Human: " + sample["history"] + " Assistant: " + response


# russian dataset
class RuInstructTranslated(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "self_instruct_translated"
        self.dataset_name_clean = "self_instruct_translated"
        self.raw_datasets = load_from_disk(
            "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/self_instruct_translated/"
        )

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return f"Human: {sample['prompt_translated']} Assistant:"

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return f" {sample['completion_translated']}"

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return self.get_prompt(sample) + self.get_chosen(sample)

    def get_prompt_and_rejected(self, sample):
        return


# russian dataset
class RuDollyInstructTranslated(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "dolly_translated_prompt"
        self.dataset_name_clean = "dolly_translated_prompt"
        self.raw_datasets = load_from_disk(
            "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/prompt_datasets/dolly_translated_prompt"
        )
        self.raw_datasets = self.raw_datasets.train_test_split(test_size=100, seed=42)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"]


# russian dataset
class RuChip2Translated(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "chip2_instruct_alpha_prompt_ru"
        self.dataset_name_clean = "chip2_instruct_alpha_prompt_ru"
        self.raw_datasets = load_from_disk(
            "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/prompt_datasets/chip2_instruct_alpha_prompt_ru"
        )
        self.raw_datasets = self.raw_datasets.train_test_split(test_size=100, seed=42)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"]


# russian dataset
class RuOpenAssTranslated(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "openass_prompt_dataset_ru"
        self.dataset_name_clean = "openass_prompt_dataset_ru"
        self.raw_datasets = load_from_disk(
            "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/prompt_datasets/openass_prompt_dataset_ru"
        )
        self.raw_datasets = self.raw_datasets.train_test_split(test_size=100, seed=42)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"]


# english dataset
class EnDollyInstructTranslated(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "dolly_original_prompt"
        self.dataset_name_clean = "dolly_original_prompt"
        self.raw_datasets = load_from_disk(
            "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/prompt_datasets/dolly_original_prompt"
        )
        self.raw_datasets = self.raw_datasets.train_test_split(test_size=100, seed=42)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"]


# english dataset
class EnChip2Translated(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "chip2_instruct_alpha_prompt_en"
        self.dataset_name_clean = "chip2_instruct_alpha_prompt_en"
        self.raw_datasets = load_from_disk(
            "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/prompt_datasets/chip2_instruct_alpha_prompt_en"
        )
        self.raw_datasets = self.raw_datasets.train_test_split(test_size=100, seed=42)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"]


# english dataset
class EnOpenAssTranslated(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "openass_prompt_dataset_en"
        self.dataset_name_clean = "openass_prompt_dataset_en"
        self.raw_datasets = load_from_disk(
            "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/prompt_datasets/openass_prompt_dataset_en"
        )
        self.raw_datasets = self.raw_datasets.train_test_split(test_size=100, seed=42)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"]


# english dataset
class EnInstructTranslated(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "self_instruct_en"
        self.dataset_name_clean = "self_instruct_en"
        self.raw_datasets = load_from_disk(
            "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/datasets/self_instruct_translated/"
        )

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return f"Human: {sample['prompt']} Assistant:"

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return f" {sample['completion']}"

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return self.get_prompt(sample) + self.get_chosen(sample)

    def get_prompt_and_rejected(self, sample):
        return
