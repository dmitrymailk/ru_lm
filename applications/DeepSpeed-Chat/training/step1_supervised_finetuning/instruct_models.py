import os

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import gc
from transformers import StoppingCriteria, StoppingCriteriaList


class GoralConversation:
    def __init__(
        self,
        message_template=" <s> {role}\n{content} </s>\n",
        system_prompt="Ты — Горал, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.",
        start_token_id=1,
        bot_token_id=9225,
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


class SaigaConversation:
    def __init__(
        self,
        message_template="<s>{role}\n{content}</s>\n",
        system_prompt="Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.",
        start_token_id=1,
        bot_token_id=9225,
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
        final_text += tokenizer.decode([self.start_token_id, self.bot_token_id])
        return final_text.strip()


def generate(model, tokenizer, prompt, generation_config):
    data = tokenizer(prompt, return_tensors="pt")
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(**data, generation_config=generation_config)[0]
    output_ids = output_ids[len(data["input_ids"][0]) :]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()


def add_special_tokens_v2(string):
    string = string.replace("\n", "</s>")
    return string


def remove_special_tokens_v2(string):
    string = string.replace("</s>", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace("<|endoftext|>", "")
    return string


def encode_v2(text: str, tokenizer, special_tokens=True):
    text = add_special_tokens_v2(text)
    text = tokenizer.encode(text, add_special_tokens=special_tokens)
    return text


def decode_v2(tokens: list[int], tokenizer):
    tokens = tokenizer.decode(tokens)
    tokens = remove_special_tokens_v2(tokens)
    return tokens


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops, tokenizer, prompt):
        super().__init__()
        self.stops = stops
        self.tokenizer = tokenizer
        self.prompt = add_special_tokens_v2(prompt)
        self.prompt = tokenizer.decode(tokenizer.encode(self.prompt))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            generated_temp_ids = input_ids.tolist()[0]
            if stop in self.tokenizer.decode(generated_temp_ids)[len(self.prompt) :]:
                return True

        return False


class XGLMConversation:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
        debug_status: int = 0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.debug_status = debug_status
        self.max_history = 3

        self.history = []

    def chat(
        self,
        user_message: str,
    ) -> str:
        self.history.append(
            {
                "source": "user",
                "message": user_message,
            },
        )
        total_prompt = ""
        self.history = self.history[-2 * self.max_history :]

        if self.debug_status:
            print(self.history)

        for item in self.history:
            message = item["message"]
            if item["source"] == "user":
                total_prompt += f"\nHuman:\n{message}"
            else:
                total_prompt += f"\nAssistant:\n{message}"

        total_prompt += "\nAssistant:\n"
        if self.debug_status:
            print(total_prompt)
            print("=" * 100)

        answer = self.generate_response(total_prompt)
        answer = self.extract_answer(
            answer,
            prev_prompt=total_prompt,
        )
        self.history.append(
            {
                "source": "bot",
                "message": answer,
            },
        )
        return answer

    def generate_response(self, prompt):
        stop_words = [
            "<|endoftext|>",
            "Human:",
        ]
        stopping_criteria = StoppingCriteriaList(
            [
                StoppingCriteriaSub(
                    stops=stop_words,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                )
            ]
        )
        gen_config = GenerationConfig(
            max_new_tokens=2048,
            repetition_penalty=1.1,
            eos_token_id=[400],
        )

        with torch.no_grad():
            input_text = encode_v2(
                prompt,
                tokenizer=self.tokenizer,
            )
            input_text = torch.tensor([input_text]).to("cuda")

            output_tokens = self.model.generate(
                input_text,
                generation_config=gen_config,
                stopping_criteria=stopping_criteria,
            )
            finetuned_result = decode_v2(output_tokens[0], tokenizer=self.tokenizer)
            torch.cuda.empty_cache()
            gc.collect()
            return finetuned_result

    def start_chat(self):
        while True:
            message = input("You: ")

            if self.debug_status == 1:
                print(message)
                print("-" * 100)

            if message == "exit":
                break
            answer = self.chat(message)

            if self.debug_status:
                print("CONTEXT:", self.history)

            if self.last_response == answer:
                self.history = []
            else:
                self.last_response = answer

            print("Bot:", answer)

    def extract_answer(self, g_answer: str, prev_prompt: str = None):
        answer = g_answer[len(prev_prompt) :].strip()
        answer = answer.replace("Human:", " ")
        answer = answer.replace("Assistant:", " ")
        return answer
