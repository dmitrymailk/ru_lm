import logging
import random
from typing import TypedDict


from telegram import (
    Update,
    ReplyKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
    CallbackQueryHandler,
)

from transformers import AutoTokenizer, AutoModelForCausalLM

import pandas as pd
import torch
import gc


class DialogBotV3:
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

    def _get_sample(
        self,
        user_message: str,
    ):
        user_message = f"Human: {user_message} Assistant:"
        sample = self.tokenizer(
            user_message,
            max_length=512,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)

        return sample

    def chat(
        self,
        user_message: str,
    ) -> str:
        sample = self._get_sample(
            user_message=user_message,
        )
        answer = self.generate_response(sample)
        answer = self.tokenizer.batch_decode(
            answer,
            skip_special_tokens=True,
        )
        answer = self.extract_answer(answer[0])
        return answer

    def generate_response(self, sample):
        with torch.no_grad():
            result = self.model.generate(
                **sample,
                max_new_tokens=512,
                penalty_alpha=0.25,
                top_k=4,
                repetition_penalty=1.1,
                # do_sample=True,
            )
            torch.cuda.empty_cache()
            gc.collect()
            return result

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

    def extract_answer(self, g_answer: str):
        search_str = "Assistant"
        search_index = g_answer.index(search_str) + len(search_str) + 1
        answer = g_answer[search_index:]
        answer = answer.replace("<|endoftext|>", "")
        if "Human:" in answer:
            search_str = "Human:"
            print(answer)
            search_index = answer.index(search_str) + len(search_str) + 1
            answer = answer = answer[:search_index]
        return answer


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

DIALOG = range(1)

# path = "/home/kosenko/deepspeed/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/models/xglm-4.5B_ru_v5/"
path = "dim/xglm_ru_v5"
model = AutoModelForCausalLM.from_pretrained(
    path,
    # load_in_8bit=True,
    # device_map="auto",
    torch_dtype=torch.float16,
)
device = "cuda:0"
model.to(device)
model.eval()
model.half()
tokenizer = AutoTokenizer.from_pretrained(
    path,
)

bot = DialogBotV3(
    model=model,
    tokenizer=tokenizer,
    debug_status=1,
    device=device,
)

SUPER_SIMPLE_DATABASE = {}

reply_markup = ReplyKeyboardMarkup(
    [["/start", "/stop"]],
    resize_keyboard=True,
)


class QueueWaiter(TypedDict):
    """A queue waiter."""

    update_object: Update
    user_username: str


class MessageQueue:
    def __init__(
        self,
    ):
        self.queue = []

    def add(self, item: QueueWaiter):
        self.queue.append(item)

    async def availability_check(self):
        """
        бесконечно отвечаем пока в очереди есть запросы.
        это сделано чтобы на видюхе память не кончилась.
        """
        if len(self.queue) > 0:
            waiter = self.queue.pop(0)
            message = waiter["update_object"].message.text

            bot_response = bot.chat(user_message=message)

            update_object = waiter["update_object"]

            logger.info("Bot response %s : %s", message, bot_response)

            await update_object.message.reply_text(
                bot_response,
                reply_markup=reply_markup,
            )

            await self.availability_check()


message_queue = MessageQueue()


async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> int:
    await update.message.reply_text(
        "Бот любит когда ты начинаешь писать первым :) Чтобы начать диалог напиши ему привет или что-то типа того."
        "Чтобы закончить диалог напиши /stop",
    )

    return DIALOG


async def dialog(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> int:
    user = update.message.from_user
    user_username = user.username
    user_text = update.message.text

    if "/stop" in user_text:
        return await stop(update, context)

    logger.info("Message from %s %s: %s", user_username, user.first_name, user_text)

    message_queue.add(
        QueueWaiter(
            update_object=update,
            user_username=user_username,
        )
    )

    await message_queue.availability_check()

    return DIALOG


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""

    if update.message is not None:
        await update.message.reply_text(
            "До новых встреч!",
        )
    else:
        await update.effective_message.reply_text(
            "До новых встреч!",
        )

    return ConversationHandler.END


async def button(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Parses the CallbackQuery and updates the message text."""
    query = update.callback_query
    update.effective_user.username

    # CallbackQueries need to be answered, even if no notification to the user is needed
    # Some clients may have trouble otherwise. See https://core.telegram.org/bots/api#callbackquery
    await query.answer()
    if query.data == "stop":
        await stop(update, context)


def main() -> None:
    """Run the bot."""
    # Create the Application and pass it your bot's token.
    TOKEN = open("./token").read()
    application = Application.builder().token(TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            DIALOG: [MessageHandler(filters.TEXT, dialog)],
        },
        fallbacks=[CommandHandler("stop", stop)],
    )

    application.add_handler(conv_handler)
    application.add_handler(CallbackQueryHandler(button))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    main()
