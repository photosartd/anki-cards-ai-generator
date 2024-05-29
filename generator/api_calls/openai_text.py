import logging
from openai import OpenAI
import cohere

from generator.config import Config
from generator.entities import WordWithContext
from generator.api_calls.text_prompt_by_language import prompt_by_language


def chat_generate_text(word_with_context: WordWithContext, model: str) -> str:
    logging.info(
        f"ChatGPT card text: processing word [{word_with_context.word}] with context [{word_with_context.context}] in language [{Config.LANGUAGE}]")

    system_prompt = prompt_by_language.get_system_prompt_by_language()

    messages = [
        {"role": "system", "content": f"{system_prompt}"},
        {"role": "user", "content": f"WORD: [{word_with_context.word}]; CONTEXT: [{word_with_context.context}]"},
    ]

    client = OpenAI(
        api_key=Config.OPENAI_API_KEY
    )

    logging.debug(f"ChatGPT card generation messages {messages}")

    response = client.chat.completions.create(
        # input prompt
        messages=messages,
        # model parameters
        model=model,
        temperature=0.2,  # keep low for conservative answers
        max_tokens=512,
        n=1,
        presence_penalty=0,
        frequency_penalty=0.1,
    )

    generated_text = response.choices[0].message.content
    logging.debug(f"ChatGPT generated card text for word {word_with_context.word}")
    logging.debug(f"ChatGPT card text: {generated_text}")
    return generated_text


def cohere_generate_text(word_with_context: WordWithContext, api_key: str) -> str:
    logging.info(
        f"Cohere card text: processing word [{word_with_context.word}] with context [{word_with_context.context}] in language [{Config.LANGUAGE}]")

    system_prompt = prompt_by_language.get_system_prompt_by_language()
    co = cohere.Client(api_key)
    message = f"""
    {system_prompt}
    WORD: [{word_with_context.word}]; CONTEXT: [{word_with_context.context}]
    """
    response = co.chat(
        message=message,
        temperature=0.2,
        max_tokens=512,
    )
    text = response.text
    text = text.replace(word_with_context.word, "")
    return text
