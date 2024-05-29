from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Union, Iterator, List, Dict
from argparse import Namespace

from generator.api_calls.openai_text import chat_generate_text, cohere_generate_text
from generator.api_calls.openai_image import chat_generate_dalle_prompt, cohere_generate_dalle_prompt
from generator.api_calls.kandinsky import Text2ImageAPI as KandinskyApi
from generator.api_calls.elevenlabs_audio import ElevenlabsAudio
from generator.entities import WordWithContext


@dataclass(frozen=True)
class GeneratorResponse:
    text: str = ""
    image_path: Union[str, Path] = ""
    image_bytes: bytes = b""
    audio_bytes: Iterator[bytes] = field(default_factory=list)

    def save_image(self, path: Union[str, Path]) -> None:
        if isinstance(self.image_bytes, bytes) and len(self.image_bytes) > 0:
            with open(path, "wb") as f:
                f.write(self.image_bytes)
        else:
            raise ValueError(f"Unsupported image type {type(self.image_bytes)} or len {len(self.image_bytes)} was 0")

    def save_audio(self, path: Union[str, Path]) -> None:
        if isinstance(self.audio_bytes, Iterator):
            with open(path, 'wb') as f:
                for chunk in self.audio_bytes:
                    f.write(chunk)
        else:
            raise ValueError(f"Unsupported audio type {type(self)}")


class AvailableModels(Enum):
    # Text
    GPT_4o: str = "gpt-4o"
    GPT_3_5_turbo: str = "gpt-3.5-turbo"
    COHERE: str = "cohere"

    # Images
    # TODO
    KANDINSKY_3: str = "kandinsky"

    # Audio
    ELEVENLABS: str = "elevenlabs"

    @staticmethod
    def available_models() -> set:
        all_models = AvailableModels.available_text_models()
        all_models.update(AvailableModels.available_image_models())
        all_models.update(AvailableModels.available_tts_models())
        return all_models

    @staticmethod
    def available_text_models() -> set:
        return {AvailableModels.GPT_4o, AvailableModels.GPT_3_5_turbo, AvailableModels.COHERE}

    @staticmethod
    def available_image_models() -> set:
        return {AvailableModels.KANDINSKY_3}

    @staticmethod
    def available_tts_models() -> set:
        return {AvailableModels.ELEVENLABS}

    @staticmethod
    def backward_mapping() -> dict:
        return {
            AvailableModels.GPT_4o.value: AvailableModels.GPT_4o,
            AvailableModels.GPT_3_5_turbo.value: AvailableModels.GPT_3_5_turbo,
            AvailableModels.COHERE.value: AvailableModels.COHERE,
            AvailableModels.ELEVENLABS.value: AvailableModels.ELEVENLABS,
            AvailableModels.KANDINSKY_3.value: AvailableModels.KANDINSKY_3
        }


@dataclass(frozen=True)
class GeneratorConfig:
    model: AvailableModels
    API_KEY: str = ""
    SECRET_KEY: str = ""
    language: str = "english"

    @staticmethod
    def from_args(args: Namespace) -> list:
        str_to_model: Dict[str, AvailableModels] = AvailableModels.backward_mapping()
        text_model_config = GeneratorConfig(
            str_to_model[args.text_model],
            API_KEY=args.text_api_key,
            SECRET_KEY=args.text_secret_key,
            language=args.language
        )
        image_model_config = GeneratorConfig(
            str_to_model[args.image_model],
            API_KEY=args.image_api_key,
            SECRET_KEY=args.image_secret_key,
            language=args.language
        )
        tts_model_config = GeneratorConfig(
            str_to_model[args.voice_model],
            API_KEY=args.voice_api_key,
            SECRET_KEY=args.voice_secret_key,
            language=args.language
        )
        return [text_model_config, image_model_config, tts_model_config]


class Generator(ABC):
    def __init__(self, config: GeneratorConfig):
        self.config: GeneratorConfig = config

    @abstractmethod
    def generate(self, input_: Union[WordWithContext, str]) -> GeneratorResponse:
        raise NotImplementedError


class TextGenerator(Generator, ABC):
    @abstractmethod
    def generate_image_prompt(
            self,
            word_with_context: WordWithContext,
            card_text: str
    ) -> GeneratorResponse:
        raise NotImplementedError


class ImageGenerator(Generator, ABC):
    pass


class TTSGenerator(Generator, ABC):
    pass


class GPTextGenerator(TextGenerator):

    def generate_image_prompt(self, word_with_context: WordWithContext, card_text: str) -> GeneratorResponse:
        image_prompt = chat_generate_dalle_prompt(
            word_with_context,
            card_text,
            self.config.model.value
        )
        return GeneratorResponse(text=image_prompt)

    def generate(self, input_: Union[WordWithContext, str]) -> GeneratorResponse:
        if isinstance(input_, WordWithContext):
            text = chat_generate_text(input_, self.config.model.value)
            return GeneratorResponse(text=text)
        else:
            raise ValueError(f"Input {input_} was of wrong type {type(input_)}")


class CohereTextGenerator(TextGenerator):
    def generate(self, input_: Union[WordWithContext, str]) -> GeneratorResponse:
        if isinstance(input_, WordWithContext):
            text = cohere_generate_text(input_, self.config.API_KEY)
            return GeneratorResponse(text=text)
        else:
            raise ValueError(f"Input {input_} was of wrong type {type(input_)}")

    def generate_image_prompt(self, word_with_context: WordWithContext, card_text: str) -> GeneratorResponse:
        image_prompt = cohere_generate_dalle_prompt(
            word_with_context,
            card_text,
            self.config.API_KEY
        )
        return GeneratorResponse(text=image_prompt)


class KandinskyImageGenerator(ImageGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.config.API_KEY), f"API key for Kandinsky was empty"
        assert len(self.config.SECRET_KEY), f"SECRET_KEY for Kandinsky was empty"

    def generate(self, input_: Union[WordWithContext, str]) -> GeneratorResponse:
        if isinstance(input_, str):
            image_bytes = KandinskyApi.generate_image(input_, self.config.API_KEY, self.config.SECRET_KEY)
            return GeneratorResponse(image_bytes=image_bytes)
        else:
            raise ValueError(f"Input {input_} was of wrong type")


class ElevenlabsTTSGenerator(TTSGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.config.API_KEY), f"API key for Kandinsky was empty"

    def generate(self, input_: Union[WordWithContext, str]) -> GeneratorResponse:
        if isinstance(input_, str):
            audio_bytes: Iterator[bytes] = ElevenlabsAudio.generate_audio(
                text=input_,
                xi_api_key=self.config.API_KEY,
                language=self.config.language
            )
            return GeneratorResponse(audio_bytes=audio_bytes)
        else:
            raise ValueError(f"input {input_} was of wrong type for audio generation")


class GeneratorFactory:
    MAPPING = {
        AvailableModels.GPT_3_5_turbo: GPTextGenerator,
        AvailableModels.GPT_4o: GPTextGenerator,
        AvailableModels.KANDINSKY_3: KandinskyImageGenerator,
        AvailableModels.ELEVENLABS: ElevenlabsTTSGenerator,
        AvailableModels.COHERE: CohereTextGenerator,
    }

    @staticmethod
    def create(config: GeneratorConfig) -> Generator:
        if config.model in AvailableModels.available_models():
            return GeneratorFactory.MAPPING[config.model](config)
        else:
            raise ValueError(f"{config.model} is not in the list of available models")
