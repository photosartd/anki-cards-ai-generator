from typing import Dict, Iterator
from copy import deepcopy

import requests


class ElevenlabsAudio:
    CHUNK_SIZE = 1024
    URL = "https://api.elevenlabs.io/v1/text-to-speech/{}"
    HEADERS = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": "<xi-api-key>"
    }
    STANDARD_VOICE_SETTINGS: Dict[str, float] = {
        "stability": 0.5,
        "similarity_boost": 0.5
    }

    ENGLISH_VOICE_ID: str = "P7x743VjyZEOihNNygQ9"
    GERMAN_VOICE_ID: str = "MHxgWgZ7ayjcFagtPw59"

    VOICE_IDS: Dict[str, str] = {
        "english": ENGLISH_VOICE_ID,
        "german": GERMAN_VOICE_ID
    }

    MODEL1: str = "eleven_monolingual_v1"

    def __init__(
            self,
            voice_id: str,
            xi_api_key: str,
            model_id: str,
            voice_settings: Dict[str, float]
    ):
        self.voice_id = voice_id
        self.xi_api_key = xi_api_key
        self.model_id = model_id
        self.voice_settings = voice_settings

    def post(self, text: str) -> requests.Response:
        response = requests.post(self.url, json=self.data(text), headers=self.headers)
        return response

    @property
    def url(self) -> str:
        return self.URL.format(self.voice_id)

    @property
    def headers(self) -> dict:
        headers = deepcopy(self.HEADERS)
        headers["xi-api-key"] = self.xi_api_key
        return headers

    def data(self, text: str) -> dict:
        return {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": self.voice_settings
        }

    @staticmethod
    def generate_audio(text: str, xi_api_key: str, language: str = "english") -> Iterator[bytes]:
        voice_id = ElevenlabsAudio.VOICE_IDS[language]
        return ElevenlabsAudio(
            voice_id=voice_id,
            xi_api_key=xi_api_key,
            model_id=ElevenlabsAudio.MODEL1,
            voice_settings=ElevenlabsAudio.STANDARD_VOICE_SETTINGS
        ).post(text).iter_content(ElevenlabsAudio.CHUNK_SIZE)
