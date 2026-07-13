from __future__ import annotations

import os
from abc import ABC, abstractmethod

from langchain.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


class InferenceProvider(ABC):

    __MODEL_REGISTRY: dict[str, InferenceProvider] = {}

    @abstractmethod
    def _get_model_key(self) -> str: ...

    @abstractmethod
    def _get_model(self) -> BaseChatModel: ...

    def get_model(self) -> BaseChatModel:
        key = self._get_model_key()
        if key in self.__MODEL_REGISTRY:
            return self.__MODEL_REGISTRY[key]._get_model()
        model = self._get_model()
        self.__MODEL_REGISTRY[key] = self
        return model


class ChatOllamaProvider(InferenceProvider):

    def _get_model_key(self) -> str:
        return f"{self.model_name}-{self.temperature}-{self.num_gpu}"

    def __init__(self, model_name: str, temperature: float = 0, num_gpu: int = 1):
        self.model_name = model_name
        self.temperature = temperature
        self.num_gpu = num_gpu

    def _get_model(self) -> BaseChatModel:
        return ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            num_gpu=self.num_gpu,
        )


class HfCloudProvider(InferenceProvider):

    def _get_model_key(self) -> str:
        return f"{self.model_name}"

    def __init__(self, model_name: str = "openai/gpt-oss-120b:groq"):
        self.model_name = model_name

    def _get_model(self) -> BaseChatModel:
        return ChatOpenAI(
            model=self.model_name,
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ["HF_TOKEN"],
        )
