from __future__ import annotations

import os
from abc import ABC, abstractmethod

from langchain.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


class ModelProvider(ABC):

    __MODEL_REGISTRY: dict[str, ModelProvider] = {}

    @abstractmethod
    def get_model_key(self) -> str: ...

    @abstractmethod
    def _get_model(self) -> BaseChatModel: ...

    def get_model(self) -> BaseChatModel:
        key = self.get_model_key()
        if key in self.__MODEL_REGISTRY:
            return self.__MODEL_REGISTRY[key]._get_model()  # pylint: disable=W0212
        model = self._get_model()
        self.__MODEL_REGISTRY[key] = self
        return model


class OllamaProvider(ModelProvider):

    def get_model_key(self) -> str:
        return f"{self.model_name}-{self.temperature}-{self.num_gpu}"

    def __init__(self, model_name: str = None, temperature: float = 0, num_gpu: int = 1):
        self.model_name = model_name or os.environ.get("OLLAMA_MODEL")
        if not self.model_name:
            raise ValueError(
                "Model name must be provided either as an argument or through the OLLAMA_MODEL environment variable."
            )
        self.temperature = temperature
        self.num_gpu = num_gpu

    def _get_model(self) -> BaseChatModel:
        return ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            num_gpu=self.num_gpu,
        )


class HfCloudProvider(ModelProvider):

    def get_model_key(self) -> str:
        return f"{self.model_name}"

    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.environ.get("HF_CLOUD_MODEL")
        if not self.model_name:
            raise ValueError(
                "Model name must be provided either as an argument or through the HF_CLOUD_MODEL environment variable."
            )

    def _get_model(self) -> BaseChatModel:
        return ChatOpenAI(
            model=self.model_name,
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ["HF_TOKEN"],
        )


class OpenAICloudProvider(ModelProvider):

    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.environ.get("OPENAI_MODEL")
        if not self.model_name:
            raise ValueError(
                "Model name must be provided either as an argument or through the OPENAI_MODEL environment variable."
            )

    def get_model_key(self) -> str:
        return f"{self.model_name}"

    def _get_model(self) -> BaseChatModel:
        return ChatOpenAI(
            model=self.model_name,
            api_key=os.environ["OPENAI_API_KEY"],
        )
