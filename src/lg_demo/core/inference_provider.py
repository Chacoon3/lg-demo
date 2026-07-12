import os
from typing import Protocol

from langchain.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


class InferenceProvider(Protocol):

    def get_model(self) -> BaseChatModel: ...

    def __str__(self) -> str:
        return self.__class__.__name__


class ChatOllamaProvider:

    def __init__(self, model_name: str, temperature: float = 0, num_gpu: int = 1):
        model = ChatOllama(
            model=model_name,
            temperature=temperature,
            num_gpu=num_gpu,
        )
        self.model = model

    def get_model(self) -> BaseChatModel:
        return self.model


class HfCloudProvider:

    def __init__(self, model_name: str = "openai/gpt-oss-120b:groq"):
        model = ChatOpenAI(
            model=model_name,
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ["HF_TOKEN"],
        )
        self.model = model

    def get_model(self) -> BaseChatModel:
        return self.model
