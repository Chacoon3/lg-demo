from typing import Protocol

from pydantic import BaseModel


class Prompt(BaseModel):
    message: str
    context: str


class LLMResponse(BaseModel):
    message: str


class LLM(Protocol):
    def __init__(self, model_name: str, temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature

    def invoke(self, message: Prompt) -> LLMResponse:
        """Invoke the LLM with the given prompt and return the response."""
        ...
