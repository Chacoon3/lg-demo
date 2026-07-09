from typing import Annotated

from fastapi import Depends, Request
from langchain.chat_models import BaseChatModel


def get_model(request: Request) -> BaseChatModel:
    return request.app.state.model_with_tools


ModelDep = Annotated[
    BaseChatModel,
    Depends(get_model),
]
