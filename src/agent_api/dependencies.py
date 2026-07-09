from typing import Annotated

from fastapi import Depends, Request
from langchain.chat_models import BaseChatModel


def get_agent(request: Request):
    return request.app.state.agent


AgentDep = Annotated[
    BaseChatModel,
    Depends(get_agent),
]
