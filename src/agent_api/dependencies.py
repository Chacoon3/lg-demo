from typing import Annotated

from fastapi import Depends, Request
from langchain.chat_models import BaseChatModel


def get_agent(request: Request):
    return request.app.state.agent


def get_finance_agent(request: Request):
    return request.app.state.finance_agent


GeneralAgentDep = Annotated[
    BaseChatModel,
    Depends(get_agent),
]

FinanceAgentDep = Annotated[
    BaseChatModel,
    Depends(get_finance_agent),
]
