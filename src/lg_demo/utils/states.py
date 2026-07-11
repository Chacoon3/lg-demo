import operator
from typing import Annotated

from langchain.messages import AnyMessage
from pydantic import BaseModel


class MessagesState(BaseModel):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: Annotated[int, operator.add]
    tool_calls: Annotated[int, operator.add]
