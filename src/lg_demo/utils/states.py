import operator
from typing import Annotated

from langchain.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel


class MessagesState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: Annotated[int, operator.add]
    tool_calls: Annotated[int, operator.add]
