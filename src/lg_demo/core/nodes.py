from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence

from build.lib.lg_demo.state import MessagesState
from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage, SystemMessage, ToolMessage
from langchain.tools import BaseTool


class BaseNode(ABC):

    def __init__(
        self,
        name: str,
    ):
        self.name = name


class InferenceNode(BaseNode):

    def __init__(self, name: str, model: BaseChatModel):
        super().__init__(name)
        self.model: BaseChatModel = model

    @abstractmethod
    def __call__(self, state: MessagesState) -> MessagesState: ...


class ToolNode(BaseNode):

    def __init__(self, name: str, tools: Sequence[BaseTool]):
        super().__init__(name)
        self.tools = list(tools)
        self.registry: dict = {tool.name: tool for tool in tools}

    def get_tool(self, name: str) -> BaseTool:
        return self.registry.get(name)

    def list_tools(self) -> Sequence[BaseTool]:
        return self.tools

    def __call__(self, state: MessagesState) -> MessagesState:
        result = []
        for tool_call in state.messages[-1].tool_calls:
            tool = self.get_tool(tool_call["name"])
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return MessagesState(
            messages=result,
            llm_calls=0,
            tool_calls=len(result),
        )

    def bind_with_model(self, model: BaseChatModel) -> BaseChatModel:
        return model.bind_tools(self.tools)


class ArithmeticInferenceNode(InferenceNode):

    def __call__(self, state: MessagesState) -> MessagesState:
        # Implement the arithmetic inference logic here
        return MessagesState(
            messages=[self.model.invoke([SystemMessage(content="""
You are an assistant tasked with performing arithmetic on a set of inputs.
Use tools when necessary.
Return the arithmetic result itself without any additional text.
                            """)] + state.messages)],
            llm_calls=1,
            tool_calls=0,
        )


class PromptClassifierNode(InferenceNode):

    def __init__(self, name, model: BaseChatModel, prompt_class: type[Enum]):
        super().__init__(name, model)
        self.prompt_class = prompt_class
        self.model = model.with_structured_output(prompt_class)
        self._class_members = list(prompt_class)

    def __call__(self, state: MessagesState) -> MessagesState:

        resp = self.model.invoke([SystemMessage(content=f"""
You are an assistant tasked with classifying prompts into one of the categories:
{', '.join([member.value for member in self._class_members])}
""")] + state.messages)

        wrapped_msg = AIMessage(content=resp["value"])
        return MessagesState(
            messages=[wrapped_msg],
            llm_calls=1,
            tool_calls=0,
        )
