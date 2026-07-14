from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence

from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage, SystemMessage, ToolMessage
from langchain.tools import BaseTool

from lg_demo.core.states import AgentPlan, RuntimeState


class BaseNode(ABC):

    def __init__(self, name: str, priority: int):
        self.name = name
        self.priority = priority


class InferenceNode(BaseNode):

    def __init__(self, name: str, model: BaseChatModel):
        super().__init__(name, priority=1)
        self.model: BaseChatModel = model

    @abstractmethod
    def __call__(self, state: RuntimeState) -> RuntimeState: ...


class ToolNode(BaseNode):

    def __init__(self, name: str, tools: Sequence[BaseTool]):
        super().__init__(name, priority=0)
        self.tools = list(tools)
        self.registry: dict = {tool.name: tool for tool in tools}

    def get_tool(self, name: str) -> BaseTool:
        return self.registry.get(name)

    def list_tools(self) -> Sequence[BaseTool]:
        return self.tools

    def __call__(self, state: RuntimeState) -> RuntimeState:
        result = []
        for tool_call in state.messages[-1].tool_calls:
            tool = self.get_tool(tool_call["name"])
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return RuntimeState(
            messages=result,
            llm_calls=0,
            tool_calls=len(result),
        )

    def bind_with_model(self, model: BaseChatModel) -> BaseChatModel:
        return model.bind_tools(self.tools)


class PromptClassifierNode(InferenceNode):

    def __init__(self, name, model: BaseChatModel, prompt_class: type[Enum]):
        super().__init__(name, model)
        self.prompt_class = prompt_class
        self.model = model.with_structured_output(prompt_class)
        self._class_members = list(prompt_class)

    def __call__(self, state: RuntimeState) -> RuntimeState:

        resp = self.model.invoke([SystemMessage(content=f"""
You are an assistant tasked with classifying prompts into one of the categories:
{', '.join([member.value for member in self._class_members])}
""")] + state.messages)

        wrapped_msg = AIMessage(content=resp["value"])
        return RuntimeState(
            messages=[wrapped_msg],
            llm_calls=1,
            tool_calls=0,
        )


class PlannerNode(InferenceNode):

    def __init__(self, name, model: BaseChatModel):
        super().__init__(name, model)
        self.model = model.with_structured_output(AgentPlan)

    def __call__(self, state: RuntimeState) -> RuntimeState:
        return RuntimeState(
            messages=[self.model.invoke([SystemMessage(content="""
You are a generalist to come up with a plan that breaks down complex tasks into a sequence of actionable steps for further processing.
""")] + state.messages)],
            llm_calls=1,
            tool_calls=0,
        )
