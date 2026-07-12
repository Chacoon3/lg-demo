from abc import ABC, abstractmethod

from lg_demo.core.nodes import BaseNode
from lg_demo.core.states import RuntimeState


class BaseRouter(ABC):
    @abstractmethod
    def __call__(self, state: RuntimeState) -> str: ...


class EntryRouter(BaseRouter):

    def __init__(self, entry_node: BaseNode):
        self.entry_node = entry_node

    def __call__(self, state: RuntimeState) -> str:
        raise NotImplementedError("router should not be called directly")


class DirectRouter(BaseRouter):

    def __init__(self, from_nodes: list[BaseNode], to_nodes: list[BaseNode]):
        self.from_nodes = from_nodes
        self.to_nodes = to_nodes
        self.exit_set: set[str] = set(n.name for n in to_nodes)

    def __call__(self, state: RuntimeState) -> str:
        raise NotImplementedError("router should not be called directly")


class ConditionalRouter(BaseRouter):
    END = "__end__"

    def __init__(self, from_nodes: list[BaseNode], to_nodes: list[BaseNode]):
        self.from_nodes = from_nodes
        self.to_nodes = to_nodes
        self.exit_set: set[str] = set(n.name for n in to_nodes)


class PromptClassRouter(ConditionalRouter):

    def __call__(self, state: RuntimeState) -> str:
        classification = state.messages[-1].content
        if classification in self.exit_set:
            return classification
        return self.END


class ToolCallRouter(ConditionalRouter):

    def __call__(self, state: RuntimeState) -> str:
        last_message = state.messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_name = last_message.tool_calls[-1]["name"]
            if tool_name in self.exit_set:
                return tool_name
        return self.END
