from typing import Any

from langchain.chat_models import BaseChatModel
from langchain.messages import SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from lg_demo.core.nodes import InferenceNode, bind_tools_to_model, execute_tool_calls
from lg_demo.core.states import RuntimeState
from lg_demo.core.tools import add, divide, multiply, power


class ArithmeticInferenceNode(InferenceNode):

    def __call__(self, state: RuntimeState) -> RuntimeState:
        # Implement the arithmetic inference logic here
        return RuntimeState(
            messages=[self.model.invoke([SystemMessage(content="""
    You are an assistant to answer arithmetic questions.
                                """)] + state.messages)],
            llm_calls=1,
            tool_calls=0,
        )


def build_simple_arithmetic_agent(
    model: BaseChatModel,
    checkpointer: Any | None = None,
) -> CompiledStateGraph[RuntimeState]:
    tools = [add, multiply, divide, power]
    model = bind_tools_to_model(model, tools)

    arith_node = ArithmeticInferenceNode(name="arith_node", model=model)

    def tool_node(state: RuntimeState) -> RuntimeState:
        return execute_tool_calls(state, tools)

    graph = StateGraph(RuntimeState)
    graph.add_node(arith_node.name, arith_node)
    graph.add_node("tool_node", tool_node)
    graph.add_edge(START, arith_node.name)
    graph.add_edge(arith_node.name, "tool_node")

    return graph.compile(checkpointer=checkpointer)
