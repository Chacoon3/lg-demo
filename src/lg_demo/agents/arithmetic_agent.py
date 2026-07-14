from langchain.chat_models import BaseChatModel
from langchain.messages import SystemMessage
from langgraph.graph.state import CompiledStateGraph

from lg_demo.core.nodes import InferenceNode, ToolNode
from lg_demo.core.router import EntryRouter, ToolCallRouter
from lg_demo.core.runtime import RuntimeBuilder
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


def build_simple_arithmetic_agent(model: BaseChatModel) -> CompiledStateGraph:
    math_tool = ToolNode(name="tool_node", tools=[add, multiply, divide, power])
    model = model.bind_tools(math_tool.tools)

    arith_node = ArithmeticInferenceNode(name="arith_node", model=model)

    entry_router = EntryRouter(entry_node=arith_node)

    conditional_tool_call_router = ToolCallRouter(
        from_nodes=[arith_node],
        to_nodes=[math_tool],
    )

    agent = RuntimeBuilder(
        nodes=[math_tool, arith_node],
        routers=[entry_router, conditional_tool_call_router],
    ).build()

    return agent
