from langchain.messages import SystemMessage
from langgraph.graph.state import CompiledStateGraph

from lg_demo.core.model_provider import HfCloudProvider
from lg_demo.core.nodes import InferenceNode, ToolNode
from lg_demo.core.router import EntryRouter, ToolCallRouter
from lg_demo.core.runtime import RuntimeBuilder
from lg_demo.core.states import RuntimeState
from lg_demo.core.tools import web_search


class TradingInferenceNode(InferenceNode):
    def __call__(self, state: RuntimeState) -> RuntimeState:
        return RuntimeState(
            messages=[self.model.invoke([SystemMessage(content="""
You are an experienced financial analyst. Your task is to analyze available information about a stock and estimate its near-term price direction.

Guidelines:
- Use recent market news, macroeconomic conditions, analyst sentiment, and historical price trends if available.
- Do not fabricate facts. If information is unavailable or uncertain, explicitly say so in reason and lower confidence.
""")] + state.messages)],
            llm_calls=1,
            tool_calls=0,
        )


def build_trading_agent() -> CompiledStateGraph:
    model = HfCloudProvider().get_model()
    tools = ToolNode(
        name="trading_tools", tools=[web_search]
    )  # Replace with actual tools as needed
    model_with_tools = model.bind_tools(tools.tools)

    trading_node = TradingInferenceNode(name="trading_node", model=model_with_tools)

    entry = EntryRouter(entry_node=trading_node)

    conditional_tool_call_router = ToolCallRouter(
        from_nodes=[trading_node],
        to_nodes=[tools],
    )

    agent = RuntimeBuilder(
        nodes=[tools, trading_node],
        routers=[entry, conditional_tool_call_router],
    ).build()

    return agent
