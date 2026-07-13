from langchain.messages import SystemMessage
from langgraph.graph.state import CompiledStateGraph

from lg_demo.core.model_provider import HfCloudProvider
from lg_demo.core.nodes import InferenceNode, ToolNode
from lg_demo.core.router import DirectRouter, EntryRouter, ToolCallRouter
from lg_demo.core.runtime import RuntimeBuilder
from lg_demo.core.states import RuntimeState
from lg_demo.core.tools import web_search


class TradingInferenceNode(InferenceNode):
    def __call__(self, state: RuntimeState) -> RuntimeState:
        return RuntimeState(
            messages=[self.model.invoke([SystemMessage(content="""
You are an experienced financial analyst to fetch relevant financial information from the web and summarize it for further analysis.
""")] + state.messages)],
            llm_calls=1,
            tool_calls=0,
        )


class FinanceSearchSummaryNode(InferenceNode):
    def __call__(self, state: RuntimeState) -> RuntimeState:
        return RuntimeState(
            messages=[self.model.invoke([SystemMessage(content="""
Your task is to summarize the structured JSON output and convert it into human-readable sentences which provide investment and trading insights.
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

    summary_node = FinanceSearchSummaryNode(name="summary_node", model=model_with_tools)

    entry = EntryRouter(entry_node=trading_node)

    conditional_tool_call_router = ToolCallRouter(
        from_nodes=[trading_node],
        to_nodes=[tools],
    )

    tool_to_summary_router = DirectRouter(
        from_nodes=[tools],
        to_nodes=[summary_node],
    )

    agent = RuntimeBuilder(
        nodes=[trading_node, tools, summary_node],
        routers=[entry, conditional_tool_call_router, tool_to_summary_router],
    ).build()

    return agent
