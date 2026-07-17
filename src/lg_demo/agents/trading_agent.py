from typing import Any

from langchain.chat_models import BaseChatModel
from langchain.messages import SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from lg_demo.core.nodes import InferenceNode, bind_tools_to_model, execute_tool_calls
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


def build_trading_agent(
    model: BaseChatModel,
    checkpointer: Any | None = None,
) -> CompiledStateGraph[RuntimeState]:
    tools = [web_search]
    model_with_tools = bind_tools_to_model(model, tools)

    trading_node = TradingInferenceNode(name="trading_node", model=model_with_tools)

    summary_node = FinanceSearchSummaryNode(name="summary_node", model=model_with_tools)

    def trading_tools(state: RuntimeState) -> RuntimeState:
        return execute_tool_calls(state, tools)

    graph = StateGraph(RuntimeState)
    graph.add_node(trading_node.name, trading_node)
    graph.add_node("trading_tools", trading_tools)
    graph.add_node(summary_node.name, summary_node)
    graph.add_edge(START, trading_node.name)
    graph.add_edge(trading_node.name, "trading_tools")
    graph.add_edge("trading_tools", summary_node.name)

    return graph.compile(checkpointer=checkpointer)
