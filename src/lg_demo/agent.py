from typing import Literal

from langgraph.graph import END, START, StateGraph

import lg_demo.core.inference_provider as inference_provider
from lg_demo.core.nodes import ArithmeticInferenceNode, ToolNode
from lg_demo.core.states import MessagesState
from lg_demo.core.tools import add, divide, multiply

model = inference_provider.HfCloudProvider().get_model()
# model = inference_provider.ChatOllamaProvider(
#     model_name=os.environ["MODEL"], temperature=0.5, num_gpu=1
# ).get_model()


tool_node = ToolNode(name="tool_node", tools=[add, multiply, divide])

model_with_tools = model.bind_tools(tool_node.list_tools())

arith_node = ArithmeticInferenceNode(name="arith_node", model=model_with_tools)


def should_continue(
    state: MessagesState,
) -> Literal["tool_node", END]:  # pyright: ignore[reportInvalidTypeForm]
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state.messages
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END


# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node(arith_node.name, arith_node)
agent_builder.add_node(tool_node.name, tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, arith_node.name)
agent_builder.add_conditional_edges(arith_node.name, should_continue, ["tool_node", END])
agent_builder.add_edge(tool_node.name, arith_node.name)

# Compile the agent
Agent = agent_builder.compile()
