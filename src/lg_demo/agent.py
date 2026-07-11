import os
from typing import Literal

from langchain.messages import SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from lg_demo.utils.states import MessagesState
from lg_demo.utils.tools import add, divide, multiply

model = ChatOllama(
    model=os.getenv("MODEL"),
    temperature=0,
    num_gpu=1,
)
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


def llm_call(state: MessagesState):
    return MessagesState(
        messages=[
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.",
                        think=True,
                    )
                ]
                + state.messages
            )
        ],
        llm_calls=1,
        tool_calls=state.tool_calls,
    )


def tool_node(state: MessagesState):
    """Performs the tool call"""

    result = []
    for tool_call in state.messages[-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return MessagesState(
        messages=result,
        llm_calls=0,
        tool_calls=len(result),
    )


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
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
Agent = agent_builder.compile()
