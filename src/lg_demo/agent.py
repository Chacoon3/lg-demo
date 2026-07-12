from enum import Enum, unique
from typing import Literal

from langgraph.graph import END, START, StateGraph

import lg_demo.core.inference_provider as inference_provider
from lg_demo.core.nodes import (
    ArithmeticInferenceNode,
    PromptClassifierNode,
    ToolNode,
)
from lg_demo.core.states import MessagesState
from lg_demo.core.tools import add, divide, multiply, power


@unique
class PromptType(str, Enum):
    MATH = "math"
    TRAVEL_AND_DINING = "travel_and_dining_plan"
    INFORMATION_RETRIEVAL = "information_retrieval"
    GENERAL_INQUIRY = "general_inquiry"


model = inference_provider.HfCloudProvider().get_model()
# model = inference_provider.ChatOllamaProvider(
#     model_name=os.environ["MODEL"], temperature=0.5, num_gpu=1
# ).get_model()

classifier_node = PromptClassifierNode(
    name="classifier_node",
    model=model,
    prompt_class=PromptType,
)

tool_set = ToolNode(name="tool_node", tools=[add, multiply, divide, power])

model = model.bind_tools(tool_set.tools)

arith_node = ArithmeticInferenceNode(name="arith_node", model=model)


def route_classified_prompt(
    state: MessagesState,
) -> Literal["arith_node", END]:  # pyright: ignore[reportInvalidTypeForm]
    """Route the prompt based upon its classification"""

    messages = state.messages
    last_message = messages[-1]

    classification = last_message.content
    if classification == PromptType.MATH:
        return "arith_node"
    return END


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
agent_builder.add_node(tool_set.name, tool_set)
agent_builder.add_node(classifier_node.name, classifier_node)

# Add edges to connect nodes
agent_builder.add_edge(START, classifier_node.name)
agent_builder.add_conditional_edges(
    classifier_node.name, route_classified_prompt, ["arith_node", END]
)
agent_builder.add_conditional_edges(arith_node.name, should_continue, ["tool_node", END])
agent_builder.add_edge(tool_set.name, arith_node.name)

# Compile the agent
Agent = agent_builder.compile()
