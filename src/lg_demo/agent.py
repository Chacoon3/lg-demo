from enum import Enum, unique

import lg_demo.core.inference_provider as inference_provider
from lg_demo.core.nodes import (
    ArithmeticInferenceNode,
    PromptClassifierNode,
    ToolNode,
)
from lg_demo.core.router import DirectRouter, EntryRouter, ToolCallRouter
from lg_demo.core.runtime import RuntimeBuilder
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

entry_router = EntryRouter(entry_node=classifier_node)

direct_router = DirectRouter(
    from_nodes=[classifier_node],
    to_nodes=[arith_node],
)

conditional_tool_call_router = ToolCallRouter(
    from_nodes=[arith_node],
    to_nodes=[tool_set],
)

Agent = RuntimeBuilder(
    nodes=[classifier_node, tool_set, arith_node],
    routers=[entry_router, direct_router, conditional_tool_call_router],
).build()
