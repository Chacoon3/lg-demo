import os
from enum import Enum
from importlib import util
from pathlib import Path

import pytest
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

os.environ.setdefault("TAVILY_API_KEY", "test-key")


def _load_arithmetic_inference_node():
    module_path = (
        Path(__file__).resolve().parents[4] / "src" / "lg_demo" / "agents" / "arithmetic_agent.py"
    )
    spec = util.spec_from_file_location("test_arithmetic_agent", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load arithmetic_agent module for tests")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ArithmeticInferenceNode


ArithmeticInferenceNode = _load_arithmetic_inference_node()
from lg_demo.core.nodes import PromptClassifierNode, ToolNode
from lg_demo.core.states import RuntimeState


class FakeTool:
    def __init__(self, name, result):
        self.name = name
        self._result = result
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        return self._result


class FakeModel:
    def __init__(self, response=None):
        self.response = response
        self.last_messages = None
        self.bound_tools = None

    def invoke(self, messages):
        self.last_messages = messages
        return self.response

    def bind_tools(self, tools):
        self.bound_tools = tools
        return "bound-model"


class FakeStructuredModel:
    def __init__(self, response):
        self.response = response
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return self.response


def test_tool_node_registry_and_tool_listing():
    tool_a = FakeTool(name="a", result="1")
    tool_b = FakeTool(name="b", result="2")
    node = ToolNode(name="tool_node", tools=[tool_a, tool_b])

    assert node.get_tool("a") is tool_a
    assert node.get_tool("missing") is None
    assert list(node.list_tools()) == [tool_a, tool_b]


def test_tool_node_executes_tool_calls_and_returns_runtime_state():
    add_tool = FakeTool(name="add", result="3")
    mult_tool = FakeTool(name="multiply", result="6")
    node = ToolNode(name="tool_node", tools=[add_tool, mult_tool])

    state = RuntimeState(
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "add", "args": {"a": 1, "b": 2}, "id": "call-1"},
                    {"name": "multiply", "args": {"a": 2, "b": 3}, "id": "call-2"},
                ],
            )
        ],
        llm_calls=0,
        tool_calls=0,
        state={},
    )

    with pytest.raises(ValidationError, match="state"):
        node(state)

    # Tool invocations still happen before RuntimeState validation fails.
    assert add_tool.calls == [{"a": 1, "b": 2}]
    assert mult_tool.calls == [{"a": 2, "b": 3}]


def test_tool_node_bind_with_model():
    tool = FakeTool(name="add", result="3")
    node = ToolNode(name="tool_node", tools=[tool])
    model = FakeModel()

    bound = node.bind_with_model(model)

    assert bound == "bound-model"
    assert model.bound_tools == [tool]


def test_arithmetic_inference_node_invokes_model_with_system_prompt():
    model_response = AIMessage(content="42")
    model = FakeModel(response=model_response)
    node = ArithmeticInferenceNode(name="arithmetic", model=model)
    state = RuntimeState(
        messages=[HumanMessage(content="6 * 7")],
        llm_calls=0,
        tool_calls=0,
        state={},
    )

    with pytest.raises(ValidationError, match="state"):
        node(state)

    assert isinstance(model.last_messages[0], SystemMessage)
    assert "arithmetic" in model.last_messages[0].content.lower()
    assert model.last_messages[1].content == "6 * 7"


def test_prompt_classifier_node_wraps_structured_response_value():
    class PromptClass(Enum):
        MATH = "math"
        CHAT = "chat"

    structured = FakeStructuredModel(response={"value": "math"})

    class BaseModelForClassifier:
        def with_structured_output(self, _):
            return structured

    node = PromptClassifierNode(
        name="classifier",
        model=BaseModelForClassifier(),
        prompt_class=PromptClass,
    )
    state = RuntimeState(
        messages=[HumanMessage(content="what is 1+1?")],
        llm_calls=0,
        tool_calls=0,
        state={},
    )

    with pytest.raises(ValidationError, match="state"):
        node(state)

    assert isinstance(structured.last_messages[0], SystemMessage)
    assert "math, chat" in structured.last_messages[0].content
