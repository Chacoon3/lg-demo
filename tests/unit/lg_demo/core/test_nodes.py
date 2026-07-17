import os
from enum import Enum
from importlib import util
from pathlib import Path

from langchain.messages import AIMessage, HumanMessage, SystemMessage

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
from lg_demo.core.nodes import PromptClassifierNode, bind_tools_to_model, execute_tool_calls
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


def test_execute_tool_calls_returns_runtime_state():
    add_tool = FakeTool(name="add", result="3")
    mult_tool = FakeTool(name="multiply", result="6")

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

    result = execute_tool_calls(state, [add_tool, mult_tool])

    assert add_tool.calls == [{"a": 1, "b": 2}]
    assert mult_tool.calls == [{"a": 2, "b": 3}]
    assert result.llm_calls == 0
    assert result.tool_calls == 2
    assert result.state is None
    assert [message.content for message in result.messages] == ["3", "6"]


def test_execute_tool_calls_raises_for_unknown_tool():
    add_tool = FakeTool(name="add", result="3")
    state = RuntimeState(
        messages=[
            AIMessage(
                content="",
                tool_calls=[{"name": "multiply", "args": {"a": 1, "b": 2}, "id": "call-1"}],
            )
        ],
        llm_calls=0,
        tool_calls=0,
        state={},
    )

    try:
        execute_tool_calls(state, [add_tool])
    except ValueError as exc:
        assert "not supported" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported tool")


def test_bind_tools_to_model_calls_model_bind_tools():
    tool = FakeTool(name="add", result="3")
    model = FakeModel()

    bound = bind_tools_to_model(model, [tool])

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

    result = node(state)

    assert isinstance(model.last_messages[0], SystemMessage)
    assert "arithmetic" in model.last_messages[0].content.lower()
    assert model.last_messages[1].content == "6 * 7"
    assert result.messages == [model_response]
    assert result.llm_calls == 1
    assert result.tool_calls == 0
    assert result.state is None


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

    result = node(state)

    assert isinstance(structured.last_messages[0], SystemMessage)
    assert "math, chat" in structured.last_messages[0].content
    assert result.messages[0].content == "math"
    assert result.llm_calls == 1
    assert result.tool_calls == 0
    assert result.state is None
