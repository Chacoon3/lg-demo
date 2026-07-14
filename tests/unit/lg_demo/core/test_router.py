import pytest
from langchain.messages import AIMessage

from lg_demo.core.nodes import BaseNode
from lg_demo.core.router import EntryRouter, PromptClassRouter, ToolCallRouter
from lg_demo.core.states import RuntimeState


def test_entry_router_call_raises_not_implemented():
    node = BaseNode(name="entry", priority=0)
    router = EntryRouter(entry_node=node)
    state = RuntimeState(messages=[AIMessage(content="x")], llm_calls=0, tool_calls=0, state={})

    with pytest.raises(NotImplementedError, match="router should not be called directly"):
        router(state)


def test_prompt_class_router_routes_to_matching_classification():
    router = PromptClassRouter(
        from_nodes=[BaseNode(name="from", priority=0)],
        to_nodes=[BaseNode(name="math", priority=0), BaseNode(name="chat", priority=0)],
    )
    state = RuntimeState(messages=[AIMessage(content="math")], llm_calls=0, tool_calls=0, state={})

    assert router(state) == "math"


def test_prompt_class_router_returns_end_when_not_in_exit_set():
    router = PromptClassRouter(
        from_nodes=[BaseNode(name="from", priority=0)],
        to_nodes=[BaseNode(name="math", priority=0)],
    )
    state = RuntimeState(
        messages=[AIMessage(content="unknown")], llm_calls=0, tool_calls=0, state={}
    )

    assert router(state) == router.END


def test_tool_call_router_routes_to_tool_name_when_present_and_allowed():
    router = ToolCallRouter(
        from_nodes=[BaseNode(name="from", priority=0)],
        to_nodes=[BaseNode(name="add", priority=0), BaseNode(name="divide", priority=0)],
    )
    state = RuntimeState(
        messages=[AIMessage(content="", tool_calls=[{"name": "divide", "args": {}, "id": "1"}])],
        llm_calls=0,
        tool_calls=0,
        state={},
    )

    assert router(state) == "divide"


def test_tool_call_router_returns_end_without_tool_calls_or_unknown_tool():
    router = ToolCallRouter(
        from_nodes=[BaseNode(name="from", priority=0)],
        to_nodes=[BaseNode(name="add", priority=0)],
    )
    no_call_state = RuntimeState(
        messages=[AIMessage(content="")], llm_calls=0, tool_calls=0, state={}
    )
    unknown_call_state = RuntimeState(
        messages=[AIMessage(content="", tool_calls=[{"name": "multiply", "args": {}, "id": "1"}])],
        llm_calls=0,
        tool_calls=0,
        state={},
    )

    assert router(no_call_state) == router.END
    with pytest.raises(ValueError, match="not in the exit set"):
        router(unknown_call_state)
