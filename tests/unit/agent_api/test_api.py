import asyncio
import sys
import types

from langchain.messages import AIMessage

# Avoid importing real agent registry during module import in unit tests.
dependencies_stub = types.ModuleType("agent_api.dependencies")
dependencies_stub.AgentRegistryDep = object
sys.modules["agent_api.dependencies"] = dependencies_stub

from agent_api.api import general, health_check


class FakeRequest:
    def __init__(self, payload, headers=None):
        self._payload = payload
        self.headers = headers or {}

    async def json(self):
        return self._payload


class FakeAgent:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def invoke(self, payload, config=None):
        self.calls.append((payload, config))
        return self._response


class FakeAgentRegistry:
    def __init__(self, general_agent):
        self.general_agent = general_agent


def test_health_check_returns_ok_status():
    result = asyncio.run(health_check())
    assert result == {"data": {"status": "ok"}}


def test_prompt_returns_last_message_content_when_not_debug():
    agent = FakeAgent(response={"messages": [AIMessage(content="answer")]})
    agent_registry = FakeAgentRegistry(general_agent=agent)
    request = FakeRequest(payload={"prompt": "hello"})

    result = asyncio.run(general(request, agent_registry))

    assert result == {"data": "answer"}
    assert agent.calls[0][0]["messages"][0].content == "hello"
    assert agent.calls[0][1] is None


def test_prompt_returns_full_response_when_debug_enabled():
    response = {"messages": [AIMessage(content="answer")], "meta": {"ok": True}}
    agent = FakeAgent(response=response)
    agent_registry = FakeAgentRegistry(general_agent=agent)
    request = FakeRequest(payload={"prompt": "hello", "debug": True})

    result = asyncio.run(general(request, agent_registry))

    assert result == {"data": response}


def test_prompt_returns_none_when_no_messages():
    agent = FakeAgent(response={"messages": []})
    agent_registry = FakeAgentRegistry(general_agent=agent)
    request = FakeRequest(payload={"prompt": "hello"})

    result = asyncio.run(general(request, agent_registry))

    assert result == {"data": None}


def test_prompt_passes_thread_id_to_agent_invoke_config():
    agent = FakeAgent(response={"messages": [AIMessage(content="answer")]})
    agent_registry = FakeAgentRegistry(general_agent=agent)
    request = FakeRequest(payload={"prompt": "hello", "thread_id": "thread-123"})

    result = asyncio.run(general(request, agent_registry))

    assert result == {"data": "answer"}
    assert agent.calls[0][1] == {"configurable": {"thread_id": "thread-123"}}
