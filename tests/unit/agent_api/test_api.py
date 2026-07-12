import asyncio

from langchain.messages import AIMessage

from agent_api.api import health_check, prompt


class FakeRequest:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


class FakeAgent:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return self._response


def test_health_check_returns_ok_status():
    result = asyncio.run(health_check())
    assert result == {"status": "ok"}


def test_prompt_returns_last_message_content_when_not_debug():
    agent = FakeAgent(response={"messages": [AIMessage(content="answer")]})
    request = FakeRequest(payload={"prompt": "hello"})

    result = asyncio.run(prompt(request, agent))

    assert result == "answer"
    assert agent.calls[0]["messages"][0].content == "hello"


def test_prompt_returns_full_response_when_debug_enabled():
    response = {"messages": [AIMessage(content="answer")], "meta": {"ok": True}}
    agent = FakeAgent(response=response)
    request = FakeRequest(payload={"prompt": "hello", "debug": True})

    result = asyncio.run(prompt(request, agent))

    assert result == response


def test_prompt_returns_none_when_no_messages():
    agent = FakeAgent(response={"messages": []})
    request = FakeRequest(payload={"prompt": "hello"})

    result = asyncio.run(prompt(request, agent))

    assert result is None
