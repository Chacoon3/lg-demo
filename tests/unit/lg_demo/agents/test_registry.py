import os

os.environ.setdefault("TAVILY_API_KEY", "test-key")

import lg_demo.agents as agents_module


def test_agent_registry_passes_checkpointer_to_all_builders(monkeypatch):
    calls = []

    def fake_general(model, checkpointer=None):
        calls.append(("general", model, checkpointer))
        return "general-agent"

    def fake_finance(model, checkpointer=None):
        calls.append(("finance", model, checkpointer))
        return "finance-agent"

    def fake_travel(model, checkpointer=None):
        calls.append(("travel", model, checkpointer))
        return "travel-agent"

    monkeypatch.setattr(agents_module, "build_simple_arithmetic_agent", fake_general)
    monkeypatch.setattr(agents_module, "build_trading_agent", fake_finance)
    monkeypatch.setattr(agents_module, "build_travel_agent", fake_travel)

    model = object()
    checkpointer = object()
    registry = agents_module.AgentRegistry(model, checkpointer=checkpointer)

    assert registry.general_agent == "general-agent"
    assert registry.finance_agent == "finance-agent"
    assert registry.travel_agent == "travel-agent"
    assert calls == [
        ("general", model, checkpointer),
        ("finance", model, checkpointer),
        ("travel", model, checkpointer),
    ]
