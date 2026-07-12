import pytest

from lg_demo.core.nodes import BaseNode
from lg_demo.core.router import DirectRouter, EntryRouter, PromptClassRouter
from lg_demo.core.runtime import RuntimeBuilder


class FakeStateGraph:
    last_instance = None

    def __init__(self, state_type):
        self.state_type = state_type
        self.nodes = []
        self.edges = []
        FakeStateGraph.last_instance = self

    def add_node(self, name, node):
        self.nodes.append((name, node))

    def add_edge(self, from_name, to_name):
        self.edges.append((from_name, to_name))

    def compile(self):
        return "compiled-graph"


def test_runtime_builder_adds_nodes_in_priority_order_and_wires_routers(monkeypatch):
    import lg_demo.core.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "StateGraph", FakeStateGraph)

    node_high = BaseNode(name="high", priority=2)
    node_low = BaseNode(name="low", priority=1)

    routers = [
        EntryRouter(entry_node=node_low),
        DirectRouter(from_nodes=[node_high], to_nodes=[node_low]),
        PromptClassRouter(from_nodes=[node_low], to_nodes=[node_high]),
    ]

    builder = RuntimeBuilder(nodes=[node_high, node_low], routers=routers)
    compiled = builder.build()
    graph = FakeStateGraph.last_instance

    assert compiled == "compiled-graph"
    assert [name for name, _ in graph.nodes] == ["low", "high"]
    assert ("__start__", "low") in graph.edges
    assert ("high", "low") in graph.edges
    assert ("low", "high") in graph.edges


def test_runtime_builder_rejects_multiple_entry_routers(monkeypatch):
    import lg_demo.core.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "StateGraph", FakeStateGraph)

    node = BaseNode(name="n", priority=0)
    builder = RuntimeBuilder(
        nodes=[node],
        routers=[EntryRouter(node), EntryRouter(node)],
    )

    with pytest.raises(ValueError, match="Multiple entry routers"):
        builder.build()


def test_runtime_builder_rejects_unsupported_router_type(monkeypatch):
    import lg_demo.core.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "StateGraph", FakeStateGraph)

    node = BaseNode(name="n", priority=0)
    builder = RuntimeBuilder(nodes=[node], routers=[object()])

    with pytest.raises(ValueError, match="Unsupported router type"):
        builder.build()
