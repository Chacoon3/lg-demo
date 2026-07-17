from typing import Generic, TypeVar

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from lg_demo.core.nodes import BaseNode
from lg_demo.core.router import BaseRouter, ConditionalRouter, DirectRouter, EntryRouter
from lg_demo.core.states import BaseRuntimeState

T = TypeVar("T", bound=BaseRuntimeState)


class RuntimeBuilder(Generic[T]):
    def __init__(self, nodes: list[BaseNode], routers: list[BaseRouter]):
        self.nodes = sorted(nodes, key=lambda node: node.priority)
        self.routers = routers

    def build(self, state_type: type[T]) -> CompiledStateGraph[T]:
        graph = StateGraph(state_type)
        for n in self.nodes:
            graph.add_node(n.name, n)

        start_defined = False
        for r in self.routers:
            if isinstance(r, DirectRouter):
                for from_node in r.from_nodes:
                    for to_node in r.to_nodes:
                        graph.add_edge(from_node.name, to_node.name)
            elif isinstance(r, ConditionalRouter):
                for from_node in r.from_nodes:
                    for to_node in r.to_nodes:
                        graph.add_edge(from_node.name, to_node.name)
            elif isinstance(r, EntryRouter):
                if not start_defined:
                    start_defined = True
                    graph.add_edge("__start__", r.entry_node.name)
                else:
                    raise ValueError("Multiple entry routers are not allowed")
            else:
                raise ValueError(f"Unsupported router type: {type(r)}")

        return graph.compile()
