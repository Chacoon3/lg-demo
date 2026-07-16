from typing import Generator, Sequence

import networkx as nx

from lg_demo.core.states import AgentTask


class Dag:

    def __init__(self, tasks: Sequence[AgentTask]):
        self.tasks = tasks
        self.graph = nx.DiGraph()
        for task in tasks:
            self.graph.add_node(task.name, task=task)
            for dep in task.dependencies:
                self.graph.add_edge(dep, task.name)

    def __repr__(self):
        return f"Dag(tasks={[task.name for task in self.tasks]})"

    def __str__(self):
        return self.__repr__()

    def __iter__(self) -> Generator[AgentTask, None, None]:
        return (self.graph.nodes[node]["task"] for node in nx.topological_sort(self.graph))

    def __len__(self) -> int:
        return len(self.tasks)

    def iter_layers(self) -> Generator[list[AgentTask], None, None]:
        for layer in nx.topological_generations(self.graph):
            yield [self.graph.nodes[node]["task"] for node in layer]
