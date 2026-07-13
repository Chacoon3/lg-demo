from __future__ import annotations

import operator
import uuid
from functools import cached_property
from typing import Annotated, Literal, Optional

from langchain.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel


class RuntimeState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: Annotated[int, operator.add]
    tool_calls: Annotated[int, operator.add]


class AgentTask(BaseModel):
    name: str
    task_id: uuid.UUID
    description: str | None = None
    task_type: Literal["tool_call", "inference"]
    dependencies: list[uuid.UUID] = []

    @cached_property
    def get_dependencies(self) -> list[AgentTaskDependency]:
        if self.dependencies:
            return [
                AgentTaskDependency(predecessor_id=dep, successor_id=self.task_id)
                for dep in self.dependencies
            ]
        return [AgentTaskDependency(predecessor_id=None, successor_id=self.task_id)]


class AgentTaskDependency(BaseModel):
    predecessor_id: Optional[uuid.UUID]
    successor_id: uuid.UUID


class AgentPlan(BaseModel):
    tasks: list[AgentTask]
