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
    state: Annotated[dict, operator.or_]


class AgentTaskDependency(BaseModel):
    predecessor_id: Optional[uuid.UUID]
    successor_id: uuid.UUID


class AgentTask(BaseModel):
    task_type: Literal["tool_call", "inference"]
    name: str
    task_id: uuid.UUID
    description: str | None = None
    dependencies: list[uuid.UUID] = []
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"

    @cached_property
    def get_dependencies(self) -> list[AgentTaskDependency]:
        if self.dependencies:
            return [
                AgentTaskDependency(predecessor_id=dep, successor_id=self.task_id)
                for dep in self.dependencies
            ]
        return [AgentTaskDependency(predecessor_id=None, successor_id=self.task_id)]


class AgentPlan(BaseModel):
    tasks: list[AgentTask]
