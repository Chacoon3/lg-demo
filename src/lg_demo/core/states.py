from __future__ import annotations

import operator
from functools import cached_property
from typing import Annotated, Any, Literal, Optional

from langchain.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class RuntimeState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: Annotated[int, operator.add]
    tool_calls: Annotated[int, operator.add]
    state: Annotated[dict[str, Any] | None, operator.or_] = Field(None, exclude=True)


class AgentTaskDependency(BaseModel):
    predecessor: Optional[str]
    successor: str


class AgentTask(BaseModel):
    name: str
    description: str
    dependencies: list[str] = []
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"

    @cached_property
    def get_dependencies(self) -> list[AgentTaskDependency]:
        if self.dependencies:
            return [
                AgentTaskDependency(predecessor=dep, successor=self.name)
                for dep in self.dependencies
            ]
        return [AgentTaskDependency(predecessor=None, successor=self.name)]


class AgentPlan(BaseModel):
    tasks: list[AgentTask]
