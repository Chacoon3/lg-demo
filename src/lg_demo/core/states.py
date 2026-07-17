from __future__ import annotations

import operator
from functools import cached_property
from typing import Annotated, Any, Literal, Optional

from langchain.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

TaskState = Literal["pending", "in_progress", "completed", "failed"]


class BaseRuntimeState(BaseModel):
    pass


class RuntimeState(BaseRuntimeState):
    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: Annotated[int, operator.add]
    tool_calls: Annotated[int, operator.add]
    state: Annotated[dict[str, Any] | None, operator.or_] = Field(None, exclude=True)


class SnapshotState(BaseRuntimeState):
    messages: Annotated[list[AnyMessage], None]
    llm_calls: Annotated[int, None] = 0
    tool_calls: Annotated[int, None] = 0
    state: Annotated[Any, None] = Field(None, exclude=True)


class AgentTaskDependency(BaseModel):
    predecessor: Optional[str]
    successor: str


class AgentTask(BaseModel):
    name: str
    description: str
    dependencies: list[str] = []
    status: TaskState = "pending"

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


class Action(BaseModel):
    name: str
    description: str
    state: TaskState = "pending"


class ChainOfAction(BaseModel):
    next_step: int = Field(0, description="Index of the next step to execute")
    steps: list[Action] = Field(
        default_factory=list, description="List of actions to be executed in order"
    )

    # validator to validate the next_step index is within the bounds of the steps list
    @classmethod
    def validate_next_step(cls, v, values):
        if "steps" in values and (v < 0 or v >= len(values["steps"])):
            raise ValueError("next_step index is out of bounds of the steps list")
        # also validate the step being pointed is in a valid state (not completed or failed)
        if "steps" in values and values["steps"]:
            step = values["steps"][v]
            if step.state in ["completed", "failed"]:
                raise ValueError(f"next_step index points to a step that is already {step.state}")
        return v

    def set_current_step_state(self, state: TaskState) -> None:
        if self.steps and 0 <= self.next_step < len(self.steps):
            self.steps[self.next_step].state = state
            if state == "completed":
                self.next_step += 1
        else:
            raise IndexError("next_step index is out of bounds of the steps list")
