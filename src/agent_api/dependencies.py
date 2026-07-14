from typing import Annotated

from fastapi import Depends, Request

from lg_demo.agents import AgentRegistry


def agent_registry(request: Request):
    return request.app.state.agent_registry


AgentRegistryDep = Annotated[
    AgentRegistry,
    Depends(agent_registry),
]
