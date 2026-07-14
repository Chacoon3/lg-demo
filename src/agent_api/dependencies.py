from typing import Annotated

from fastapi import Depends, Request

from lg_demo.agents import _AgentRegistry


def agent_registry(request: Request):
    return request.app.state.agent_registry


AgentRegistryDep = Annotated[
    _AgentRegistry,
    Depends(agent_registry),
]
