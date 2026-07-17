from typing import Literal

from fastapi import APIRouter, Request, Response
from langchain.messages import HumanMessage

from agent_api.dependencies import AgentRegistryDep
from agent_api.response_builder import response_ok

agent_api_router = APIRouter(prefix="/prompt")


@agent_api_router.get("/health_check")
async def health_check():
    return response_ok({"status": "ok"})


@agent_api_router.get("/graph", response_class=Response)
async def graph(request: Request, agent_class: Literal["general", "finance", "general_cot"]):
    agent_map = {
        "general": request.app.state.agent_registry.general_agent,
        "finance": request.app.state.agent_registry.finance_agent,
        "general_cot": request.app.state.agent_registry.general_cot,
    }
    agt = agent_map.get(agent_class)
    graph_png = agt.get_graph().draw_mermaid_png()
    return Response(content=graph_png, media_type="image/png")


@agent_api_router.post("/general")
async def general(request: Request, agent_registry: AgentRegistryDep):
    data = await request.json()
    prompt = data.get("prompt")
    is_debug = data.get("debug", False)
    messages = [HumanMessage(content=prompt)]
    ans = agent_registry.general_agent.invoke({"messages": messages})
    last_msg = ans["messages"][-1].content if ans["messages"] else None
    return response_ok(last_msg if not is_debug else ans)


@agent_api_router.post("/general_cot")
async def general_cot(request: Request, agent_registry: AgentRegistryDep):
    data = await request.json()
    prompt = data.get("prompt")
    is_debug = data.get("debug", False)
    messages = [HumanMessage(content=prompt)]
    ans = agent_registry.general_cot.invoke({"messages": messages})
    last_msg = ans["messages"][-1].content if ans["messages"] else None
    return response_ok(last_msg if not is_debug else ans)


@agent_api_router.post("/finance")
async def finance(request: Request, agent_registry: AgentRegistryDep):
    data = await request.json()
    prompt = data.get("prompt")
    is_debug = data.get("debug", False)
    messages = [HumanMessage(content=prompt)]
    ans = agent_registry.finance_agent.invoke({"messages": messages})
    last_msg = ans["messages"][-1].content if ans["messages"] else None
    return response_ok(last_msg if not is_debug else ans)


@agent_api_router.post("/travel")
async def travel(request: Request, agent_registry: AgentRegistryDep):
    data = await request.json()
    prompt = data.get("prompt")
    is_debug = data.get("debug", False)
    messages = [HumanMessage(content=prompt)]
    ans = agent_registry.travel_agent.invoke({"messages": messages})
    ans.pop("state")
    last_msg = ans["messages"][-1].content if ans["messages"] else None
    return response_ok(last_msg if not is_debug else ans)
