from typing import Literal

from fastapi import APIRouter, Request, Response
from langchain.messages import HumanMessage

from agent_api.dependencies import AgentRegistryDep
from agent_api.response_builder import response_ok

agent_api_router = APIRouter(prefix="/prompt")


def _build_invoke_config(request: Request, payload: dict) -> dict | None:
    headers = getattr(request, "headers", {})
    thread_id = payload.get("thread_id") or headers.get("x-thread-id")
    if not thread_id:
        return None
    return {"configurable": {"thread_id": str(thread_id)}}


def _invoke_agent(agent, payload: dict, config: dict | None):
    if config is None:
        return agent.invoke(payload)
    return agent.invoke(payload, config=config)


@agent_api_router.get("/health_check")
async def health_check():
    return response_ok({"status": "ok"})


@agent_api_router.get("/graph", response_class=Response)
async def graph(request: Request, agent_class: Literal["general", "finance"]):
    agent_map = {
        "general": request.app.state.agent_registry.general_agent,
        "finance": request.app.state.agent_registry.finance_agent,
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
    config = _build_invoke_config(request, data)
    ans = _invoke_agent(agent_registry.general_agent, {"messages": messages}, config)
    last_msg = ans["messages"][-1].content if ans["messages"] else None
    return response_ok(last_msg if not is_debug else ans)


@agent_api_router.post("/finance")
async def finance(request: Request, agent_registry: AgentRegistryDep):
    data = await request.json()
    prompt = data.get("prompt")
    is_debug = data.get("debug", False)
    messages = [HumanMessage(content=prompt)]
    config = _build_invoke_config(request, data)
    ans = _invoke_agent(agent_registry.finance_agent, {"messages": messages}, config)
    last_msg = ans["messages"][-1].content if ans["messages"] else None
    return response_ok(last_msg if not is_debug else ans)


@agent_api_router.post("/travel")
async def travel(request: Request, agent_registry: AgentRegistryDep):
    data = await request.json()
    prompt = data.get("prompt")
    is_debug = data.get("debug", False)
    messages = [HumanMessage(content=prompt)]
    config = _build_invoke_config(request, data)
    ans = _invoke_agent(agent_registry.travel_agent, {"messages": messages}, config)
    ans.pop("state")
    last_msg = ans["messages"][-1].content if ans["messages"] else None
    return response_ok(last_msg if not is_debug else ans)
