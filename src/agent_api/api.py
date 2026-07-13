from typing import Literal

from fastapi import APIRouter, Request, Response
from langchain.messages import HumanMessage

from agent_api.dependencies import FinanceAgentDep, GeneralAgentDep
from agent_api.response_builder import response_ok

agent_api_router = APIRouter(prefix="/prompt")


@agent_api_router.get("/health_check")
async def health_check():
    return response_ok({"status": "ok"})


@agent_api_router.get("/graph", response_class=Response)
async def graph(request: Request, agent_class: Literal["general", "finance"]):
    agent_map = {
        "general": request.app.state.agent,
        "finance": request.app.state.finance_agent,
    }
    agt = agent_map.get(agent_class)
    graph_png = agt.get_graph().draw_mermaid_png()
    return Response(content=graph_png, media_type="image/png")


@agent_api_router.post("/general")
async def general(request: Request, agent: GeneralAgentDep):
    data = await request.json()
    prompt = data.get("prompt")
    is_debug = data.get("debug", False)
    messages = [HumanMessage(content=prompt)]
    ans = agent.invoke({"messages": messages})
    last_msg = ans["messages"][-1].content if ans["messages"] else None
    return response_ok(last_msg if not is_debug else ans)


@agent_api_router.post("/finance")
async def finance(request: Request, agent: FinanceAgentDep):
    data = await request.json()
    prompt = data.get("prompt")
    is_debug = data.get("debug", False)
    messages = [HumanMessage(content=prompt)]
    ans = agent.invoke({"messages": messages})
    last_msg = ans["messages"][-1].content if ans["messages"] else None
    return response_ok(last_msg if not is_debug else ans)


@agent_api_router.post("/planned_finance")
async def planned_finance(request: Request, agent: FinanceAgentDep):
    pass
