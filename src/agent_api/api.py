from typing import Literal

from fastapi import APIRouter, Request, Response
from langchain.messages import HumanMessage, SystemMessage

from agent_api.dependencies import FinanceAgentDep, GeneralAgentDep

agent_api_router = APIRouter(prefix="/prompt")


@agent_api_router.get("/health_check")
async def health_check():
    return {"status": "ok"}


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
    with_plan = data.get("with_plan", False)
    messages = [HumanMessage(content=prompt)]
    if with_plan:
        messages.append(
            SystemMessage(
                content="Plan the necessary tasks to complete the request and then execute them."
            )
        )
    ans = agent.invoke({"messages": messages})
    last_msg = ans["messages"][-1].content if ans["messages"] else None
    return last_msg if not is_debug else ans


@agent_api_router.post("/finance")
async def finance(request: Request, agent: FinanceAgentDep):
    data = await request.json()
    prompt = data.get("prompt")
    is_debug = data.get("debug", False)
    with_plan = data.get("with_plan", False)
    messages = [HumanMessage(content=prompt)]
    if with_plan:
        messages.append(
            SystemMessage(
                content="Plan the necessary tasks to complete the request and then execute them."
            )
        )
    ans = agent.invoke({"messages": messages})
    last_msg = ans["messages"][-1].content if ans["messages"] else None
    return last_msg if not is_debug else ans
