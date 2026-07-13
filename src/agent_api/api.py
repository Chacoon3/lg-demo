from fastapi import APIRouter, Request
from langchain.messages import HumanMessage, SystemMessage

from agent_api.dependencies import FinanceAgentDep, GeneralAgentDep

agent_api_router = APIRouter(prefix="/prompt")


@agent_api_router.get("/health_check")
async def health_check():
    return {"status": "ok"}


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
