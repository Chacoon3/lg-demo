from fastapi import APIRouter, Request
from langchain.messages import HumanMessage

from agent_api.dependencies import AgentDep

agent_api_router = APIRouter()


@agent_api_router.get("/health_check")
async def health_check():
    return {"status": "ok"}


@agent_api_router.post("/prompt")
async def prompt(request: Request, agent: AgentDep):
    data = await request.json()
    prompt = data.get("prompt")
    messages = [HumanMessage(content=prompt)]
    ans = agent.invoke({"messages": messages})
    return {"answer": ans}
