from fastapi import APIRouter, Request

from agent_api.dependencies import ModelDep

agent_api_router = APIRouter()


@agent_api_router.get("/health_check")
async def health_check():
    return {"status": "ok"}


@agent_api_router.post("/prompt")
async def prompt(request: Request, model: ModelDep):
    data = await request.json()
    prompt = data.get("prompt")
    ans = model.invoke(prompt)
    return {"answer": ans}
