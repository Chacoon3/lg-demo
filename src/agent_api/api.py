from fastapi import APIRouter

agent_api_router = APIRouter()


@agent_api_router.post("/prompt")
async def prompt(prompt: str):
    return {"prompt": prompt}
