"""FastAPI application entry point."""

from fastapi.concurrency import asynccontextmanager
from langchain_ollama import ChatOllama
import uvicorn
from fastapi import FastAPI
from agent_api.api import agent_api_router
from agent_api.logging import get_logger
from lg_demo.tools import divide, add, multiply


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_logger().info("Starting application and configuring model.")

    model = ChatOllama(
        model="qwen2.5:7b-instruct",
        temperature=0,
        num_gpu=1,
    )

    # Augment the LLM with tools
    tools = [add, multiply, divide]
    model_with_tools = model.bind_tools(tools)

    app.state.model = model
    app.state.model_with_tools = model_with_tools

    get_logger().info("Model and tools have been configured.")

    yield

    get_logger().info("Application exiting.")


app = FastAPI(title="LG Demo Agent API", version="0.1.0", lifespan=lifespan)


app.include_router(
    prefix="/agent_api",
    tags=["agent_api"],
    router=agent_api_router,
)

if __name__ == "__main__":

    uvicorn.run("agent_api.main:app", host="127.0.0.1", port=8000, reload=True)
