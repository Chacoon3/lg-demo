"""FastAPI application entry point."""

import os

from fastapi.concurrency import asynccontextmanager
from langchain.tools import tool
import uvicorn
from fastapi import FastAPI
from agent_api.api import agent_api_router
from agent_api.logging import get_logger
from lg_demo.llm import Agent


# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    get_logger().info("Starting application and configuring model.")

    app.state.agent = Agent

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
