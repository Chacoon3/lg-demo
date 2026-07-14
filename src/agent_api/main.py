"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager

from agent_api.api import agent_api_router
from agent_api.app_logging import get_logger
from agent_api.middleware import logging_middleware
from lg_demo.agents import AgentRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_logger().info("Starting application and configuring model.")

    app.state.agent_registry = AgentRegistry

    get_logger().info("Model and tools have been configured.")

    yield

    get_logger().info("Application exiting.")


app = FastAPI(title="LG Demo Agent API", version="0.1.0", lifespan=lifespan)

app.middleware("http")(logging_middleware)


app.include_router(
    prefix="/agent_api",
    tags=["agent_api"],
    router=agent_api_router,
)
