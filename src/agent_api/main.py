"""FastAPI application entry point."""

import os

from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager

from agent_api.api import agent_api_router
from agent_api.logging import get_logger
from lg_demo.agent import Agent


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
