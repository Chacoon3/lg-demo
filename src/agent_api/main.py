"""FastAPI application entry point."""

import os
from contextlib import ExitStack

from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

import lg_demo.core.model_provider as model_provider
from agent_api.api import agent_api_router
from agent_api.app_logging import get_logger
from agent_api.middleware import (
    build_pydantic_error_response,
    exception_handling_middleware,
    logging_middleware,
)
from lg_demo.agents import AgentRegistry
from lg_demo.core.checkpointing import app_checkpointer_context


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_logger().info("Starting application and configuring model.")

    provider = model_provider.OpenAICloudProvider()
    get_logger().info("Model provider configured.", model=provider.get_model_key())

    with ExitStack() as stack:
        checkpointer = stack.enter_context(app_checkpointer_context())
        app.state.checkpointer = checkpointer
        app.state.agent_registry = AgentRegistry(provider.get_model(), checkpointer=checkpointer)

        get_logger().info(
            "Model and tools have been configured.",
            checkpointer_backend=os.getenv("LG_DEMO_CHECKPOINTER_BACKEND", "none"),
        )

        yield

    get_logger().info("Application exiting.")


app = FastAPI(
    title="LG Demo Agent API",
    version="0.1.0",
    lifespan=lifespan,
    debug=os.getenv("DEBUG", "false").lower() == "true",
)

app.middleware("http")(exception_handling_middleware)
app.middleware("http")(logging_middleware)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request, exc: RequestValidationError):
    request_id = request.headers.get("x-request-id") or getattr(
        request.state,
        "request_id",
        "unknown",
    )
    return build_pydantic_error_response(request_id, exc)


@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request, exc: ValidationError):
    request_id = request.headers.get("x-request-id") or getattr(
        request.state,
        "request_id",
        "unknown",
    )
    return build_pydantic_error_response(request_id, exc)


app.include_router(
    prefix="/agent_api",
    tags=["agent_api"],
    router=agent_api_router,
)
