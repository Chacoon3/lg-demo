"""HTTP middleware for request/response logging and error handling."""

from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from agent_api.app_logging import get_logger
from agent_api.response_builder import response_client_error

logger = get_logger(__name__)


def _format_validation_details(exc: ValidationError | RequestValidationError) -> list[str]:
    details: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error.get("loc", ())) or "input"
        msg = error.get("msg", "Invalid value")
        details.append(f"{loc}: {msg}")
    return details


def build_pydantic_error_response(
    request_id: str,
    exc: ValidationError | RequestValidationError,
) -> JSONResponse:
    """Return a readable client error response for validation failures."""

    details = _format_validation_details(exc)
    message = "Validation failed"
    if details:
        message = f"Validation failed: {'; '.join(details)}"

    response = JSONResponse(
        status_code=422,
        content=response_client_error(message, meta={"details": details}),
    )
    response.headers["x-request-id"] = request_id
    return response


def _request_id_from(request: Request) -> str:
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        return request_id
    return request.headers.get("x-request-id", str(uuid4()))


async def exception_handling_middleware(request: Request, call_next):
    """Handle unhandled exceptions and normalize 5xx responses.

    In debug mode, error details are returned to the client.
    In non-debug mode, 5xx details are masked with a generic message.
    """

    request_id = _request_id_from(request)

    try:
        response = await call_next(request)
    except (ValidationError, RequestValidationError) as exc:
        logger.info(
            "validation_error",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            errors=exc.errors(),
        )
        return build_pydantic_error_response(request_id, exc)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception(
            "unhandled_exception",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            error_type=type(exc).__name__,
        )

        if request.app.debug:
            response = JSONResponse(
                status_code=500,
                content=response_client_error(
                    str(exc),
                    meta={"type": type(exc).__name__},
                ),
            )
        else:
            response = JSONResponse(
                status_code=500,
                content=response_client_error("Internal Server Error"),
            )

        response.headers["x-request-id"] = request_id
        return response

    if response.status_code >= 500 and not request.app.debug:
        logger.error(
            "server_error_response_masked",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
        )
        masked = JSONResponse(
            status_code=response.status_code,
            content=response_client_error("Internal Server Error"),
        )
        masked.headers["x-request-id"] = request_id
        return masked

    response.headers["x-request-id"] = request_id
    return response


async def logging_middleware(request: Request, call_next):
    """Log request and response metadata for each HTTP call."""

    request_id = request.headers.get("x-request-id", str(uuid4()))
    request.state.request_id = request_id
    start_time = perf_counter()
    client_host = request.client.host if request.client else "unknown"

    logger.info(
        "request_started",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        query=request.url.query,
        client_ip=client_host,
    )

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round((perf_counter() - start_time) * 1000, 2)
        logger.exception(
            "request_failed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            duration_ms=duration_ms,
        )
        raise

    duration_ms = round((perf_counter() - start_time) * 1000, 2)
    response.headers["x-request-id"] = request_id
    content_length = response.headers.get("content-length")

    logger.info(
        "request_completed",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
        response_size_bytes=int(content_length) if content_length else None,
    )

    return response
