"""HTTP middleware for request/response logging."""

from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from fastapi import Request

from agent_api.app_logging import get_logger

logger = get_logger(__name__)


async def logging_middleware(request: Request, call_next):
    """Log request and response metadata for each HTTP call."""

    request_id = request.headers.get("x-request-id", str(uuid4()))
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
