from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from agent_api.middleware import (
    build_pydantic_error_response,
    exception_handling_middleware,
    logging_middleware,
)


def create_test_app(*, debug: bool) -> FastAPI:
    app = FastAPI(debug=debug)
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

    class ValidateBody(BaseModel):
        age: int
        name: str

    class RuntimeModel(BaseModel):
        count: int

    @app.get("/explode")
    def explode() -> dict[str, bool]:
        raise RuntimeError("boom")

    @app.get("/http-500")
    def http_500() -> None:
        raise HTTPException(status_code=500, detail="db unavailable")

    @app.post("/validate-body")
    def validate_body(payload: ValidateBody) -> dict[str, bool]:
        return {"ok": bool(payload.name)}

    @app.get("/runtime-validation")
    def runtime_validation() -> None:
        RuntimeModel.model_validate({"count": "bad"})

    return app


def test_non_debug_masks_unhandled_exception() -> None:
    client = TestClient(create_test_app(debug=False))

    response = client.get("/explode")

    assert response.status_code == 500
    assert response.json() == {"error": "Internal Server Error"}
    assert response.headers.get("x-request-id")


def test_non_debug_masks_explicit_5xx_response() -> None:
    client = TestClient(create_test_app(debug=False))

    response = client.get("/http-500")

    assert response.status_code == 500
    assert response.json() == {"error": "Internal Server Error"}
    assert response.headers.get("x-request-id")


def test_debug_surfaces_unhandled_exception_details() -> None:
    client = TestClient(create_test_app(debug=True))

    response = client.get("/explode")

    assert response.status_code == 500
    assert response.json() == {
        "error": "boom",
        "meta": {"type": "RuntimeError"},
    }
    assert response.headers.get("x-request-id")


def test_debug_keeps_explicit_5xx_response_details() -> None:
    client = TestClient(create_test_app(debug=True))

    response = client.get("/http-500")

    assert response.status_code == 500
    assert response.json() == {"detail": "db unavailable"}
    assert response.headers.get("x-request-id")


def test_request_validation_error_is_human_readable() -> None:
    client = TestClient(create_test_app(debug=False))

    response = client.post("/validate-body", json={"age": "oops"})

    assert response.status_code == 422
    body = response.json()
    assert body["error"].startswith("Validation failed:")
    assert "body.age: Input should be a valid integer" in body["error"]
    assert "body.name: Field required" in body["error"]
    assert body["meta"]["details"] == [
        "body.age: Input should be a valid integer, unable to parse string as an integer",
        "body.name: Field required",
    ]
    assert response.headers.get("x-request-id")


def test_runtime_pydantic_error_is_human_readable() -> None:
    client = TestClient(create_test_app(debug=False))

    response = client.get("/runtime-validation")

    assert response.status_code == 422
    body = response.json()
    assert body["error"].startswith("Validation failed:")
    assert "count: Input should be a valid integer" in body["error"]
    assert body["meta"]["details"] == [
        "count: Input should be a valid integer, unable to parse string as an integer"
    ]
    assert response.headers.get("x-request-id")
