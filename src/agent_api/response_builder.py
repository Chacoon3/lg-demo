def response_ok(data, *, meta: dict | None = None) -> dict:
    """Return a standard JSON response envelope for all API endpoints."""
    resp = {"data": data}
    if meta is not None:
        resp["meta"] = meta
    return resp


def response_client_error(error_message: str, *, meta: dict | None = None) -> dict:
    """Return a standard JSON response envelope for client errors."""
    resp = {"error": error_message}
    if meta is not None:
        resp["meta"] = meta
    return resp
