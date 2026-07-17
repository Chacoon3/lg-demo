from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator


def _read_backend() -> str:
    return os.getenv("LG_DEMO_CHECKPOINTER_BACKEND", "").strip().lower()


def _read_postgres_dsn() -> str:
    dsn = os.getenv("LG_DEMO_POSTGRES_DSN") or os.getenv("DATABASE_URL")
    if not dsn:
        raise ValueError(
            "Postgres checkpointer requires LG_DEMO_POSTGRES_DSN or DATABASE_URL to be set."
        )
    return dsn


@contextmanager
def postgres_checkpointer_context(
    connection_string: str | None = None,
    *,
    setup: bool = True,
) -> Iterator[Any]:
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as exc:
        raise ImportError(
            "Postgres checkpoint support requires the langgraph-checkpoint-postgres package."
        ) from exc

    dsn = connection_string or _read_postgres_dsn()

    with PostgresSaver.from_conn_string(dsn) as checkpointer:
        if setup and hasattr(checkpointer, "setup"):
            checkpointer.setup()
        yield checkpointer


@contextmanager
def app_checkpointer_context() -> Iterator[Any | None]:
    backend = _read_backend()

    if backend in {"", "none"}:
        yield None
        return

    if backend == "postgres":
        with postgres_checkpointer_context() as checkpointer:
            yield checkpointer
        return

    raise ValueError(f"Unsupported checkpointer backend: {backend}")
