import sys
import types

import pytest

from lg_demo.core.checkpointing import app_checkpointer_context, postgres_checkpointer_context


def test_app_checkpointer_context_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("LG_DEMO_CHECKPOINTER_BACKEND", raising=False)

    with app_checkpointer_context() as checkpointer:
        assert checkpointer is None


def test_app_checkpointer_context_raises_for_unsupported_backend(monkeypatch):
    monkeypatch.setenv("LG_DEMO_CHECKPOINTER_BACKEND", "invalid")

    with pytest.raises(ValueError, match="Unsupported checkpointer backend"):
        with app_checkpointer_context():
            pass


def test_postgres_checkpointer_context_uses_postgres_saver_and_runs_setup(monkeypatch):
    calls = {}

    class FakeCheckpointer:
        def setup(self):
            calls["setup"] = calls.get("setup", 0) + 1

    class FakeContextManager:
        def __enter__(self):
            calls["entered"] = True
            return FakeCheckpointer()

        def __exit__(self, exc_type, exc, tb):
            calls["exited"] = True
            return False

    class FakePostgresSaver:
        @classmethod
        def from_conn_string(cls, dsn):
            calls["dsn"] = dsn
            return FakeContextManager()

    fake_module = types.ModuleType("langgraph.checkpoint.postgres")
    fake_module.PostgresSaver = FakePostgresSaver
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres", fake_module)

    with postgres_checkpointer_context("postgresql://demo") as checkpointer:
        assert isinstance(checkpointer, FakeCheckpointer)

    assert calls == {
        "dsn": "postgresql://demo",
        "entered": True,
        "setup": 1,
        "exited": True,
    }
