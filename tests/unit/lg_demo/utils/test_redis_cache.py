import time

import pytest

from lg_demo.utils.redis_cache import RedisCache


class FakeRedis:
    def __init__(self) -> None:
        self._values: dict[str, bytes] = {}
        self._expires_at: dict[str, float] = {}

    def get(self, key: str):
        expires_at = self._expires_at.get(key)
        if expires_at is not None and time.time() >= expires_at:
            self._values.pop(key, None)
            self._expires_at.pop(key, None)
            return None
        return self._values.get(key)

    def set(self, key: str, value: bytes, px: int | None = None):
        self._values[key] = value
        if px is None:
            self._expires_at.pop(key, None)
            return True

        self._expires_at[key] = time.time() + (px / 1000.0)
        return True

    def delete(self, *keys: str):
        for key in keys:
            self._values.pop(key, None)
            self._expires_at.pop(key, None)

    def scan_iter(self, match: str):
        prefix = match.rstrip("*")
        return (key for key in list(self._values) if key.startswith(prefix))


def test_redis_cache_returns_cached_value():
    calls = {"count": 0}
    cache = RedisCache(client=FakeRedis())

    @cache
    def expensive(value: int) -> int:
        calls["count"] += 1
        return value * 10

    assert expensive(3) == 30
    assert expensive(3) == 30
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_redis_cache_supports_async_functions():
    calls = {"count": 0}
    cache = RedisCache(client=FakeRedis())

    @cache.wrap
    async def fetch(value: int) -> int:
        calls["count"] += 1
        return value + 5

    assert await fetch(9) == 14
    assert await fetch(9) == 14
    assert calls["count"] == 1


def test_redis_cache_ttl_expires_entry(monkeypatch):
    now = {"value": 1_000.0}

    def fake_time() -> float:
        return now["value"]

    monkeypatch.setattr(time, "time", fake_time)

    calls = {"count": 0}
    cache = RedisCache(client=FakeRedis(), ttl=1)

    @cache
    def compute(value: int) -> int:
        calls["count"] += 1
        return value + 1

    assert compute(5) == 6
    assert compute(5) == 6
    assert calls["count"] == 1

    now["value"] += 2

    assert compute(5) == 6
    assert calls["count"] == 2


def test_redis_cache_clear_removes_entries():
    redis = FakeRedis()
    cache = RedisCache(client=redis, prefix="test-prefix")

    @cache
    def ping() -> str:
        return "ok"

    assert ping() == "ok"
    assert any(redis.scan_iter("test-prefix:*"))

    cache.clear()

    assert list(redis.scan_iter("test-prefix:*")) == []
