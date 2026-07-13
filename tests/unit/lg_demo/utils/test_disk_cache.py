import pytest

from lg_demo.utils.disk_cache import DiskCache


def test_disk_cache_returns_cached_value(tmp_path):
    calls = {"count": 0}
    cache = DiskCache(tmp_path / "cache")

    @cache
    def expensive(value: int) -> int:
        calls["count"] += 1
        return value * 10

    assert expensive(3) == 30
    assert expensive(3) == 30
    assert calls["count"] == 1


def test_disk_cache_persists_between_instances(tmp_path):
    cache_dir = tmp_path / "cache"
    calls = {"count": 0}

    cache_one = DiskCache(cache_dir)

    @cache_one
    def compute(value: int) -> dict[str, int]:
        calls["count"] += 1
        return {"value": value}

    assert compute(7) == {"value": 7}
    assert calls["count"] == 1

    cache_two = DiskCache(cache_dir)

    @cache_two
    def compute(value: int) -> dict[str, int]:
        calls["count"] += 1
        return {"value": value, "fresh": 1}

    # Should return the cached payload from the first function execution.
    assert compute(7) == {"value": 7}
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_disk_cache_supports_async_functions(tmp_path):
    calls = {"count": 0}
    cache = DiskCache(tmp_path / "cache")

    @cache.wrap
    async def fetch(value: int) -> int:
        calls["count"] += 1
        return value + 5

    assert await fetch(9) == 14
    assert await fetch(9) == 14
    assert calls["count"] == 1


def test_disk_cache_wraps_instance_methods(tmp_path):
    cache = DiskCache(tmp_path / "cache")

    class Service:
        def __init__(self) -> None:
            self.calls = 0

        @cache
        def score(self, value: int) -> int:
            self.calls += 1
            return value * 2

    first = Service()
    second = Service()

    assert first.score(4) == 8
    assert second.score(4) == 8
    assert first.calls == 1
    assert second.calls == 0


def test_disk_cache_clear_removes_entries(tmp_path):
    cache_dir = tmp_path / "cache"
    cache = DiskCache(cache_dir)

    @cache
    def ping() -> str:
        return "ok"

    assert ping() == "ok"
    assert list(cache_dir.glob("*.pkl"))

    cache.clear()

    assert list(cache_dir.glob("*.pkl")) == []


def test_disk_cache_ttl_expires_entry(tmp_path, monkeypatch):
    now = {"value": 1_000.0}

    def fake_time() -> float:
        return now["value"]

    monkeypatch.setattr("lg_demo.utils.disk_cache.time.time", fake_time)

    calls = {"count": 0}
    cache = DiskCache(tmp_path / "cache", ttl=10)

    @cache
    def compute(value: int) -> int:
        calls["count"] += 1
        return value + 1

    assert compute(5) == 6
    assert compute(5) == 6
    assert calls["count"] == 1

    now["value"] += 11

    assert compute(5) == 6
    assert calls["count"] == 2


def test_disk_cache_wrap_ttl_overrides_default(tmp_path, monkeypatch):
    now = {"value": 2_000.0}

    def fake_time() -> float:
        return now["value"]

    monkeypatch.setattr("lg_demo.utils.disk_cache.time.time", fake_time)

    calls = {"count": 0}
    cache = DiskCache(tmp_path / "cache", ttl=100)

    @cache.wrap(ttl=1)
    def compute(value: int) -> int:
        calls["count"] += 1
        return value * 3

    assert compute(2) == 6
    assert compute(2) == 6
    assert calls["count"] == 1

    now["value"] += 2

    assert compute(2) == 6
    assert calls["count"] == 2
