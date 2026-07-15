from lg_demo.utils import caching
from lg_demo.utils.disk_cache import DiskCache
from lg_demo.utils.redis_cache import RedisCache


def test_build_app_cache_defaults_to_disk(monkeypatch):
    monkeypatch.delenv("APP_CACHE_BACKEND", raising=False)

    cache = caching._build_app_cache()

    assert isinstance(cache, DiskCache)


def test_build_app_cache_uses_redis(monkeypatch):
    class DummyRedisCache:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setenv("APP_CACHE_BACKEND", "redis")
    monkeypatch.setenv("REDIS_URL", "redis://cache.internal:6379/2")
    monkeypatch.setenv("REDIS_CACHE_PREFIX", "demo-prefix")
    monkeypatch.setenv("APP_CACHE_TTL_SECONDS", "42")
    monkeypatch.setattr(caching, "RedisCache", DummyRedisCache)

    cache = caching._build_app_cache()

    assert isinstance(cache, DummyRedisCache)
    assert cache.args == ("redis://cache.internal:6379/2",)
    assert cache.kwargs["prefix"] == "demo-prefix"
    assert cache.kwargs["ttl"] == 42.0


def test_app_disk_cache_aliases_app_cache():
    assert caching.AppDiskCache is caching.AppCache


def test_redis_cache_requires_url_if_client_missing():
    try:
        RedisCache(url=None)
    except ValueError as exc:
        assert "Redis URL" in str(exc)
    else:
        raise AssertionError("Expected ValueError when no URL or client is provided")
