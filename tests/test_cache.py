import asyncio
import json
import os
import tempfile
import time

from axiompy import Axiom


def test_put_get():
    Axiom.cache.clear()
    Axiom.cache.put("key1", 42)
    assert Axiom.cache.get("key1") == 42
    Axiom.cache.invalidate("key1")
    assert Axiom.cache.get("key1") is None


def test_ttl():
    Axiom.cache.put("tmp", "value", ttl=0.1)
    assert Axiom.cache.get("tmp") == "value"
    time.sleep(0.15)
    assert Axiom.cache.get("tmp") is None


def test_clear():
    Axiom.cache.put("a", 1)
    Axiom.cache.put("b", 2)
    assert Axiom.cache.size == 2
    Axiom.cache.clear()
    assert Axiom.cache.size == 0


def test_memoize():
    Axiom.cache.clear()
    call_count = 0

    @Axiom.cache.memoize(ttl=5)
    def fn(x):
        nonlocal call_count
        call_count += 1
        return x * x

    assert fn(3) == 9
    assert call_count == 1
    assert fn(3) == 9
    assert call_count == 1  # cached
    assert fn(4) == 16
    assert call_count == 2


def test_hits_and_misses():
    Axiom.cache.clear()
    assert Axiom.cache.hits == 0
    assert Axiom.cache.misses == 0

    Axiom.cache.put("a", 1)
    Axiom.cache.get("a")  # hit
    assert Axiom.cache.hits == 1

    Axiom.cache.get("nonexistent")  # miss
    assert Axiom.cache.misses == 1


def test_hit_ratio():
    Axiom.cache.clear()
    assert Axiom.cache.hit_ratio == 0.0

    Axiom.cache.put("a", 1)
    Axiom.cache.get("a")  # hit
    Axiom.cache.get("a")  # hit
    Axiom.cache.get("x")  # miss
    assert Axiom.cache.hit_ratio == 2 / 3


def test_stats():
    Axiom.cache.clear()
    Axiom.cache.put("a", 1)
    Axiom.cache.get("a")
    stats = Axiom.cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 0
    assert stats["size"] == 1


def test_info():
    Axiom.cache.clear()
    info = Axiom.cache.info()
    assert "entries" in info.lower()
    assert "misses" in info.lower()


def test_save_and_load():
    Axiom.cache.clear()
    Axiom.cache.put("saved", 99)
    Axiom.cache.put("another", [1, 2, 3])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name

    try:
        Axiom.cache.save(path)

        # Verify file exists and is valid JSON
        with open(path) as f:
            data = json.load(f)
        assert "entries" in data
        assert any(e["key"] == "saved" for e in data["entries"])

        # Clear and reload
        Axiom.cache.clear()
        assert Axiom.cache.size == 0

        Axiom.cache.load(path)
        assert Axiom.cache.get("saved") == 99
        assert Axiom.cache.get("another") == [1, 2, 3]
    finally:
        os.unlink(path)


def test_async_memoize():
    Axiom.cache.clear()
    call_count = 0

    @Axiom.cache.async_memoize(ttl=5)
    async def afn(x):
        nonlocal call_count
        call_count += 1
        return x * 3

    async def run():
        assert await afn(3) == 9
        assert call_count == 1
        assert await afn(3) == 9
        assert call_count == 1  # cached
        assert await afn(4) == 12
        assert call_count == 2

    asyncio.run(run())
