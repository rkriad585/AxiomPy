import time

from axiompy import Axiom


def test_put_get():
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
