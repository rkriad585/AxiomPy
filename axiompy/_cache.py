"""Least-Recently-Used (LRU) cache with optional TTL and thread safety.

Used internally by ``Axiom.cache`` and the magic function system.

Examples:
    >>> cache = Axiom.cache
    >>> @cache.memoize(ttl=60)
    ... def expensive(x):
    ...     return x ** 2
    >>> expensive(5)  # computed
    25
    >>> expensive(5)  # returned from cache
    25
"""

import time
from collections import OrderedDict
from functools import wraps
from threading import Lock
from typing import Any, Callable, Optional


class LRUCache:
    """A thread-safe LRU cache with optional time-to-live (TTL).

    Args:
        capacity: Maximum number of entries (default 128).
        ttl: Default time-to-live in seconds (default None = no expiry).

    Examples:
        >>> c = LRUCache(capacity=3)
        >>> c.put("a", 1)
        >>> c.get("a")
        1
        >>> c.get("b")  # returns None
    """

    def __init__(self, capacity: int = 128, ttl: Optional[float] = None):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.default_ttl = ttl
        self._store: OrderedDict[Any, tuple[Any, float]] = OrderedDict()
        self._lock = Lock()

    def _now(self) -> float:
        return time.monotonic()

    def _is_expired(self, entry: tuple[Any, float]) -> bool:
        _, expiry = entry
        return expiry > 0 and self._now() > expiry

    def get(self, key: Any) -> Optional[Any]:
        """Get a value by *key*. Returns None if missing or expired."""
        with self._lock:
            if key not in self._store:
                return None
            entry = self._store[key]
            if self._is_expired(entry):
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return entry[0]

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update a key-value pair.

        Args:
            key: Cache key.
            value: Cache value.
            ttl: Time-to-live in seconds (overrides default).
        """
        if ttl is not None:
            expiry: float = self._now() + ttl
        elif self.default_ttl is not None:
            expiry = self._now() + self.default_ttl
        else:
            expiry = 0.0  # no expiry
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, expiry)
            while len(self._store) > self.capacity:
                self._store.popitem(last=False)

    def invalidate(self, key: Any) -> None:
        """Remove a specific key from the cache."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        with self._lock:
            return len(self._store)

    def memoize(self, ttl: Optional[float] = None, capacity: Optional[int] = None):
        """Decorator that caches function results.

        Args:
            ttl: Time-to-live in seconds (default uses cache default_ttl).
            capacity: Maximum entries for this function (default uses cache capacity).

        Examples:
            >>> cache = LRUCache(capacity=32, ttl=5)
            >>> @cache.memoize(ttl=10)
            ... def slow_add(a, b):
            ...     return a + b
        """
        cap = capacity or self.capacity

        def decorator(func: Callable) -> Callable:
            local_cache = LRUCache(capacity=cap, ttl=ttl or self.default_ttl)

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                key = (args, tuple(sorted(kwargs.items())))
                result = local_cache.get(key)
                if result is not None:
                    return result
                result = func(*args, **kwargs)
                local_cache.put(key, result)
                return result

            wrapper.cache = local_cache  # type: ignore[attr-defined]
            return wrapper

        return decorator


# ---------------------------------------------------------------------------
# Facade wrapper
# ---------------------------------------------------------------------------

class AxiomCache:
    """A thread-safe LRU cache with TTL support.

    Accessible via ``Axiom.cache``.

    Examples:
        >>> Axiom.cache.put("key", 42, ttl=10)
        >>> Axiom.cache.get("key")
        42
    """

    def __init__(self, capacity: int = 128, ttl: Optional[float] = None):
        self._cache = LRUCache(capacity=capacity, ttl=ttl)

    def get(self, key: Any) -> Optional[Any]:
        return self._cache.get(key)

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        self._cache.put(key, value, ttl=ttl)

    def invalidate(self, key: Any) -> None:
        self._cache.invalidate(key)

    def clear(self) -> None:
        self._cache.clear()

    @property
    def size(self) -> int:
        return self._cache.size

    def memoize(self, ttl: Optional[float] = None, capacity: Optional[int] = None):
        return self._cache.memoize(ttl=ttl, capacity=capacity)

    @property
    def _lru(self) -> LRUCache:
        return self._cache

    def help(self) -> str:
        return (
            "=== AxiomPy Cache Help ===\n"
            "\n"
            "Axiom.cache is a thread-safe LRU cache with optional TTL.\n"
            "\n"
            "Basic usage:\n"
            "  Axiom.cache.put('mykey', value, ttl=10.0)   # store for 10s\n"
            "  val = Axiom.cache.get('mykey')               # retrieve\n"
            "  Axiom.cache.invalidate('mykey')               # remove key\n"
            "  Axiom.cache.clear()                           # clear all\n"
            "  n = Axiom.cache.size                          # entry count\n"
            "\n"
            "Decorator (memoization):\n"
            "  @Axiom.cache.memoize(ttl=30)\n"
            "  def expensive_fn(x):\n"
            "      return x ** 2\n"
        )
