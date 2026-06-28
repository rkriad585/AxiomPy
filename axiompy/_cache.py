"""Least-Recently-Used (LRU) cache with optional TTL, thread safety, async support, and statistics.

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

import asyncio
import json
import os
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
        self._hits = 0
        self._misses = 0

    def _now(self) -> float:
        return time.monotonic()

    def _is_expired(self, entry: tuple[Any, float]) -> bool:
        _, expiry = entry
        return expiry > 0 and self._now() > expiry

    def get(self, key: Any) -> Optional[Any]:
        """Get a value by *key*. Returns None if missing or expired.

        Examples:
            >>> c = LRUCache()
            >>> c.put("name", "Alice")
            >>> c.get("name")
            'Alice'
            >>> c.get("missing")  # returns None
        """
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None
            entry = self._store[key]
            if self._is_expired(entry):
                del self._store[key]
                self._misses += 1
                return None
            self._hits += 1
            self._store.move_to_end(key)
            return entry[0]

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Insert or update a key-value pair.

        Args:
            key: Cache key.
            value: Cache value.
            ttl: Time-to-live in seconds (overrides default).
                Pass ``0`` for no expiry.

        Examples:
            >>> c = LRUCache()
            >>> c.put("answer", 42)
            >>> c.put("temp", "expires", ttl=5.0)
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
        """Remove a specific key from the cache.

        Examples:
            >>> c = LRUCache()
            >>> c.put("key", "val")
            >>> c.invalidate("key")
            >>> c.get("key") is None
            True
        """
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Clear the entire cache and reset statistics."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    @property
    def size(self) -> int:
        """Current number of entries in the cache.

        Examples:
            >>> c = LRUCache(capacity=3)
            >>> c.put("a", 1); c.put("b", 2)
            >>> c.size
            2
        """
        with self._lock:
            return len(self._store)

    @property
    def hits(self) -> int:
        """Number of successful cache lookups.

        Examples:
            >>> c = LRUCache()
            >>> c.put("x", 1); c.get("x")
            1
            >>> c.hits
            1
        """
        return self._hits

    @property
    def misses(self) -> int:
        """Number of failed cache lookups (missing or expired).

        Examples:
            >>> c = LRUCache()
            >>> c.get("missing") is None
            True
            >>> c.misses
            1
        """
        return self._misses

    @property
    def hit_ratio(self) -> float:
        """Fraction of lookups that were hits (0.0 to 1.0).

        Returns 0.0 if no lookups have been made.

        Examples:
            >>> c = LRUCache()
            >>> c.put("a", 1); c.get("a"); c.get("b")
            >>> c.hit_ratio
            0.5
        """
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        """Return a dict of cache statistics.

        Returns:
            dict with keys: ``size``, ``capacity``, ``hits``, ``misses``,
            ``hit_ratio``, ``default_ttl``.

        Examples:
            >>> c = LRUCache(capacity=64, ttl=30)
            >>> c.put("a", 1)
            >>> "hit_ratio" in c.stats()
            True
        """
        return {
            "size": self.size,
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hit_ratio,
            "default_ttl": self.default_ttl,
        }

    def info(self) -> str:
        """Return a formatted string with cache statistics.

        Examples:
            >>> c = LRUCache()
            >>> print(c.info())  # doctest: +SKIP
        """
        s = self.stats()
        return (
            f"Cache: {s['size']}/{s['capacity']} entries\n"
            f"  Hits:   {s['hits']}\n"
            f"  Misses: {s['misses']}\n"
            f"  Ratio:  {s['hit_ratio']:.1%}\n"
            f"  TTL:    {s['default_ttl']}"
        )

    def save(self, path: str) -> None:
        """Persist cache contents to a JSON file.

        Only entries that have not expired are saved.

        Args:
            path: File path to write to.

        Examples:
            >>> c = LRUCache()
            >>> c.put("a", 1)
            >>> c.save("/tmp/cache.json")  # doctest: +SKIP
        """
        data = {
            "capacity": self.capacity,
            "default_ttl": self.default_ttl,
            "entries": [],
        }
        now = self._now()
        with self._lock:
            for key, (value, expiry) in self._store.items():
                if expiry > 0 and now > expiry:
                    continue
                remaining = expiry - now if expiry > 0 else None
                data["entries"].append({
                    "key": key,
                    "value": value,
                    "ttl_remaining": remaining,
                })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load cache contents from a JSON file saved with ``save()``.

        Args:
            path: File path to read from.

        Examples:
            >>> c = LRUCache()
            >>> c.load("/tmp/cache.json")  # doctest: +SKIP
        """
        with open(path) as f:
            data = json.load(f)
        self.capacity = data.get("capacity", self.capacity)
        self.default_ttl = data.get("default_ttl", self.default_ttl)
        now = self._now()
        with self._lock:
            self._store.clear()
            for entry in data.get("entries", []):
                ttl = entry.get("ttl_remaining")
                expiry = (now + ttl) if ttl is not None else 0.0
                self._store[entry["key"]] = (entry["value"], expiry)

    def memoize(self, ttl: Optional[float] = None, capacity: Optional[int] = None):
        """Decorator that caches function results (sync functions only).

        For async functions, use :meth:`async_memoize` instead.

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

    def async_memoize(self, ttl: Optional[float] = None, capacity: Optional[int] = None):
        """Decorator that caches results of ``async def`` functions.

        Args:
            ttl: Time-to-live in seconds (default uses cache default_ttl).
            capacity: Maximum entries for this function (default uses cache capacity).

        Examples:
            >>> cache = LRUCache(capacity=16, ttl=30)
            >>> @cache.async_memoize(ttl=10)
            ... async def fetch_data(url):
            ...     return {"url": url}
        """
        cap = capacity or self.capacity

        def decorator(func: Callable) -> Callable:
            local_cache = LRUCache(capacity=cap, ttl=ttl or self.default_ttl)

            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                key = (args, tuple(sorted(kwargs.items())))
                result = local_cache.get(key)
                if result is not None:
                    return result
                result = await func(*args, **kwargs)
                local_cache.put(key, result)
                return result

            wrapper.cache = local_cache  # type: ignore[attr-defined]
            return wrapper

        return decorator


# ---------------------------------------------------------------------------
# Facade wrapper
# ---------------------------------------------------------------------------

class AxiomCache:
    """A thread-safe LRU cache with TTL support, statistics, and async memoization.

    Accessible via ``Axiom.cache``.

    Examples:
        >>> Axiom.cache.put("key", 42, ttl=10)
        >>> Axiom.cache.get("key")
        42
        >>> Axiom.cache.size
        1
        >>> Axiom.cache.hit_ratio
        1.0
    """

    def __init__(self, capacity: int = 128, ttl: Optional[float] = None):
        self._cache = LRUCache(capacity=capacity, ttl=ttl)

    def get(self, key: Any) -> Optional[Any]:
        """Look up a cached value.

        Args:
            key: Cache key.

        Returns:
            The cached value, or None if not found or expired.
        """
        return self._cache.get(key)

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """Store a value in the cache.

        Args:
            key: Cache key.
            value: Value to store.
            ttl: Time-to-live in seconds (optional).
        """
        self._cache.put(key, value, ttl=ttl)

    def invalidate(self, key: Any) -> None:
        """Remove a specific key from the cache."""
        self._cache.invalidate(key)

    def clear(self) -> None:
        """Clear all entries and reset statistics."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return self._cache.size

    @property
    def hits(self) -> int:
        """Number of successful cache lookups (hit count)."""
        return self._cache.hits

    @property
    def misses(self) -> int:
        """Number of failed cache lookups (miss count)."""
        return self._cache.misses

    @property
    def hit_ratio(self) -> float:
        """Fraction of lookups that found a valid entry (0.0 to 1.0)."""
        return self._cache.hit_ratio

    def stats(self) -> dict:
        """Return cache statistics as a dictionary.

        Returns:
            dict with keys: ``size``, ``capacity``, ``hits``, ``misses``,
            ``hit_ratio``, ``default_ttl``.
        """
        return self._cache.stats()

    def info(self) -> str:
        """Return a formatted string with cache statistics."""
        return self._cache.info()

    def save(self, path: str) -> None:
        """Save cache contents to a JSON file."""
        self._cache.save(path)

    def load(self, path: str) -> None:
        """Load cache contents from a JSON file."""
        self._cache.load(path)

    def memoize(self, ttl: Optional[float] = None, capacity: Optional[int] = None):
        """Decorator to cache results of sync functions.

        Args:
            ttl: Time-to-live in seconds.
            capacity: Max entries for this function's cache.

        Examples:
            >>> @Axiom.cache.memoize(ttl=30)
            ... def compute(x):
            ...     return x * x
        """
        return self._cache.memoize(ttl=ttl, capacity=capacity)

    def async_memoize(self, ttl: Optional[float] = None, capacity: Optional[int] = None):
        """Decorator to cache results of async functions.

        Args:
            ttl: Time-to-live in seconds.
            capacity: Max entries for this function's cache.

        Examples:
            >>> @Axiom.cache.async_memoize(ttl=30)
            ... async def fetch(url):
            ...     return "data"
        """
        return self._cache.async_memoize(ttl=ttl, capacity=capacity)

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
            "\n"
            "Info & stats:\n"
            "  n = Axiom.cache.size                          # entry count\n"
            "  n = Axiom.cache.hits                          # successful lookups\n"
            "  n = Axiom.cache.misses                        # failed lookups\n"
            "  r = Axiom.cache.hit_ratio                     # hit fraction\n"
            "  d = Axiom.cache.stats()                       # all stats as dict\n"
            "  Axiom.cache.info()                            # formatted stats\n"
            "\n"
            "Persistence:\n"
            "  Axiom.cache.save('cache.json')                # save to disk\n"
            "  Axiom.cache.load('cache.json')                # load from disk\n"
            "\n"
            "Decorators:\n"
            "  @Axiom.cache.memoize(ttl=30)                  # sync functions\n"
            "  @Axiom.cache.async_memoize(ttl=30)            # async functions\n"
        )
