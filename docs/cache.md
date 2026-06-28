# Cache (`Axiom.cache`)

Thread-safe LRU (Least Recently Used) cache with optional TTL (time-to-live), hit/miss tracking, and save/load persistence.

## Quick Start

```python
from axiompy import Axiom

Axiom.cache.put("mykey", [1, 2, 3], ttl=10.0)
val = Axiom.cache.get("mykey")  # [1, 2, 3]
Axiom.cache.size                # 1
Axiom.cache.invalidate("mykey")
Axiom.cache.clear()
```

## Memoization Decorator

```python
@Axiom.cache.memoize(ttl=30)
def expensive_fn(x):
    return sum(range(x))
```

The decorator creates a per-function LRU cache. Results are cached by positional and keyword arguments.

### Async memoization

For `async def` functions, use `async_memoize`:

```python
@Axiom.cache.async_memoize(ttl=60)
async def fetch_data(url):
    # ... async I/O ...
    return result

data = await fetch_data("https://example.com/api")
```

## Cache Statistics

Track how well your cache is performing:

```python
Axiom.cache.hits       # number of successful lookups
Axiom.cache.misses     # number of failed lookups
Axiom.cache.hit_ratio  # hits / (hits + misses) as float
Axiom.cache.stats()    # dict with all stats
Axiom.cache.info()     # human-readable summary string
```

## Persistence

Save and restore cache state:

```python
Axiom.cache.save("my_cache.json")
Axiom.cache.load("my_cache.json")  # replaces current cache contents
```

## API

| Method / Property | Description |
|---|---|
| `put(key, value, ttl=None)` | Store a value |
| `get(key)` | Retrieve a value (`None` if missing/expired) |
| `invalidate(key)` | Remove a key |
| `clear()` | Clear all entries |
| `size` | Current entry count |
| `hits` | Number of cache hits |
| `misses` | Number of cache misses |
| `hit_ratio` | Hit / (hits + misses) |
| `stats()` | Dict of all statistics |
| `info()` | Human-readable summary |
| `save(path)` | Serialize cache to JSON file |
| `load(path)` | Load cache from JSON file |
| `memoize(ttl=None, capacity=None)` | Decorator for sync functions |
| `async_memoize(ttl=None, capacity=None)` | Decorator for async functions |
