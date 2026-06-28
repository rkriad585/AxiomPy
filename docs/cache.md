# Cache (`Axiom.cache`)

Thread-safe LRU (Least Recently Used) cache with optional TTL (time-to-live).

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

The decorator creates a per-function LRU cache. Results are cached by arguments.

## API

| Method | Description |
|--------|-------------|
| `put(key, value, ttl=None)` | Store a value |
| `get(key)` | Retrieve a value (None if missing/expired) |
| `invalidate(key)` | Remove a key |
| `clear()` | Clear all entries |
| `size` | Current entry count |
| `memoize(ttl=None, capacity=None)` | Decorator for function result caching |
