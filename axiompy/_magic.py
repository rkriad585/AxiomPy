"""Magical functions: decorators, utilities, and functional helpers.

Provides ``@timer``, ``@magic``, ``pipe``, ``compose`` accessible via
``Axiom.magic``.

Examples:
    >>> @Axiom.magic.timer
    ... def slow_add(a, b):
    ...     return a + b
    >>> slow_add(1, 2)  # prints timing info
    3
"""

import functools
import time
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Timer decorator
# ---------------------------------------------------------------------------

def timer(func: Optional[Callable] = None, *, unit: str = "ms") -> Callable:
    """Decorator that measures and prints function execution time.

    Args:
        func: Function to decorate (if used without arguments).
        unit: Time unit — "s", "ms" (default), or "us".

    Examples:
        >>> @timer
        ... def work():
        ...     return sum(range(1000))
        >>> work()
    """

    def _decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = f(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            if unit == "us":
                suffix = "us"
                display = elapsed * 1_000_000
            elif unit == "s":
                suffix = "s"
                display = elapsed
            else:
                suffix = "ms"
                display = elapsed * 1000
            print(f"# {f.__name__} took {display:.3f} {suffix}")
            return result

        return wrapper

    return _decorator(func) if func is not None else _decorator


# ---------------------------------------------------------------------------
# Magic decorator (auto-enhance functions)
# ---------------------------------------------------------------------------

def magic(
    func: Optional[Callable] = None,
    *,
    memoize: bool = False,
    time_it: bool = False,
    ttl: Optional[float] = None,
) -> Callable:
    """Transform a function with optional caching and/or timing.

    Args:
        func: Function to wrap.
        memoize: If True, cache results (uses internal LRU cache).
        time_it: If True, print execution time.
        ttl: Time-to-live in seconds for cache entries.

    Examples:
        >>> @Axiom.magic.magic(memoize=True, time_it=True)
        ... def expensive(n):
        ...     return sum(range(n))
        >>> expensive(1000000)   # computed & timed
        >>> expensive(1000000)   # cached
    """

    def _decorator(f: Callable) -> Callable:
        from ._cache import LRUCache

        local_cache = LRUCache(capacity=32, ttl=ttl) if memoize else None

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            cache_key = (args, tuple(sorted(kwargs.items())))
            if local_cache is not None:
                cached = local_cache.get(cache_key)
                if cached is not None:
                    return cached
            t0 = time.perf_counter() if time_it else None
            result = f(*args, **kwargs)
            if time_it and t0 is not None:
                elapsed = time.perf_counter() - t0
                print(f"# {f.__name__} took {elapsed*1000:.3f} ms")
            if local_cache is not None:
                local_cache.put(cache_key, result)
            return result

        wrapper._magic_cache = local_cache  # type: ignore[attr-defined]
        return wrapper

    return _decorator(func) if func is not None else _decorator


# ---------------------------------------------------------------------------
# Pipe / compose
# ---------------------------------------------------------------------------

def pipe(value: Any, *functions: Callable) -> Any:
    """Feed *value* through a pipeline of functions.

    Examples:
        >>> pipe(5, lambda x: x * 2, lambda x: x + 1)
        11
    """
    for fn in functions:
        value = fn(value)
    return value


def compose(*functions: Callable) -> Callable:
    """Return the right-to-left composition of *functions*.

    ``compose(f, g)(x) == f(g(x))``

    Examples:
        >>> add1 = lambda x: x + 1
        >>> double = lambda x: x * 2
        >>> compose(add1, double)(5)
        11
    """
    functions = tuple(reversed(functions))

    def composed(value: Any) -> Any:
        for fn in functions:
            value = fn(value)
        return value

    return composed


# ---------------------------------------------------------------------------
# Number-theoretic novelty functions
# ---------------------------------------------------------------------------

def digit_sum(n: int) -> int:
    """Sum of digits of *n*."""
    return sum(int(d) for d in str(abs(n)))


def digital_root(n: int) -> int:
    """Repeated digit sum until single digit."""
    while n >= 10:
        n = digit_sum(n)
    return n


def is_palindrome(n: int) -> bool:
    """Check if *n* is a numeric palindrome."""
    s = str(abs(n))
    return s == s[::-1]


def reverse_number(n: int) -> int:
    """Reverse the digits of *n*."""
    return int(str(abs(n))[::-1]) * (1 if n >= 0 else -1)


def collatz(n: int) -> list[int]:
    """Return the Collatz sequence starting from *n*."""
    seq = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        seq.append(n)
    return seq


def happy_numbers(limit: int) -> list[int]:
    """Return all happy numbers up to *limit*."""
    def _is_happy(num: int) -> bool:
        seen = set()
        while num != 1 and num not in seen:
            seen.add(num)
            num = sum(int(d) ** 2 for d in str(num))
        return num == 1
    return [i for i in range(1, limit + 1) if _is_happy(i)]


def armstrong_number(n: int) -> bool:
    """Check if *n* is an Armstrong number (sum of digits^digit_count)."""
    s = str(n)
    k = len(s)
    return sum(int(d) ** k for d in s) == n


def perfect_number(n: int) -> bool:
    """Check if *n* is a perfect number (sum of proper divisors equals *n*)."""
    if n < 2:
        return False
    total = 1
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            total += i
            if i != n // i:
                total += n // i
    return total == n


def friendly_numbers(a: int, b: int) -> bool:
    """Check if *a* and *b* form a friendly pair (same abundancy index)."""
    def divisor_sum(n: int) -> int:
        if n < 2:
            return 0
        s = 1
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                s += i
                if i != n // i:
                    s += n // i
        return s
    return divisor_sum(a) / a == divisor_sum(b) / b


def visualize_number(n: int) -> str:
    """ASCII visual summary of number properties."""
    lines = [f"Number: {n}"]
    lines.append(f"  Digit sum: {digit_sum(n)}")
    lines.append(f"  Digital root: {digital_root(n)}")
    lines.append(f"  Palindrome: {'yes' if is_palindrome(n) else 'no'}")
    lines.append(f"  Reversed: {reverse_number(n)}")
    lines.append(f"  Armstrong: {'yes' if armstrong_number(n) else 'no'}")
    lines.append(f"  Perfect: {'yes' if perfect_number(n) else 'no'}")
    if n % 2 == 0:
        lines.append("  Parity: even")
    else:
        lines.append("  Parity: odd")
    lines.append(f"  Collatz length: {len(collatz(n))}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Facade wrapper
# ---------------------------------------------------------------------------

class AxiomMagic:
    """Functional programming utilities, decorators, and number-theoretic functions.

    Accessible via ``Axiom.magic``.

    Examples:
        >>> @Axiom.magic.timer
        ... def work():
        ...     pass
        >>> Axiom.magic.pipe(5, lambda x: x * 2)
        10
        >>> Axiom.magic.digit_sum(123)
        6
    """

    timer = staticmethod(timer)
    magic = staticmethod(magic)
    pipe = staticmethod(pipe)
    compose = staticmethod(compose)
    digit_sum = staticmethod(digit_sum)
    digital_root = staticmethod(digital_root)
    is_palindrome = staticmethod(is_palindrome)
    reverse_number = staticmethod(reverse_number)
    collatz = staticmethod(collatz)
    happy_numbers = staticmethod(happy_numbers)
    armstrong_number = staticmethod(armstrong_number)
    perfect_number = staticmethod(perfect_number)
    friendly_numbers = staticmethod(friendly_numbers)
    visualize_number = staticmethod(visualize_number)

    def help(self) -> str:
        return (
            "=== AxiomPy Magic Help ===\n"
            "\n"
            "Decorators:\n"
            "  @Axiom.magic.timer          - time function execution\n"
            "  @Axiom.magic.timer(unit='s') - time in seconds\n"
            "  @Axiom.magic.magic(memoize=True, time_it=True)  - cache + time\n"
            "\n"
            "Functional utilities:\n"
            "  Axiom.magic.pipe(value, fn1, fn2, ...)    - pipeline\n"
            "  Axiom.magic.compose(f, g, ...)(value)     - composition\n"
            "\n"
            "Number fun:\n"
            "  Axiom.magic.digit_sum(n)       - sum of digits\n"
            "  Axiom.magic.digital_root(n)    - repeated digit sum\n"
            "  Axiom.magic.is_palindrome(n)   - palindrome check\n"
            "  Axiom.magic.reverse_number(n)  - reverse digits\n"
            "  Axiom.magic.collatz(n)         - Collatz sequence\n"
            "  Axiom.magic.happy_numbers(lim) - list of happy numbers\n"
            "  Axiom.magic.armstrong_number(n)- Armstrong check\n"
            "  Axiom.magic.perfect_number(n)  - perfect number check\n"
            "  Axiom.magic.friendly_numbers(a,b) - friendly pair\n"
            "  Axiom.magic.visualize_number(n)- ASCII summary\n"
        )
