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


def sieve_of_eratosthenes(n: int) -> list[int]:
    """Return all prime numbers up to *n* using the Sieve of Eratosthenes.

    Examples:
        >>> sieve_of_eratosthenes(20)
        [2, 3, 5, 7, 11, 13, 17, 19]
    """
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]


def kaprekar_routine(n: int) -> list[int]:
    """Return the Kaprekar routine sequence for a 4-digit number.

    Kaprekar's routine repeatedly applies: sort digits ascending and descending,
    subtract smaller from larger. Most 4-digit numbers converge to 6174.

    Examples:
        >>> kaprekar_routine(3524)
        [3524, 3087, 8352, 6174]
    """
    n = int(n)
    seq = [n]
    while n != 6174 and n != 0:
        s = str(n).zfill(4)
        asc = int("".join(sorted(s)))
        desc = int("".join(sorted(s, reverse=True)))
        n = desc - asc
        seq.append(n)
    return seq


def look_and_say(n: int) -> list[int]:
    """Return the first *n* terms of the look-and-say sequence.

    The sequence: 1, 11, 21, 1211, 111221, ... where each term
    describes the previous term's digit groups.

    Examples:
        >>> look_and_say(5)
        [1, 11, 21, 1211, 111221]
    """
    seq = [1]
    for _ in range(1, n):
        prev = str(seq[-1])
        result = []
        count = 1
        for i in range(1, len(prev)):
            if prev[i] == prev[i - 1]:
                count += 1
            else:
                result.append(str(count) + prev[i - 1])
                count = 1
        result.append(str(count) + prev[-1])
        seq.append(int("".join(result)))
    return seq


def ulam_spiral(n: int) -> list[tuple[int, int, int]]:
    """Return coordinates and values for the Ulam spiral up to *n*.

    Each element is ``(x, y, value)`` where ``value`` is the number placed
    at that coordinate in the spiral.

    Examples:
        >>> ulam_spiral(10)  # doctest: +SKIP
    """
    if n < 1:
        return []
    result = [(0, 0, 1)]
    x, y = 0, 0
    step = 1
    val = 2
    while val <= n:
        for dx, dy, steps in [(1, 0, step), (0, -1, step), (-1, 0, step + 1), (0, 1, step + 1)]:
            for _ in range(steps):
                if val > n:
                    break
                x += dx
                y += dy
                result.append((x, y, val))
                val += 1
            if val > n:
                break
        step += 2
    return result


def narcissistic_numbers(limit: int) -> list[int]:
    """Return all narcissistic (Armstrong) numbers up to *limit*.

    A narcissistic number equals the sum of its digits raised to the
    power of the number of digits.

    Examples:
        >>> narcissistic_numbers(200)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 153]
    """
    return [i for i in range(1, limit + 1) if armstrong_number(i)]


def _divisors(n: int) -> set[int]:
    """Return the set of proper divisors of *n* (excluding *n*)."""
    if n < 2:
        return set()
    divs = {1}
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return divs


def smith_numbers(limit: int) -> list[int]:
    """Return all Smith numbers up to *limit*.

    A Smith number's digit sum equals the digit sum of its prime factors.

    Examples:
        >>> smith_numbers(30)
        [4, 22, 27]
    """
    result = []
    for num in range(2, limit + 1):
        if _is_prime(num):
            continue
        sd = digit_sum(num)
        pf = []
        n = num
        for p in sieve_of_eratosthenes(num):
            while n % p == 0:
                pf.append(p)
                n //= p
        if n > 1:
            pf.append(n)
        pf_sum = sum(digit_sum(f) for f in pf)
        if sd == pf_sum:
            result.append(num)
    return result


def _is_prime(n: int) -> bool:
    """Check if n is prime."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def emirp_numbers(limit: int) -> list[int]:
    """Return all emirps (primes that stay prime when reversed) up to *limit*.

    Palindromic primes are excluded.

    Examples:
        >>> emirp_numbers(50)
        [13, 17, 31, 37]
    """
    result = []
    for n in range(10, limit + 1):
        rev = int(str(n)[::-1])
        if n != rev and _is_prime(n) and _is_prime(rev):
            result.append(n)
    return result


def goldbach_conjecture(n: int) -> list[tuple[int, int]]:
    """Return all Goldbach partitions of an even number *n*.

    Each partition is a pair of primes ``(p, q)`` with ``p + q = n``.

    Examples:
        >>> goldbach_conjecture(20)
        [(3, 17), (7, 13)]
    """
    if n % 2 != 0 or n < 4:
        return []
    primes = [p for p in sieve_of_eratosthenes(n) if p <= n // 2]
    result = []
    for p in primes:
        q = n - p
        if _is_prime(q):
            result.append((p, q))
    return result


def twin_primes(limit: int) -> list[tuple[int, int]]:
    """Return all twin prime pairs up to *limit*.

    Twin primes are pairs ``(p, p+2)`` where both are prime.

    Examples:
        >>> twin_primes(30)
        [(3, 5), (5, 7), (11, 13), (17, 19)]
    """
    primes = sieve_of_eratosthenes(limit)
    prime_set = set(primes)
    return [(p, p + 2) for p in primes if p + 2 in prime_set]


def circular_primes(limit: int) -> list[int]:
    """Return all circular primes up to *limit*.

    A circular prime stays prime under all digit rotations.

    Examples:
        >>> circular_primes(100)
        [2, 3, 5, 7, 11, 13, 17, 31, 37, 71, 73, 79, 97]
    """
    result = []
    for n in range(2, limit + 1):
        s = str(n)
        if not all(_is_prime(int(s[i:] + s[:i])) for i in range(len(s))):
            continue
        result.append(n)
    return result


def number_to_words(n: int) -> str:
    """Spell out a number in English words.

    Args:
        n: Integer to spell out (supports up to 999,999,999,999).

    Returns:
        English word representation.

    Examples:
        >>> number_to_words(42)
        'forty-two'
        >>> number_to_words(2024)
        'two thousand twenty-four'
    """
    if n == 0:
        return "zero"

    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    scales = ["", "thousand", "million", "billion"]

    def _under_1000(num: int) -> str:
        parts = []
        h = num // 100
        if h > 0:
            parts.append(ones[h] + " hundred")
        r = num % 100
        if r == 0:
            return " ".join(parts)
        if h > 0:
            parts.append("and")
        if r < 20:
            parts.append(ones[r])
        else:
            parts.append(tens[r // 10] + ("-" + ones[r % 10] if r % 10 else ""))
        return " ".join(parts)

    if n < 0:
        return "negative " + number_to_words(-n)

    result = []
    scale_idx = 0
    while n > 0:
        chunk = n % 1000
        if chunk > 0:
            chunk_words = _under_1000(chunk)
            if scales[scale_idx]:
                chunk_words += " " + scales[scale_idx]
            result.insert(0, chunk_words)
        n //= 1000
        scale_idx += 1

    return " ".join(result)


def roman_numeral(n: int) -> str:
    """Convert an integer to Roman numerals.

    Args:
        n: Integer between 1 and 3999.

    Returns:
        Roman numeral string.

    Examples:
        >>> roman_numeral(42)
        'XLII'
        >>> roman_numeral(2024)
        'MMXXIV'
    """
    if n < 1 or n > 3999:
        raise ValueError("Roman numeral conversion supports 1–3999")
    values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    symbols = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    result = []
    for v, s in zip(values, symbols):
        while n >= v:
            result.append(s)
            n -= v
    return "".join(result)


def factorial_digit_sum(n: int) -> int:
    """Return the sum of digits of *n*! (n factorial).

    Args:
        n: Non-negative integer.

    Examples:
        >>> factorial_digit_sum(10)
        27
    """
    fact = 1
    for i in range(2, n + 1):
        fact *= i
    return sum(int(d) for d in str(fact))


def fibonacci_spiral(n: int) -> list[int]:
    """Return the first *n* Fibonacci numbers.

    Args:
        n: How many Fibonacci numbers to generate.

    Examples:
        >>> fibonacci_spiral(10)
        [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    """
    if n < 1:
        return []
    seq = [1]
    if n == 1:
        return seq
    seq.append(1)
    for _ in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq


def automorphic_numbers(limit: int) -> list[int]:
    """Return all automorphic numbers up to *limit*.

    An automorphic number's square ends with the number itself.

    Examples:
        >>> automorphic_numbers(100)
        [1, 5, 6, 25, 76]
    """
    return [i for i in range(1, limit + 1) if str(i * i).endswith(str(i))]


def harshad_numbers(limit: int) -> list[int]:
    """Return all Harshad numbers up to *limit*.

    A Harshad (or Niven) number is divisible by the sum of its digits.

    Examples:
        >>> harshad_numbers(30)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 18, 20, 21, 24, 27]
    """
    return [i for i in range(1, limit + 1) if i % digit_sum(i) == 0]


def visualize_number(n: int) -> str:
    """ASCII visual summary of number properties.

    Shows digit sum, digital root, palindrome status, Armstrong check,
    perfect number check, automorphic check, Harshad check, and more.

    Examples:
        >>> print(visualize_number(8128))  # doctest: +SKIP
    """
    lines = [f"Number: {n}"]
    lines.append(f"  Digit sum:      {digit_sum(n)}")
    lines.append(f"  Digital root:   {digital_root(n)}")
    lines.append(f"  Palindrome:     {'yes' if is_palindrome(n) else 'no'}")
    lines.append(f"  Reversed:       {reverse_number(n)}")
    lines.append(f"  Armstrong:      {'yes' if armstrong_number(n) else 'no'}")
    lines.append(f"  Perfect:        {'yes' if perfect_number(n) else 'no'}")
    lines.append(f"  Automorphic:    {'yes' if str(n * n).endswith(str(n)) else 'no'}")
    lines.append(f"  Harshad:        {'yes' if n % digit_sum(n) == 0 else 'no'}")
    lines.append(f"  Roman:          {roman_numeral(n) if 1 <= n <= 3999 else 'N/A'}")
    lines.append(f"  Words:          {number_to_words(n)}")
    lines.append(f"  Binary:         {bin(n)}")
    lines.append(f"  Hex:            {hex(n)}")
    lines.append(f"  Collatz length: {len(collatz(n))}")
    lines.append(f"  Goldbach pairs: {len(goldbach_conjecture(n))}")
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
    sieve_of_eratosthenes = staticmethod(sieve_of_eratosthenes)
    kaprekar_routine = staticmethod(kaprekar_routine)
    look_and_say = staticmethod(look_and_say)
    ulam_spiral = staticmethod(ulam_spiral)
    narcissistic_numbers = staticmethod(narcissistic_numbers)
    smith_numbers = staticmethod(smith_numbers)
    emirp_numbers = staticmethod(emirp_numbers)
    goldbach_conjecture = staticmethod(goldbach_conjecture)
    twin_primes = staticmethod(twin_primes)
    circular_primes = staticmethod(circular_primes)
    number_to_words = staticmethod(number_to_words)
    roman_numeral = staticmethod(roman_numeral)
    factorial_digit_sum = staticmethod(factorial_digit_sum)
    fibonacci_spiral = staticmethod(fibonacci_spiral)
    automorphic_numbers = staticmethod(automorphic_numbers)
    harshad_numbers = staticmethod(harshad_numbers)
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
            "Number theory:\n"
            "  Axiom.magic.sieve_of_eratosthenes(n) - primes up to n\n"
            "  Axiom.magic.digit_sum(n)             - sum of digits\n"
            "  Axiom.magic.digital_root(n)          - repeated digit sum\n"
            "  Axiom.magic.is_palindrome(n)         - palindrome check\n"
            "  Axiom.magic.reverse_number(n)        - reverse digits\n"
            "  Axiom.magic.collatz(n)               - Collatz sequence\n"
            "  Axiom.magic.happy_numbers(lim)       - happy numbers\n"
            "  Axiom.magic.armstrong_number(n)      - Armstrong check\n"
            "  Axiom.magic.perfect_number(n)        - perfect number check\n"
            "  Axiom.magic.friendly_numbers(a,b)    - friendly pair\n"
            "  Axiom.magic.kaprekar_routine(n)      - Kaprekar routine\n"
            "  Axiom.magic.look_and_say(n)          - look-and-say seq\n"
            "  Axiom.magic.ulam_spiral(n)           - Ulam spiral coords\n"
            "  Axiom.magic.narcissistic_numbers(lim)- narcissistic nums\n"
            "  Axiom.magic.smith_numbers(lim)       - Smith numbers\n"
            "  Axiom.magic.emirp_numbers(lim)       - emirps\n"
            "  Axiom.magic.goldbach_conjecture(n)   - Goldbach partitions\n"
            "  Axiom.magic.twin_primes(lim)         - twin prime pairs\n"
            "  Axiom.magic.circular_primes(lim)     - circular primes\n"
            "  Axiom.magic.number_to_words(n)       - spell out number\n"
            "  Axiom.magic.roman_numeral(n)         - Roman numerals\n"
            "  Axiom.magic.factorial_digit_sum(n)   - digit sum of n!\n"
            "  Axiom.magic.fibonacci_spiral(n)      - Fibonacci numbers\n"
            "  Axiom.magic.automorphic_numbers(lim) - automorphic nums\n"
            "  Axiom.magic.harshad_numbers(lim)     - Harshad numbers\n"
            "  Axiom.magic.visualize_number(n)      - ASCII summary\n"
        )
