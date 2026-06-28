"""Beginner-friendly basic math operations.

Covers arithmetic, PEMDAS, fractions, decimals, percentages, linear equations,
exponents, square roots, geometry formulas, statistics, and probability.

Accessible via ``Axiom.math``.

Examples:
    >>> Axiom.math.add(5, 3)
    8
    >>> Axiom.math.evaluate("2 + 3 * 4")
    14
    >>> Axiom.math.Fraction(1, 2) + Axiom.math.Fraction(1, 4)
    Fraction(3, 4)
"""

import math
import random
from typing import Union


# ---------------------------------------------------------------------------
# Prime utilities
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    """Check if a number is prime. Returns True/False.

    Examples:
        >>> is_prime(7)
        True
        >>> is_prime(4)
        False
    """
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def nth_prime(n: int) -> int:
    """Return the n-th prime number (1-indexed).

    Examples:
        >>> nth_prime(1)
        2
        >>> nth_prime(5)
        11
    """
    count = 0
    num = 1
    while count < n:
        num += 1
        if is_prime(num):
            count += 1
    return num


def primes_up_to(limit: int) -> list[int]:
    """Return a list of all primes up to (and including) limit.

    Examples:
        >>> primes_up_to(10)
        [2, 3, 5, 7]
    """
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:limit + 1:step] = [False] * ((limit - start) // step + 1)
    return [i for i, is_p in enumerate(sieve) if is_p]


# ---------------------------------------------------------------------------
# PEMDAS expression evaluator
# ---------------------------------------------------------------------------

class ExpressionError(Exception):
    """Raised when an expression cannot be parsed or evaluated."""


PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3}
OPERATORS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    "^": lambda a, b: a ** b,
}


def _tokenize(expr: str) -> list:
    tokens = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit() or ch == ".":
            start = i
            while i < len(expr) and (expr[i].isdigit() or expr[i] == "."):
                i += 1
            tokens.append(("num", float(expr[start:i])))
            continue
        if ch in OPERATORS:
            tokens.append(("op", ch))
            i += 1
            continue
        if ch == "(":
            tokens.append(("lp", ch))
            i += 1
            continue
        if ch == ")":
            tokens.append(("rp", ch))
            i += 1
            continue
        raise ExpressionError(f"Unexpected character: {ch}")
    return tokens


def _to_rpn(tokens: list) -> list:
    """Convert infix tokens to RPN (Shunting-yard)."""
    output = []
    ops = []
    for typ, val in tokens:
        if typ == "num":
            output.append(val)
        elif typ == "op":
            while (
                ops
                and ops[-1] != "("
                and PRECEDENCE.get(ops[-1], 0) >= PRECEDENCE.get(val, 0)
            ):
                output.append(ops.pop())
            ops.append(val)
        elif typ == "lp":
            ops.append(val)
        elif typ == "rp":
            while ops and ops[-1] != "(":
                output.append(ops.pop())
            if not ops or ops.pop() != "(":
                raise ExpressionError("Mismatched parentheses")
    output.extend(reversed(ops))
    return output


def evaluate(expression: str) -> float:
    """Evaluate a math expression following PEMDAS order of operations.

    Supports: ``+``, ``-``, ``*``, ``/``, ``^``, parentheses, and decimals.

    Examples:
        >>> evaluate("2 + 3 * 4")
        14
        >>> evaluate("(2 + 3) * 4")
        20
        >>> evaluate("3 ^ 2")
        9.0
    """
    tokens = _tokenize(expression)
    rpn = _to_rpn(tokens)
    stack = []
    for token in rpn:
        if isinstance(token, (int, float)):
            stack.append(token)
        elif token in OPERATORS:
            if len(stack) < 2:
                raise ExpressionError("Not enough operands")
            b = stack.pop()
            a = stack.pop()
            stack.append(OPERATORS[token](a, b))
        else:
            raise ExpressionError(f"Unknown token: {token}")
    if len(stack) != 1:
        raise ExpressionError("Invalid expression")
    return stack[0]


# ---------------------------------------------------------------------------
# Fraction
# ---------------------------------------------------------------------------

def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


class Fraction:
    """A fraction with numerator and denominator.

    Examples:
        >>> f = Fraction(1, 2)
        >>> g = Fraction(3, 4)
        >>> f + g
        Fraction(5, 4)
    """

    def __init__(self, numerator: int, denominator: int = 1):
        if denominator == 0:
            raise ZeroDivisionError("Fraction denominator cannot be zero")
        if denominator < 0:
            numerator = -numerator
            denominator = -denominator
        g = _gcd(abs(numerator), abs(denominator))
        self._num = numerator // g
        self._den = denominator // g

    @property
    def numerator(self) -> int:
        return self._num

    @property
    def denominator(self) -> int:
        return self._den

    @property
    def value(self) -> float:
        return self._num / self._den

    def __repr__(self) -> str:
        if self._den == 1:
            return f"Fraction({self._num})"
        return f"Fraction({self._num}, {self._den})"

    def __str__(self) -> str:
        if self._den == 1:
            return str(self._num)
        return f"{self._num}/{self._den}"

    def __add__(self, other):
        if isinstance(other, Fraction):
            return Fraction(
                self._num * other._den + other._num * self._den,
                self._den * other._den,
            )
        return Fraction(self._num + other * self._den, self._den)

    def __sub__(self, other):
        if isinstance(other, Fraction):
            return Fraction(
                self._num * other._den - other._num * self._den,
                self._den * other._den,
            )
        return Fraction(self._num - other * self._den, self._den)

    def __mul__(self, other):
        if isinstance(other, Fraction):
            return Fraction(self._num * other._num, self._den * other._den)
        return Fraction(self._num * other, self._den)

    def __truediv__(self, other):
        if isinstance(other, Fraction):
            return Fraction(self._num * other._den, self._den * other._num)
        return Fraction(self._num, self._den * other)

    def __eq__(self, other):
        if isinstance(other, Fraction):
            return self._num == other._num and self._den == other._den
        return self.value == other

    def __lt__(self, other):
        if isinstance(other, Fraction):
            return self._num * other._den < other._num * self._den
        return self.value < other

    def __le__(self, other):
        if isinstance(other, Fraction):
            return self._num * other._den <= other._num * self._den
        return self.value <= other

    def __gt__(self, other):
        if isinstance(other, Fraction):
            return self._num * other._den > other._num * self._den
        return self.value > other

    def __ge__(self, other):
        if isinstance(other, Fraction):
            return self._num * other._den >= other._num * self._den
        return self.value >= other

    def __float__(self) -> float:
        return self.value

    def __int__(self) -> int:
        return self._num // self._den

    def simplify(self):
        """Return a new Fraction in lowest terms (already done at init)."""
        return Fraction(self._num, self._den)


# ---------------------------------------------------------------------------
# Linear equations
# ---------------------------------------------------------------------------

def solve_linear(equation: str) -> float:
    """Solve a single-variable linear equation in the form "ax + b = c".

    The variable must be ``x``. Spaces optional.

    Examples:
        >>> solve_linear("x + 2 = 5")
        3.0
        >>> solve_linear("3x + 1 = 10")
        3.0
        >>> solve_linear("2x - 4 = 0")
        2.0
    """
    # Parse left and right sides
    expr = equation.replace(" ", "")
    left, right = expr.split("=")
    right_val = evaluate(right)

    # Build coefficient and constant from left side
    coeff = 0.0
    const = 0.0
    i = 0
    while i < len(left):
        if left[i] in "+-":
            sign = 1 if left[i] == "+" else -1
            i += 1
        else:
            sign = 1
        start = i
        while i < len(left) and left[i] not in "+-":
            i += 1
        term = left[start:i]
        if term == "":
            continue
        if "x" in term:
            num_part = term.replace("x", "")
            coeff += sign * (float(num_part) if num_part and num_part != "-" else (1.0 if num_part != "-" else -1.0))
        else:
            const += sign * float(term)

    if coeff == 0:
        raise ExpressionError("No variable 'x' found in equation")
    return (right_val - const) / coeff


# ---------------------------------------------------------------------------
# Geometry formulas
# ---------------------------------------------------------------------------

def rectangle_perimeter(length: float, width: float) -> float:
    """Perimeter of a rectangle: 2 * (length + width).

    Example:
        >>> rectangle_perimeter(4, 6)
        20.0
    """
    return 2 * (length + width)


def rectangle_area(length: float, width: float) -> float:
    """Area of a rectangle: length * width."""
    return length * width


def square_perimeter(side: float) -> float:
    """Perimeter of a square: 4 * side."""
    return 4 * side


def square_area(side: float) -> float:
    """Area of a square: side * side."""
    return side * side


def circle_circumference(radius: float) -> float:
    """Circumference of a circle: 2 * pi * radius."""
    return 2 * math.pi * radius


def circle_area(radius: float) -> float:
    """Area of a circle: pi * radius ** 2."""
    return math.pi * radius * radius


def triangle_area(base: float, height: float) -> float:
    """Area of a triangle: 0.5 * base * height."""
    return 0.5 * base * height


def pythagorean(a: float, b: float) -> float:
    """Return the hypotenuse c given legs a and b: c = sqrt(a^2 + b^2).

    Example:
        >>> pythagorean(3, 4)
        5.0
    """
    return (a * a + b * b) ** 0.5


def pythagorean_leg(c: float, a: float) -> float:
    """Return the missing leg b given hypotenuse c and leg a: b = sqrt(c^2 - a^2)."""
    return (c * c - a * a) ** 0.5


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def mean(numbers: list[float]) -> float:
    """Arithmetic mean (average). Sum divided by count.

    Example:
        >>> mean([2, 4, 6])
        4.0
    """
    return sum(numbers) / len(numbers)


def median(numbers: list[float]) -> float:
    """Middle value when numbers are sorted."""
    s = sorted(numbers)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def mode(numbers: list[float]) -> list[float]:
    """Most frequent value(s). Returns a list (handles multi-modal)."""
    from collections import Counter
    c = Counter(numbers)
    max_count = max(c.values())
    return [k for k, v in c.items() if v == max_count]


# ---------------------------------------------------------------------------
# Probability
# ---------------------------------------------------------------------------

def coin_flip() -> str:
    """Return ``'Heads'`` or ``'Tails'`` with equal probability."""
    return "Heads" if random.random() < 0.5 else "Tails"


def dice_roll(sides: int = 6) -> int:
    """Roll a dice with the given number of sides.

    Example:
        >>> dice_roll(6)  # returns 1-6
    """
    return random.randint(1, sides)


def factorial(n: int) -> int:
    """Compute n! (n factorial)."""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    return math.factorial(n)


def permutations(n: int, r: int) -> int:
    """Number of ways to choose r items from n where order matters: P(n, r)."""
    return math.perm(n, r)


def combinations(n: int, r: int) -> int:
    """Number of ways to choose r items from n where order does NOT matter: C(n, r)."""
    return math.comb(n, r)


# ---------------------------------------------------------------------------
# Speed / distance / time
# ---------------------------------------------------------------------------

def speed(distance: float, time: float) -> float:
    """Speed = distance / time.

    Example:
        >>> speed(60, 1)
        60.0
    """
    return distance / time


def distance_traveled(speed_val: float, time_val: float) -> float:
    """Distance = speed * time."""
    return speed_val * time_val


def travel_time(distance: float, speed_val: float) -> float:
    """Time = distance / speed."""
    return distance / speed_val


# ---------------------------------------------------------------------------
# Exponents and roots
# ---------------------------------------------------------------------------

def power(base: float, exp: float) -> float:
    """Raise base to the power of exp.

    Example:
        >>> power(3, 2)
        9.0
    """
    return base ** exp


def nth_root(value: float, n: float = 2) -> float:
    """Return the n-th root of value.

    Example:
        >>> nth_root(16, 2)
        4.0
        >>> nth_root(27, 3)
        3.0
    """
    return value ** (1.0 / n)


def sqrt(value: float) -> float:
    """Square root (shortcut for nth_root with n=2).

    Example:
        >>> sqrt(16)
        4.0
    """
    return value ** 0.5


# ---------------------------------------------------------------------------
# Percentages
# ---------------------------------------------------------------------------

def percent_to_decimal(pct: float) -> float:
    """Convert a percentage to decimal.

    Example:
        >>> percent_to_decimal(50)
        0.5
    """
    return pct / 100.0


def decimal_to_percent(dec: float) -> float:
    """Convert a decimal to percentage.

    Example:
        >>> decimal_to_percent(0.75)
        75.0
    """
    return dec * 100.0


def percent_of(part: float, whole: float) -> float:
    """What percent is *part* of *whole*?

    Example:
        >>> percent_of(25, 100)
        25.0
    """
    return (part / whole) * 100.0


def percentage(percent: float, whole: float) -> float:
    """Return *percent*% of *whole*.

    Example:
        >>> percentage(20, 50)
        10.0
    """
    return (percent / 100.0) * whole


# ---------------------------------------------------------------------------
# Module wrapper for the facade
# ---------------------------------------------------------------------------

class BasicMath:
    """Beginner-friendly basic math operations.

    Access all operations as methods on ``Axiom.math``.
    """

    # Arithmetic
    add = staticmethod(lambda a, b: a + b)
    sub = staticmethod(lambda a, b: a - b)
    mul = staticmethod(lambda a, b: a * b)
    div = staticmethod(lambda a, b: a / b)
    int_div = staticmethod(lambda a, b: a // b)

    # PEMDAS evaluator
    evaluate = staticmethod(evaluate)

    # Primes
    is_prime = staticmethod(is_prime)
    nth_prime = staticmethod(nth_prime)
    primes_up_to = staticmethod(primes_up_to)

    # Fractions
    Fraction = Fraction

    # Linear equations
    solve_linear = staticmethod(solve_linear)

    # Geometry
    rectangle_perimeter = staticmethod(rectangle_perimeter)
    rectangle_area = staticmethod(rectangle_area)
    square_perimeter = staticmethod(square_perimeter)
    square_area = staticmethod(square_area)
    circle_circumference = staticmethod(circle_circumference)
    circle_area = staticmethod(circle_area)
    triangle_area = staticmethod(triangle_area)
    pythagorean = staticmethod(pythagorean)
    pythagorean_leg = staticmethod(pythagorean_leg)

    # Statistics
    mean = staticmethod(mean)
    median = staticmethod(median)
    mode = staticmethod(mode)

    # Probability
    coin_flip = staticmethod(coin_flip)
    dice_roll = staticmethod(dice_roll)
    factorial = staticmethod(factorial)
    permutations = staticmethod(permutations)
    combinations = staticmethod(combinations)

    # Speed/distance/time
    speed = staticmethod(speed)
    distance_traveled = staticmethod(distance_traveled)
    travel_time = staticmethod(travel_time)

    # Exponents / roots
    power = staticmethod(power)
    nth_root = staticmethod(nth_root)
    sqrt = staticmethod(sqrt)

    # Percentages
    percent_to_decimal = staticmethod(percent_to_decimal)
    decimal_to_percent = staticmethod(decimal_to_percent)
    percent_of = staticmethod(percent_of)
    percentage = staticmethod(percentage)

    def help(self) -> str:
        """Print a beginner-friendly overview of all operations."""
        lines = [
            "=== AxiomPy Basic Math Help ===",
            "",
            "Arithmetic:",
            "  Axiom.math.add(a, b)      -> a + b",
            "  Axiom.math.sub(a, b)      -> a - b",
            "  Axiom.math.mul(a, b)      -> a * b",
            "  Axiom.math.div(a, b)      -> a / b",
            "",
            "PEMDAS Evaluator:",
            "  Axiom.math.evaluate('2 + 3 * 4')  -> 14",
            "",
            "Fractions:",
            "  f = Axiom.math.Fraction(1, 2)     -> 1/2",
            "  f + Fraction(1, 4)                -> 3/4",
            "",
            "Linear Equations:",
            "  Axiom.math.solve_linear('x + 2 = 5')  -> 3.0",
            "",
            "Geometry:",
            "  Axiom.math.rectangle_perimeter(4, 6)  -> 20.0",
            "  Axiom.math.pythagorean(3, 4)          -> 5.0",
            "",
            "Statistics:",
            "  Axiom.math.mean([2, 4, 6])      -> 4.0",
            "",
            "Probability:",
            "  Axiom.math.coin_flip()          -> 'Heads' or 'Tails'",
            "  Axiom.math.dice_roll(6)         -> 1..6",
            "",
            "Exponents & Roots:",
            "  Axiom.math.power(3, 2)          -> 9.0",
            "  Axiom.math.sqrt(16)             -> 4.0",
            "",
            "Percentages:",
            "  Axiom.math.percent_to_decimal(50)  -> 0.5",
            "",
            "Speed / Distance / Time:",
            "  Axiom.math.speed(60, 1)           -> 60.0",
            "",
            ">>> Use 'help(function)' for detailed docstrings.",
        ]
        return "\n".join(lines)
