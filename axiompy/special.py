import math
from typing import Union

Number = Union[int, float]


# ---- Lanczos coefficients for Gamma approximation -------------------------

_LANCZOS_COEFFS = [
    0.99999999999980993,
    676.5203681218851,
   -1259.1392167224028,
    771.32342877765313,
   -176.61502916214059,
    12.507343278686905,
   -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
]


def _lanczos_gamma(z: float) -> float:
    """Compute Gamma(z) via Lanczos approximation for real z."""
    if z < 0.5:
        # Reflection formula
        return math.pi / (math.sin(math.pi * z) * _lanczos_gamma(1.0 - z))

    z -= 1.0
    x = _LANCZOS_COEFFS[0]
    for i in range(1, 9):
        x += _LANCZOS_COEFFS[i] / (z + i)
    t = z + 7.5
    return math.sqrt(2 * math.pi) * (t ** (z + 0.5)) * math.exp(-t) * x


def gamma(z: Number) -> float:
    """Evaluate the Gamma function for real z.

    Uses the Lanczos approximation (accurate to ~15 decimal places).

    Args:
        z: Real argument (z != 0, -1, -2, ...).

    Returns:
        float: Gamma(z).
    """
    return _lanczos_gamma(float(z))


def beta(a: Number, b: Number) -> float:
    """Evaluate the Beta function B(a, b) = Gamma(a) Gamma(b) / Gamma(a+b).

    Args:
        a: First parameter (positive).
        b: Second parameter (positive).

    Returns:
        float: Beta(a, b).
    """
    return gamma(a) * gamma(b) / gamma(a + b)


def _erf_approx(x: float) -> float:
    """Abramowitz & Stegun approximation for erf (max error 1.5e-7)."""
    if x == 0.0:
        return 0.0
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    a = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]
    poly = ((((a[4] * t + a[3]) * t + a[2]) * t + a[1]) * t + a[0]) * t
    return sign * (1.0 - poly * math.exp(-x * x))


def erf(x: Number) -> float:
    """Evaluate the error function.

    Args:
        x: Real argument.

    Returns:
        float: erf(x).
    """
    return _erf_approx(float(x))


def erfc(x: Number) -> float:
    """Evaluate the complementary error function erfc(x) = 1 - erf(x).

    Args:
        x: Real argument.

    Returns:
        float: erfc(x).
    """
    return 1.0 - erf(float(x))


def _bessel_j_series(n: int, x: float) -> float:
    """Bessel J_n(x) via ascending series."""
    if x == 0.0:
        return 1.0 if n == 0 else 0.0
    s = 0.0
    term = (x / 2) ** n / math.factorial(n)
    for k in range(0, 100):
        if k > 0:
            term *= -(x * x) / (4 * k * (n + k))
        s += term
        if abs(term) < 1e-15 * abs(s):
            break
    return s


def bessel_j(n: int, x: Number) -> float:
    """Evaluate the Bessel function of the first kind J_n(x).

    Args:
        n: Order (integer >= 0).
        x: Real argument.

    Returns:
        float: J_n(x).
    """
    return _bessel_j_series(n, float(x))


def _bessel_y0(x: float) -> float:
    """Y0 Bessel function via series (A&S 9.1.13)."""
    if x <= 0:
        return float("-inf") if x == 0 else float("nan")
    _euler_gamma = 0.57721566490153286060651209
    j0 = _bessel_j_series(0, x)
    s = 0.0
    term = 1.0  # (x/2)^{2k} / (k!)^2 for k=0
    for k in range(1, 200):
        term *= -0.25 * x * x / (k * k)
        hk = sum(1.0 / i for i in range(1, k + 1))
        s -= term * hk
        if abs(term * hk) < 1e-15 * abs(s):
            break
    return (2.0 / math.pi) * (j0 * (math.log(x / 2) + _euler_gamma) + s)


def _bessel_y_n_forward(n: int, x: float, y0: float) -> float:
    """Compute Y_n(x) via forward recurrence from Y0."""
    if n == 0:
        return y0
    # Compute Y1 via series for n=1 (A&S 9.1.13)
    _euler_gamma = 0.57721566490153286060651209
    j1 = _bessel_j_series(1, x)
    s = 0.0
    term = 1.0  # (x/2)^{2k} / (k! (k+1)!) for k=0
    for k in range(1, 200):
        term *= -0.25 * x * x / (k * (k + 1))
        hk = sum(1.0 / i for i in range(1, k + 1))
        s += term * hk
        if abs(term * hk) < 1e-15 * abs(s):
            break
    y1 = (2.0 / math.pi) * (j1 * (math.log(x / 2) + _euler_gamma) - 1.0 / x + s)
    if n == 1:
        return y1
    y_prev, y_curr = y0, y1
    for i in range(1, n):
        y_next = (2.0 * i / x) * y_curr - y_prev
        y_prev, y_curr = y_curr, y_next
    return y_curr


def bessel_y(n: int, x: Number) -> float:
    """Evaluate the Bessel function of the second kind Y_n(x).

    Uses series (A&S 9.1.13) and forward recurrence.

    Args:
        n: Order (integer >= 0).
        x: Positive real argument.

    Returns:
        float: Y_n(x).
    """
    x = float(x)
    y0 = _bessel_y0(x)
    return _bessel_y_n_forward(n, x, y0)


def legendre_p(n: int, x: Number) -> float:
    """Evaluate the Legendre polynomial P_n(x) via Bonnet's recurrence.

    P_0 = 1, P_1 = x,
    (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x).

    Args:
        n: Degree (integer >= 0).
        x: Real argument in [-1, 1].

    Returns:
        float: P_n(x).
    """
    x = float(x)
    if n == 0:
        return 1.0
    if n == 1:
        return x
    p0, p1 = 1.0, x
    for k in range(1, n):
        p2 = ((2 * k + 1) * x * p1 - k * p0) / (k + 1)
        p0, p1 = p1, p2
    return p1


def factorial(n: int) -> int:
    """Compute n! for integer n >= 0.

    Args:
        n: Non-negative integer.

    Returns:
        int: n!
    """
    return math.factorial(n)


def binomial(n: int, k: int) -> int:
    """Compute the binomial coefficient C(n, k).

    Args:
        n: Total items.
        k: Items chosen (0 <= k <= n).

    Returns:
        int: n choose k.
    """
    return math.comb(n, k)


def double_factorial(n: int) -> int:
    """Compute the double factorial n!!.

    n!! = n * (n-2) * (n-4) * ... * 1 (or 2).

    Args:
        n: Non-negative integer.

    Returns:
        int: n!!
    """
    if n <= 1:
        return 1
    result = 1
    while n > 1:
        result *= n
        n -= 2
    return result
