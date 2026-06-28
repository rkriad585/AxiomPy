import logging
import math
from numbers import Number
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)


class Polynomial:
    """A univariate polynomial with real coefficients, supporting arithmetic and calculus operations."""

    def __init__(self, coeffs: list[float]):
        """Initialize a polynomial with coefficients in ascending order of degree.

        Args:
            coeffs (List[float]): Coefficient list where index i corresponds to x^i.
                                  Trailing zeros are stripped.
        """
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs.pop()
        self.coeffs = [float(c) for c in coeffs]

    @property
    def degree(self) -> int:
        """Return the degree of the polynomial.

        Returns:
            int: Degree of the polynomial (0 for constant polynomials).
        """
        return len(self.coeffs) - 1

    def __repr__(self) -> str:
        terms = []
        for i, c in enumerate(self.coeffs):
            if c == 0:
                continue
            if i == 0:
                terms.append(f"{c}")
            elif i == 1:
                terms.append(f"{c}x" if c != 1 else "x")
            else:
                terms.append(f"{c}x^{i}" if c != 1 else f"x^{i}")
        return "Polynomial(" + " + ".join(reversed(terms)) + ")" if terms else "Polynomial(0)"

    def to_latex(self) -> str:
        """Return a LaTeX polynomial string.

        Returns:
            str: e.g. ``x^{2} + 2x + 1``.
        """
        terms = []
        for i, c in enumerate(self.coeffs):
            if c == 0:
                continue
            if i == 0:
                terms.append(f"{c}")
            elif i == 1:
                terms.append(f"{c}x" if c != 1 else "x")
            else:
                terms.append(f"{c}x^{{{i}}}" if c != 1 else f"x^{{{i}}}")
        return " + ".join(reversed(terms)) if terms else "0"

    def _repr_latex_(self) -> str:
        """LaTeX representation for Jupyter notebooks."""
        return f"$${self.to_latex()}$$"

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        terms = []
        for i, c in enumerate(self.coeffs):
            if c == 0:
                continue
            if i == 0:
                terms.append(f"<span>{c}</span>")
            elif i == 1:
                terms.append(f"<span>{c}x</span>" if c != 1 else "<span>x</span>")
            else:
                terms.append(f"<span>{c}x<sup>{i}</sup></span>" if c != 1 else f"<span>x<sup>{i}</sup></span>")
        expr = " + ".join(reversed(terms)) if terms else "0"
        return f"<div style='font-style:italic;'>{expr}</div>"

    def __call__(self, x: float) -> float:
        """Evaluate the polynomial at a given value of x.

        Args:
            x (float): Point at which to evaluate.

        Returns:
            float: Value of the polynomial at x.
        """
        result = 0.0
        for i, c in enumerate(self.coeffs):
            result += c * (x ** i)
        return result

    def __add__(self, other: 'Polynomial') -> 'Polynomial':
        """Add two polynomials.

        Args:
            other (Polynomial): The polynomial to add.

        Returns:
            Polynomial: A new polynomial representing the sum.
        """
        n = max(len(self.coeffs), len(other.coeffs))
        a = self.coeffs + [0.0] * (n - len(self.coeffs))
        b = other.coeffs + [0.0] * (n - len(other.coeffs))
        return Polynomial([x + y for x, y in zip(a, b)])

    def __sub__(self, other: 'Polynomial') -> 'Polynomial':
        """Subtract another polynomial from this one.

        Args:
            other (Polynomial): The polynomial to subtract.

        Returns:
            Polynomial: A new polynomial representing the difference.
        """
        n = max(len(self.coeffs), len(other.coeffs))
        a = self.coeffs + [0.0] * (n - len(self.coeffs))
        b = other.coeffs + [0.0] * (n - len(other.coeffs))
        return Polynomial([x - y for x, y in zip(a, b)])

    def __mul__(self, other: Union['Polynomial', Number]) -> 'Polynomial':
        """Multiply by another polynomial or a scalar.

        Args:
            other (Union[Polynomial, Number]): The multiplier.

        Returns:
            Polynomial: A new polynomial representing the product.
        """
        if isinstance(other, Number):
            return Polynomial([c * other for c in self.coeffs])
        n = len(self.coeffs) + len(other.coeffs) - 1
        result = [0.0] * n
        for i, a in enumerate(self.coeffs):
            for j, b in enumerate(other.coeffs):
                result[i + j] += a * b
        return Polynomial(result)

    def __rmul__(self, other: Number) -> 'Polynomial':
        """Multiply a scalar by this polynomial.

        Args:
            other (Number): The scalar multiplier.

        Returns:
            Polynomial: A new polynomial representing the product.
        """
        return self.__mul__(other)

    def __floordiv__(self, other: 'Polynomial') -> 'Polynomial':
        """Compute the quotient of polynomial division.

        Args:
            other (Polynomial): The divisor polynomial.

        Returns:
            Polynomial: The quotient.
        """
        return self._divmod(other)[0]

    def __mod__(self, other: 'Polynomial') -> 'Polynomial':
        """Compute the remainder of polynomial division.

        Args:
            other (Polynomial): The divisor polynomial.

        Returns:
            Polynomial: The remainder.
        """
        return self._divmod(other)[1]

    def _divmod(self, other: 'Polynomial') -> tuple['Polynomial', 'Polynomial']:
        """Perform polynomial division with remainder.

        Args:
            other (Polynomial): The divisor.

        Returns:
            Tuple[Polynomial, Polynomial]: (quotient, remainder).
        """
        dividend = self.coeffs[:]
        divisor = other.coeffs[:]
        if len(divisor) == 1 and divisor[0] == 0:
            raise ZeroDivisionError("polynomial division by zero")
        quotient = [0.0] * (max(len(dividend), len(divisor)) - len(divisor) + 1)
        while len(dividend) >= len(divisor):
            factor = dividend[-1] / divisor[-1]
            idx = len(dividend) - len(divisor)
            quotient[idx] = factor
            for i, c in enumerate(divisor):
                dividend[idx + i] -= factor * c
            while dividend and abs(dividend[-1]) < 1e-14:
                dividend.pop()
        return Polynomial(quotient), Polynomial(dividend if dividend else [0.0])

    def __eq__(self, other) -> bool:
        """Check equality of two polynomials.

        Args:
            other: Object to compare against.

        Returns:
            bool: True if both polynomials have identical coefficients.
        """
        if not isinstance(other, Polynomial):
            return NotImplemented
        return self.coeffs == other.coeffs

    def derivative(self) -> 'Polynomial':
        """Compute the derivative of this polynomial.

        Returns:
            Polynomial: A new polynomial representing the derivative.
        """
        if self.degree < 1:
            return Polynomial([0.0])
        return Polynomial([c * i for i, c in enumerate(self.coeffs)][1:])

    def integral(self, constant: float = 0.0) -> 'Polynomial':
        """Compute the indefinite integral of this polynomial.

        Args:
            constant (float): Constant of integration.

        Returns:
            Polynomial: A new polynomial representing the integral.
        """
        result = [constant]
        for i, c in enumerate(self.coeffs):
            result.append(c / (i + 1))
        return Polynomial(result)

    def roots(self) -> list[complex]:
        """Find the roots of the polynomial using the companion matrix method.

        Returns:
            List[complex]: List of roots (real or complex).
        """
        n = self.degree
        if n == 0:
            return []
        if n == 1:
            return [-self.coeffs[0] / self.coeffs[1]]
        companion = np.zeros((n, n))
        for i in range(n - 1):
            companion[i + 1, i] = 1
        for i in range(n):
            companion[i, n - 1] = -self.coeffs[i] / self.coeffs[n]
        eigvals = np.linalg.eigvals(companion)
        return [complex(round(e.real, 12), round(e.imag, 12)) for e in eigvals]

    @staticmethod
    def lagrange_interpolate(xs: list[float], ys: list[float]) -> 'Polynomial':
        """Construct the unique polynomial that passes through given points using Lagrange interpolation.

        Args:
            xs (List[float]): x-coordinates of the points.
            ys (List[float]): y-coordinates of the points.

        Returns:
            Polynomial: The interpolating polynomial.
        """
        n = len(xs)
        result = Polynomial([0.0])
        for i in range(n):
            term = Polynomial([1.0])
            for j in range(n):
                if i == j:
                    continue
                term = term * Polynomial([-xs[j], 1.0]) * (1.0 / (xs[i] - xs[j]))
            result = result + term * ys[i]
        return result

    def gcd(self, other: 'Polynomial') -> 'Polynomial':
        """Compute the greatest common divisor of two polynomials using the Euclidean algorithm.

        Args:
            other (Polynomial): The other polynomial.

        Returns:
            Polynomial: The GCD polynomial.
        """
        a, b = self, other
        while not (len(b.coeffs) == 1 and b.coeffs[0] == 0):
            a, b = b, a % b
        lead = a.coeffs[-1]
        return Polynomial([c / lead for c in a.coeffs])

    def compose(self, other: 'Polynomial') -> 'Polynomial':
        """Compute the composition p(q(x)).

        Args:
            other (Polynomial): The inner polynomial q(x).

        Returns:
            Polynomial: The composed polynomial p(q(x)).
        """
        result = Polynomial([0.0])
        for i, c in enumerate(self.coeffs):
            term = Polynomial([c])
            for _ in range(i):
                term = term * other
            result = result + term
        return result

    @staticmethod
    def chebyshev_roots(n: int) -> list[float]:
        """Compute the roots of the Chebyshev polynomial of the first kind T_n(x).

        Args:
            n (int): The degree of the Chebyshev polynomial (>= 1).

        Returns:
            List[float]: The n roots in the interval (-1, 1).
        """
        return [math.cos((2 * k - 1) * math.pi / (2 * n)) for k in range(1, n + 1)]

    @staticmethod
    def fit(xs: list[float], ys: list[float], degree: int) -> 'Polynomial':
        """Fit a polynomial of given degree to data using least squares.

        Args:
            xs (List[float]): x-coordinates of the data points.
            ys (List[float]): y-coordinates of the data points.
            degree (int): Degree of the fitting polynomial.

        Returns:
            Polynomial: The fitted polynomial.
        """
        A = np.zeros((len(xs), degree + 1))
        for i, x in enumerate(xs):
            for j in range(degree + 1):
                A[i, j] = x ** j
        b = np.array(ys, dtype=float)
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
        return Polynomial(coeffs.tolist())
