from typing import List, Union, Tuple
from numbers import Number

import numpy as np


class Polynomial:
    def __init__(self, coeffs: List[float]):
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs.pop()
        self.coeffs = [float(c) for c in coeffs]

    @property
    def degree(self) -> int:
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

    def __call__(self, x: float) -> float:
        result = 0.0
        for i, c in enumerate(self.coeffs):
            result += c * (x ** i)
        return result

    def __add__(self, other: 'Polynomial') -> 'Polynomial':
        n = max(len(self.coeffs), len(other.coeffs))
        a = self.coeffs + [0.0] * (n - len(self.coeffs))
        b = other.coeffs + [0.0] * (n - len(other.coeffs))
        return Polynomial([x + y for x, y in zip(a, b)])

    def __sub__(self, other: 'Polynomial') -> 'Polynomial':
        n = max(len(self.coeffs), len(other.coeffs))
        a = self.coeffs + [0.0] * (n - len(self.coeffs))
        b = other.coeffs + [0.0] * (n - len(other.coeffs))
        return Polynomial([x - y for x, y in zip(a, b)])

    def __mul__(self, other: Union['Polynomial', Number]) -> 'Polynomial':
        if isinstance(other, Number):
            return Polynomial([c * other for c in self.coeffs])
        n = len(self.coeffs) + len(other.coeffs) - 1
        result = [0.0] * n
        for i, a in enumerate(self.coeffs):
            for j, b in enumerate(other.coeffs):
                result[i + j] += a * b
        return Polynomial(result)

    def __rmul__(self, other: Number) -> 'Polynomial':
        return self.__mul__(other)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Polynomial):
            return NotImplemented
        return self.coeffs == other.coeffs

    def derivative(self) -> 'Polynomial':
        if self.degree < 1:
            return Polynomial([0.0])
        return Polynomial([c * i for i, c in enumerate(self.coeffs)][1:])

    def integral(self, constant: float = 0.0) -> 'Polynomial':
        result = [constant]
        for i, c in enumerate(self.coeffs):
            result.append(c / (i + 1))
        return Polynomial(result)

    def roots(self) -> List[complex]:
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
    def lagrange_interpolate(xs: List[float], ys: List[float]) -> 'Polynomial':
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
