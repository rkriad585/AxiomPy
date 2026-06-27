import math
import random
from typing import Callable, List, Union


class Calculus:
    @staticmethod
    def numerical_derivative(f: Callable[[float], float], x: float,
                             h: float = 1e-6) -> float:
        return (f(x + h) - f(x - h)) / (2 * h)

    @staticmethod
    def integrate_trapezoid(f: Callable[[float], float], a: float, b: float,
                            n: int = 100) -> float:
        h = (b - a) / n
        s = 0.5 * (f(a) + f(b))
        for i in range(1, n):
            s += f(a + i * h)
        return s * h

    @staticmethod
    def integrate_simpson(f: Callable[[float], float], a: float, b: float,
                          n: int = 100) -> float:
        if n % 2 == 1:
            n += 1
        h = (b - a) / n
        s = f(a) + f(b)
        for i in range(1, n, 2):
            s += 4 * f(a + i * h)
        for i in range(2, n - 1, 2):
            s += 2 * f(a + i * h)
        return s * h / 3

    @staticmethod
    def integrate_monte_carlo(f: Callable[[float], float], a: float, b: float,
                              n: int = 10000) -> float:
        total = 0.0
        for _ in range(n):
            total += f(a + random.random() * (b - a))
        return (b - a) * total / n

    @staticmethod
    def gradient(f: Callable[[List[float]], float],
                 point: List[float], h: float = 1e-6) -> List[float]:
        grad = []
        for i in range(len(point)):
            shifted_plus = point[:]
            shifted_minus = point[:]
            shifted_plus[i] += h
            shifted_minus[i] -= h
            grad.append((f(shifted_plus) - f(shifted_minus)) / (2 * h))
        return grad
