from typing import Callable, Optional


class Optimization:
    @staticmethod
    def gradient_descent(f: Callable[[float], float],
                         grad_f: Callable[[float], float],
                         start: float, lr: float = 0.01,
                         steps: int = 100) -> float:
        x = start
        for _ in range(steps):
            x -= lr * grad_f(x)
        return x

    @staticmethod
    def newton_method(f: Callable[[float], float],
                      df: Callable[[float], float],
                      d2f: Callable[[float], float],
                      start: float, steps: int = 100) -> float:
        x = start
        for _ in range(steps):
            denom = d2f(x)
            if denom == 0:
                break
            x -= df(x) / denom
        return x

    @staticmethod
    def bisection(f: Callable[[float], float],
                  a: float, b: float, tol: float = 1e-6) -> Optional[float]:
        fa, fb = f(a), f(b)
        if fa * fb >= 0:
            return None
        while (b - a) / 2 > tol:
            c = (a + b) / 2
            fc = f(c)
            if fc == 0:
                return c
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
        return (a + b) / 2
