import logging
import math
import random
from typing import Callable, Optional

import numpy as np

from .matrix import Matrix
from .vector import Vector

logger = logging.getLogger(__name__)


class Optimization:
    """Numeric optimization methods for univariate and multivariate functions."""

    @staticmethod
    def gradient_descent(f: Callable[[float], float],
                         grad_f: Callable[[float], float],
                         start: float, lr: float = 0.01,
                         steps: int = 100) -> float:
        """Minimize f using gradient descent.

        Args:
            f (Callable[[float], float]): Objective function.
            grad_f (Callable[[float], float]): Gradient of f.
            start (float): Initial point.
            lr (float): Learning rate.
            steps (int): Number of iterations.

        Returns:
            float: Approximate minimum point.
        """
        x = start
        for _ in range(steps):
            x -= lr * grad_f(x)
        return x

    @staticmethod
    def newton_method(f: Callable[[float], float],
                      df: Callable[[float], float],
                      d2f: Callable[[float], float],
                      start: float, steps: int = 100) -> float:
        """Find a root of f using Newton's method.

        Args:
            f (Callable[[float], float]): Function whose root to find.
            df (Callable[[float], float]): First derivative of f.
            d2f (Callable[[float], float]): Second derivative of f.
            start (float): Initial guess.
            steps (int): Maximum number of iterations.

        Returns:
            float: Approximate root location.
        """
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
        """Find a root of f in [a, b] using the bisection method.

        Args:
            f (Callable[[float], float]): Continuous function.
            a (float): Left endpoint of the interval.
            b (float): Right endpoint of the interval.
            tol (float): Convergence tolerance.

        Returns:
            Optional[float]: Approximate root, or None if signs at a and b are the same.
        """
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

    @staticmethod
    def golden_section(f: Callable[[float], float], a: float, b: float,
                        tol: float = 1e-6) -> float:
        """Minimize a univariate function using golden section search.

        Args:
            f (Callable[[float], float]): Unimodal function to minimize.
            a (float): Left bound of search interval.
            b (float): Right bound of search interval.
            tol (float): Convergence tolerance (default 1e-6).

        Returns:
            float: Approximate minimum point.
        """
        phi = (math.sqrt(5) - 1) / 2
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        fc, fd = f(c), f(d)
        while abs(b - a) > tol:
            if fc < fd:
                b, d = d, c
                fd = fc
                c = b - phi * (b - a)
                fc = f(c)
            else:
                a, c = c, d
                fc = fd
                d = a + phi * (b - a)
                fd = f(d)
        return (a + b) / 2

    @staticmethod
    def conjugate_gradient(A: Matrix, b: Vector,
                            x0: Optional[Vector] = None,
                            tol: float = 1e-6,
                            max_iter: Optional[int] = None) -> Vector:
        """Solve the linear system Ax = b using the conjugate gradient method.

        A must be symmetric positive-definite.

        Args:
            A (Matrix): SPD coefficient matrix.
            b (Vector): Right-hand side vector.
            x0 (Vector, optional): Initial guess (default zero vector).
            tol (float): Convergence tolerance (default 1e-6).
            max_iter (int, optional): Maximum iterations (default n).

        Returns:
            Vector: Solution vector x.
        """
        n = b._data.shape[0]
        if x0 is None:
            x = np.zeros(n)
        else:
            x = x0._data.copy()
        r = b._data - A._data @ x
        p = r.copy()
        rsold = r @ r
        if max_iter is None:
            max_iter = n
        for _ in range(max_iter):
            Ap = A._data @ p
            alpha = rsold / (p @ Ap)
            x += alpha * p
            r -= alpha * Ap
            rsnew = r @ r
            if math.sqrt(rsnew) < tol:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        return Vector(x.tolist())

    @staticmethod
    def nelder_mead(f: Callable[[list[float]], float],
                    start: list[float],
                    max_iter: int = 1000,
                    tol: float = 1e-6) -> list[float]:
        """Minimize a function using the Nelder-Mead simplex algorithm.

        Args:
            f (Callable[[List[float]], float]): Objective function.
            start (List[float]): Initial point.
            max_iter (int): Maximum iterations (default 1000).
            tol (float): Convergence tolerance (default 1e-6).

        Returns:
            List[float]: Approximate minimum point.
        """
        n = len(start)
        simplex = [np.array(start, dtype=float)]
        for i in range(n):
            pt = np.array(start, dtype=float)
            pt[i] += 0.05 if pt[i] == 0 else 0.05 * abs(pt[i])
            simplex.append(pt)
        values = [f(list(pt)) for pt in simplex]

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        for _ in range(max_iter):
            order = sorted(range(len(simplex)), key=lambda i: values[i])
            if values[order[-1]] - values[order[0]] < tol:
                break

            centroid = sum(simplex[order[i]] for i in range(n)) / n

            xr = centroid + alpha * (centroid - simplex[order[-1]])
            fr = f(list(xr))
            if values[order[0]] <= fr < values[order[-2]]:
                simplex[order[-1]] = xr
                values[order[-1]] = fr
            elif fr < values[order[0]]:
                xe = centroid + gamma * (xr - centroid)
                fe = f(list(xe))
                if fe < fr:
                    simplex[order[-1]] = xe
                    values[order[-1]] = fe
                else:
                    simplex[order[-1]] = xr
                    values[order[-1]] = fr
            else:
                xc = centroid + rho * (simplex[order[-1]] - centroid)
                fc = f(list(xc))
                if fc < values[order[-1]]:
                    simplex[order[-1]] = xc
                    values[order[-1]] = fc
                else:
                    for i in range(1, len(simplex)):
                        simplex[order[i]] = simplex[order[0]] + sigma * (simplex[order[i]] - simplex[order[0]])
                        values[order[i]] = f(list(simplex[order[i]]))

        return list(simplex[order[0]])

    @staticmethod
    def lbfgs(f: Callable[[list[float]], float],
              grad_f: Callable[[list[float]], list[float]],
              start: list[float],
              max_iter: int = 100,
              tol: float = 1e-6,
              m: int = 10) -> list[float]:
        """Minimize a function using L-BFGS (limited-memory BFGS).

        Args:
            f (Callable[[List[float]], float]): Objective function.
            grad_f (Callable[[List[float]], List[float]]): Gradient function.
            start (List[float]): Initial point.
            max_iter (int): Maximum iterations (default 100).
            tol (float): Convergence tolerance (default 1e-6).
            m (int): Limited memory size (default 10).

        Returns:
            List[float]: Approximate minimum point.
        """
        x = np.array(start, dtype=float)
        grad = np.array(grad_f(list(x)), dtype=float)
        s_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        rho_list: list[float] = []

        for _ in range(max_iter):
            if np.linalg.norm(grad) < tol:
                break

            if len(s_list) == 0:
                d = -grad
            else:
                q = grad.copy()
                alphas = []
                for s, y, rho in zip(reversed(s_list), reversed(y_list), reversed(rho_list)):
                    alpha = rho * (s @ q)
                    alphas.append(alpha)
                    q -= alpha * y
                Hk0 = (s_list[-1] @ y_list[-1]) / (y_list[-1] @ y_list[-1])
                r = Hk0 * q
                for s, y, rho, alpha in zip(s_list, y_list, rho_list, reversed(alphas)):
                    beta = rho * (y @ r)
                    r += s * (alpha - beta)
                d = -r

            step = 1.0
            for _ in range(20):
                x_new = x + step * d
                if f(list(x_new)) < f(list(x)) + 1e-4 * step * (grad @ d):
                    break
                step *= 0.5

            s = x_new - x
            grad_new = np.array(grad_f(list(x_new)), dtype=float)
            y = grad_new - grad

            sy = s @ y
            if sy > 1e-12:
                rho_val = 1.0 / sy
                s_list.append(s.copy())
                y_list.append(y.copy())
                rho_list.append(rho_val)
                if len(s_list) > m:
                    s_list.pop(0)
                    y_list.pop(0)
                    rho_list.pop(0)

            x, grad = x_new, grad_new

        return list(x)

    @staticmethod
    def simulated_annealing(f: Callable[[list[float]], float],
                             start: list[float],
                             temp: float = 1.0,
                             cooling: float = 0.95,
                             max_iter: int = 1000) -> list[float]:
        """Minimize a function using simulated annealing.

        Args:
            f (Callable[[List[float]], float]): Objective function.
            start (List[float]): Initial point.
            temp (float): Initial temperature (default 1.0).
            cooling (float): Cooling rate (default 0.95).
            max_iter (int): Maximum iterations (default 1000).

        Returns:
            List[float]: Approximate global minimum point.
        """
        current = np.array(start, dtype=float)
        best = current.copy()
        current_val = f(list(current))
        best_val = current_val
        n = len(start)

        for i in range(max_iter):
            t = temp * (cooling ** i)
            candidate = current + np.random.uniform(-1, 1, n) * t
            candidate_val = f(list(candidate))
            delta = candidate_val - current_val
            if delta < 0 or random.random() < math.exp(-delta / max(t, 1e-12)):
                current = candidate
                current_val = candidate_val
                if current_val < best_val:
                    best = current.copy()
                    best_val = current_val

        return list(best)
