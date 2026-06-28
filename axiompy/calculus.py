import logging
import random
from typing import Callable

logger = logging.getLogger(__name__)

# Gauss-Legendre nodes and weights for n=2..8
_GL_QUAD = {
    2: (
        [-0.5773502691896257, 0.5773502691896257],
        [1.0, 1.0],
    ),
    3: (
        [-0.7745966692414834, 0.0, 0.7745966692414834],
        [0.5555555555555556, 0.8888888888888888, 0.5555555555555556],
    ),
    4: (
        [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526],
        [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538],
    ),
    5: (
        [-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640],
        [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891],
    ),
    6: (
        [-0.9324695142031520, -0.6612093864662645, -0.2386191860831969,
         0.2386191860831969, 0.6612093864662645, 0.9324695142031520],
        [0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
         0.4679139345726910, 0.3607615730481386, 0.1713244923791704],
    ),
    7: (
        [-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0.0,
         0.4058451513773972, 0.7415311855993945, 0.9491079123427585],
        [0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694,
         0.3818300505051189, 0.2797053914892766, 0.1294849661688697],
    ),
    8: (
        [-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
         0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363],
        [0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620,
         0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763],
    ),
}


class Calculus:
    """Numerical calculus operations including differentiation and integration."""

    @staticmethod
    def numerical_derivative(f: Callable[[float], float], x: float,
                             h: float = 1e-6) -> float:
        """Approximate the first derivative of f at x using the central difference method.

        Args:
            f (Callable[[float], float]): Function to differentiate.
            x (float): Point at which to evaluate the derivative.
            h (float): Step size for the finite difference.

        Returns:
            float: Approximate value of f'(x).
        """
        return (f(x + h) - f(x - h)) / (2 * h)

    @staticmethod
    def richardson_extrapolation(f: Callable[[float], float], x: float,
                                  h: float = 1e-3, order: int = 2) -> float:
        """Approximate the first derivative using Richardson extrapolation.

        Applies Richardson extrapolation to central difference estimates
        to obtain a higher-order derivative approximation.

        Args:
            f (Callable[[float], float]): Function to differentiate.
            x (float): Point at which to evaluate the derivative.
            h (float): Initial step size (default 1e-3).
            order (int): Order of extrapolation (default 2, yields O(h⁴)).

        Returns:
            float: Approximate value of f'(x).
        """
        d1 = (f(x + h) - f(x - h)) / (2 * h)
        d2 = (f(x + h / 2) - f(x - h / 2)) / h
        if order >= 2:
            return (4 * d2 - d1) / 3
        return d1

    @staticmethod
    def integrate_trapezoid(f: Callable[[float], float], a: float, b: float,
                            n: int = 100) -> float:
        """Approximate the definite integral of f from a to b using the trapezoidal rule.

        Args:
            f (Callable[[float], float]): Function to integrate.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.
            n (int): Number of subintervals.

        Returns:
            float: Approximate value of the integral.
        """
        h = (b - a) / n
        s = 0.5 * (f(a) + f(b))
        for i in range(1, n):
            s += f(a + i * h)
        return s * h

    @staticmethod
    def integrate_simpson(f: Callable[[float], float], a: float, b: float,
                          n: int = 100) -> float:
        """Approximate the definite integral of f from a to b using Simpson's rule.

        Args:
            f (Callable[[float], float]): Function to integrate.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.
            n (int): Number of subintervals (adjusted to be even).

        Returns:
            float: Approximate value of the integral.
        """
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
        """Approximate the definite integral of f from a to b using Monte Carlo integration.

        Args:
            f (Callable[[float], float]): Function to integrate.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.
            n (int): Number of random samples.

        Returns:
            float: Approximate value of the integral.
        """
        total = 0.0
        for _ in range(n):
            total += f(a + random.random() * (b - a))
        return (b - a) * total / n

    @staticmethod
    def gradient(f: Callable[[list[float]], float],
                 point: list[float], h: float = 1e-6) -> list[float]:
        """Approximate the gradient of a multivariate function at a given point.

        Args:
            f (Callable[[List[float]], float]): Multivariate function.
            point (List[float]): Point at which to evaluate the gradient.
            h (float): Step size for the finite difference.

        Returns:
            List[float]: Approximate gradient vector.
        """
        grad = []
        for i in range(len(point)):
            shifted_plus = point[:]
            shifted_minus = point[:]
            shifted_plus[i] += h
            shifted_minus[i] -= h
            grad.append((f(shifted_plus) - f(shifted_minus)) / (2 * h))
        return grad

    @staticmethod
    def integrate_gauss_legendre(f: Callable[[float], float], a: float, b: float,
                                  n: int = 5) -> float:
        """Approximate the definite integral using Gauss-Legendre quadrature.

        Args:
            f (Callable[[float], float]): Function to integrate.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.
            n (int): Number of quadrature points (2-8, default 5).

        Returns:
            float: Approximate value of the integral.
        """
        if n not in _GL_QUAD:
            n = 5
        nodes, weights = _GL_QUAD[n]
        mid = (b + a) / 2
        half = (b - a) / 2
        total = 0.0
        for node, w in zip(nodes, weights):
            total += w * f(mid + half * node)
        return total * half

    @staticmethod
    def integrate_romberg(f: Callable[[float], float], a: float, b: float,
                           tol: float = 1e-6, max_steps: int = 15) -> float:
        """Approximate the definite integral using Romberg integration.

        Args:
            f (Callable[[float], float]): Function to integrate.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.
            tol (float): Convergence tolerance (default 1e-6).
            max_steps (int): Maximum number of Romberg steps (default 15).

        Returns:
            float: Approximate value of the integral.
        """
        R: list[list[float]] = [[0.5 * (b - a) * (f(a) + f(b))]]
        for k in range(1, max_steps):
            n = 2 ** k
            h = (b - a) / n
            s = sum(f(a + (2 * i + 1) * h) for i in range(n // 2))
            row = [0.5 * R[k - 1][0] + h * s]
            for j in range(1, k + 1):
                val = row[j - 1] + (row[j - 1] - R[k - 1][j - 1]) / (4 ** j - 1)
                row.append(val)
            R.append(row)
            if k >= 4 and abs(R[k][k] - R[k - 1][k - 1]) < tol:
                return R[k][k]
        return R[-1][-1]

    @staticmethod
    def jacobian(f: Callable[[list[float]], list[float]],
                 point: list[float], h: float = 1e-6) -> list[list[float]]:
        """Approximate the Jacobian matrix of a vector-valued function.

        Args:
            f (Callable[[List[float]], List[float]]): Vector-valued function.
            point (List[float]): Point at which to evaluate the Jacobian.
            h (float): Step size for the finite difference.

        Returns:
            List[List[float]]: The m x n Jacobian matrix (m outputs, n inputs).
        """
        f0 = f(point)
        m = len(f0)
        n = len(point)
        J = [[0.0] * n for _ in range(m)]
        for i in range(n):
            shifted = point[:]
            shifted[i] += h
            f_plus = f(shifted)
            shifted[i] -= 2 * h
            f_minus = f(shifted)
            for j in range(m):
                J[j][i] = (f_plus[j] - f_minus[j]) / (2 * h)
        return J

    @staticmethod
    def hessian(f: Callable[[list[float]], float],
                point: list[float], h: float = 1e-5) -> list[list[float]]:
        """Approximate the Hessian matrix of a scalar-valued function.

        Uses central differences of the gradient for better accuracy.

        Args:
            f (Callable[[List[float]], float]): Scalar-valued function.
            point (List[float]): Point at which to evaluate the Hessian.
            h (float): Step size for the finite difference (default 1e-5).

        Returns:
            List[List[float]]: The symmetric n x n Hessian matrix.
        """
        n = len(point)
        H = [[0.0] * n for _ in range(n)]
        f0 = f(point)
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    p_plus = point[:]
                    p_plus[i] += h
                    p_minus = point[:]
                    p_minus[i] -= h
                    H[i][i] = (f(p_plus) - 2 * f0 + f(p_minus)) / (h * h)
                else:
                    pp = point[:]
                    pp[i] += h
                    pp[j] += h
                    pm = point[:]
                    pm[i] += h
                    pm[j] -= h
                    mp = point[:]
                    mp[i] -= h
                    mp[j] += h
                    mm = point[:]
                    mm[i] -= h
                    mm[j] -= h
                    val = (f(pp) - f(pm) - f(mp) + f(mm)) / (4 * h * h)
                    H[i][j] = val
                    H[j][i] = val
        return H

    @staticmethod
    def ode_euler(f: Callable[[float, list[float]], list[float]],
                  y0: list[float], t_span: tuple[float, float],
                  dt: float = 0.01) -> tuple[list[float], list[list[float]]]:
        """Solve an ODE initial value problem using the forward Euler method.

        Args:
            f (Callable): RHS function ``f(t, y) -> dy/dt``.
            y0 (List[float]): Initial state vector.
            t_span (Tuple[float, float]): (t_start, t_end).
            dt (float): Time step (default 0.01).

        Returns:
            Tuple[List[float], List[List[float]]]: Time points and solution trajectory.
        """
        t0, t1 = t_span
        n_steps = max(1, int((t1 - t0) / dt))
        dt = (t1 - t0) / n_steps
        ts = [t0 + i * dt for i in range(n_steps + 1)]
        ys = [y0[:]]
        y = y0[:]
        for t in ts[:-1]:
            y = [y[k] + dt * f(t, y)[k] for k in range(len(y))]
            ys.append(y[:])
        return ts, ys

    @staticmethod
    def ode_rk4(f: Callable[[float, list[float]], list[float]],
                y0: list[float], t_span: tuple[float, float],
                dt: float = 0.01) -> tuple[list[float], list[list[float]]]:
        """Solve an ODE initial value problem using the classical Runge-Kutta method.

        Args:
            f (Callable): RHS function ``f(t, y) -> dy/dt``.
            y0 (List[float]): Initial state vector.
            t_span (Tuple[float, float]): (t_start, t_end).
            dt (float): Time step (default 0.01).

        Returns:
            Tuple[List[float], List[List[float]]]: Time points and solution trajectory.
        """
        t0, t1 = t_span
        n_steps = max(1, int((t1 - t0) / dt))
        dt = (t1 - t0) / n_steps
        ts = [t0 + i * dt for i in range(n_steps + 1)]
        ys = [y0[:]]
        y = y0[:]
        for t in ts[:-1]:
            k1 = f(t, y)
            k2 = f(t + dt / 2, [y[k] + dt / 2 * k1[k] for k in range(len(y))])
            k3 = f(t + dt / 2, [y[k] + dt / 2 * k2[k] for k in range(len(y))])
            k4 = f(t + dt, [y[k] + dt * k3[k] for k in range(len(y))])
            y = [y[k] + dt / 6 * (k1[k] + 2 * k2[k] + 2 * k3[k] + k4[k]) for k in range(len(y))]
            ys.append(y[:])
        return ts, ys
