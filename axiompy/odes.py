import math
from typing import Callable

RHSFunction = Callable[[float, list[float]], list[float]]


def _euler_step(f: RHSFunction, t: float, y: list[float], dt: float) -> list[float]:
    return [y[k] + dt * f(t, y)[k] for k in range(len(y))]


def _rk4_step(f: RHSFunction, t: float, y: list[float], dt: float) -> list[float]:
    k1 = f(t, y)
    k2 = f(t + dt / 2, [y[k] + dt / 2 * k1[k] for k in range(len(y))])
    k3 = f(t + dt / 2, [y[k] + dt / 2 * k2[k] for k in range(len(y))])
    k4 = f(t + dt, [y[k] + dt * k3[k] for k in range(len(y))])
    return [y[k] + dt / 6 * (k1[k] + 2 * k2[k] + 2 * k3[k] + k4[k]) for k in range(len(y))]


def _rk45_step(f: RHSFunction, t: float, y: list[float], dt: float
               ) -> tuple[list[float], list[float], float]:
    """Fehlberg RK45 embedded method (RKF45).  Returns (y4, y5, error_estimate)."""
    k1 = f(t, y)
    k2 = f(t + dt / 4, [y[k] + dt / 4 * k1[k] for k in range(len(y))])
    k3 = f(t + 3 * dt / 8, [y[k] + dt * (3 / 32 * k1[k] + 9 / 32 * k2[k]) for k in range(len(y))])
    k4 = f(t + 12 * dt / 13, [y[k] + dt * (1932 / 2197 * k1[k] - 7200 / 2197 * k2[k]
                                            + 7296 / 2197 * k3[k]) for k in range(len(y))])
    k5 = f(t + dt, [y[k] + dt * (439 / 216 * k1[k] - 8 * k2[k] + 3680 / 513 * k3[k]
                                 - 845 / 4104 * k4[k]) for k in range(len(y))])
    k6 = f(t + dt / 2, [y[k] + dt * (-8 / 27 * k1[k] + 2 * k2[k] - 3544 / 2565 * k3[k]
                                      + 1859 / 4104 * k4[k] - 11 / 40 * k5[k]) for k in range(len(y))])
    y4 = [y[k] + dt * (25 / 216 * k1[k] + 1408 / 2565 * k3[k] + 2197 / 4104 * k4[k]
                       - 1 / 5 * k5[k]) for k in range(len(y))]
    y5 = [y[k] + dt * (16 / 135 * k1[k] + 6656 / 12825 * k3[k] + 28561 / 56430 * k4[k]
                       - 9 / 50 * k5[k] + 2 / 55 * k6[k]) for k in range(len(y))]
    err = max(abs(y4[k] - y5[k]) for k in range(len(y)))
    return y4, y5, err


def _adams_bashforth_step(f: RHSFunction, t: float, y: list[float], dt: float,
                          history: list[list[float]]) -> list[float]:
    """2-step Adams-Bashforth using history with previous f evaluations."""
    fn = f(t, y)
    fn1 = history[-1] if history else fn
    return [y[k] + dt / 2 * (3 * fn[k] - fn1[k]) for k in range(len(y))]


def solve_ivp(f: RHSFunction, y0: list[float], t_span: tuple[float, float],
              method: str = 'rk4', dt: float = 0.01, rtol: float = 1e-6,
              atol: float = 1e-8) -> tuple[list[float], list[list[float]]]:
    """Solve an ODE initial value problem.

    Args:
        f: RHS function ``f(t, y) -> dy/dt``.
        y0: Initial state vector.
        t_span: (t_start, t_end).
        method: One of ``'euler'``, ``'rk4'``, ``'rk45'`` (adaptive), ``'adams_bashforth'``.
        dt: Time step (used as initial step for adaptive methods).
        rtol: Relative tolerance (rk45 only).
        atol: Absolute tolerance (rk45 only).

    Returns:
        Tuple[List[float], List[List[float]]]: Time points and solution trajectory.
    """
    t0, t1 = t_span
    ts: list[float] = [t0]
    ys: list[list[float]] = [y0[:]]
    y = y0[:]
    t = t0
    h = dt
    history: list[list[float]] = []

    if method == 'euler':
        n_steps = max(1, int((t1 - t0) / dt))
        h = (t1 - t0) / n_steps
        for _ in range(n_steps):
            y = _euler_step(f, t, y, h)
            t += h
            ts.append(t)
            ys.append(y[:])
        return ts, ys

    if method == 'rk4':
        n_steps = max(1, int((t1 - t0) / dt))
        h = (t1 - t0) / n_steps
        for _ in range(n_steps):
            y = _rk4_step(f, t, y, h)
            t += h
            ts.append(t)
            ys.append(y[:])
        return ts, ys

    if method == 'rk45':
        while t < t1:
            if t + h > t1:
                h = t1 - t
            y4, y5, err = _rk45_step(f, t, y, h)
            if err < atol + rtol * max(max(abs(v) for v in y4), 1.0):
                y = y5[:]
                t += h
                ts.append(t)
                ys.append(y[:])
            # error-based step size adjustment
            if err > 0:
                scale = (atol + rtol * max(max(abs(v) for v in y4), 1.0)) / err
                h *= min(2.0, max(0.1, 0.84 * scale ** 0.25))
        return ts, ys

    if method == 'adams_bashforth':
        n_steps = max(1, int((t1 - t0) / dt))
        h = (t1 - t0) / n_steps
        for _ in range(n_steps):
            fn = f(t, y)
            if len(history) < 1:
                y = _rk4_step(f, t, y, h)
            else:
                fn1 = history[-1]
                y = [y[k] + h / 2 * (3 * fn[k] - fn1[k]) for k in range(len(y))]
            history.append(fn)
            if len(history) > 2:
                history.pop(0)
            t += h
            ts.append(t)
            ys.append(y[:])
        return ts, ys

    raise ValueError(f"Unknown method '{method}'. Choose 'euler', 'rk4', 'rk45', or 'adams_bashforth'.")


def solve_bvp(f: RHSFunction, bc, x_span: tuple[float, float],
              guess: list[float], dt: float = 0.01,
              tol: float = 1e-6, max_iter: int = 100
              ) -> tuple[list[float], list[list[float]]]:
    """Solve a boundary value problem via shooting method.

    Uses Newton iteration to adjust initial conditions so that the
    boundary condition ``bc(y_at_x1) ≈ 0`` is satisfied.

    Args:
        f: RHS function ``f(x, y) -> dy/dx``.
        bc: Boundary condition function ``bc(y_at_end) -> list[float]``
            (should be zero vector at solution).
        x_span: (x_start, x_end).
        guess: Initial guess for y at x_start.
        dt: Step size for IVP integration.
        tol: Tolerance for boundary condition residual.
        max_iter: Maximum Newton iterations.

    Returns:
        Tuple[List[float], List[List[float]]]: x-points and solution trajectory.
    """
    import numpy as np

    n = len(guess)
    y_start = guess[:]
    for _ in range(max_iter):
        ts, ys = solve_ivp(f, y_start, x_span, method='rk4', dt=dt)
        y_end = ys[-1][:]
        residual = bc(y_end)
        resnorm = math.sqrt(sum(r * r for r in residual))
        if resnorm < tol:
            return ts, ys
        # Numeric Jacobian of bc w.r.t. y_start
        J = np.zeros((n, n))
        eps = 1e-6
        for j in range(n):
            yp = y_start[:]
            yp[j] += eps
            _, ysp = solve_ivp(f, yp, x_span, method='rk4', dt=dt)
            bcp = bc(ysp[-1][:])
            J[:, j] = [(bcp[i] - residual[i]) / eps for i in range(n)]
        try:
            delta = np.linalg.solve(J, [-r for r in residual])
        except np.linalg.LinAlgError:
            break
        y_start = [y_start[k] + delta[k] for k in range(n)]
    return solve_ivp(f, guess, x_span, method='rk4', dt=dt)


# ---- example system factories -----------------------------------------------

def pendulum_odes(g: float = 9.81, L: float = 1.0, b: float = 0.0
                  ) -> Callable[[float, list[float]], list[float]]:
    """Return the RHS for a damped pendulum.

    ``dy/dt = [theta_dot, -g/L * sin(theta) - b * theta_dot]``

    Args:
        g: Gravitational acceleration (default 9.81).
        L: Pendulum length (default 1.0).
        b: Damping coefficient (default 0.0).

    Returns:
        RHS function ``f(t, [theta, omega])``.
    """
    return lambda t, y: [y[1], -(g / L) * math.sin(y[0]) - b * y[1]]


def lotka_volterra_odes(alpha: float = 1.0, beta: float = 0.1,
                        gamma: float = 1.5, delta: float = 0.075
                        ) -> Callable[[float, list[float]], list[float]]:
    """Return the RHS for the Lotka-Volterra predator-prey model.

    ``dy/dt = [alpha * prey - beta * prey * pred,
              -gamma * pred + delta * prey * pred]``

    Args:
        alpha: Prey growth rate (default 1.0).
        beta: Predation rate (default 0.1).
        gamma: Predator death rate (default 1.5).
        delta: Predator reproduction rate (default 0.075).

    Returns:
        RHS function ``f(t, [prey, pred])``.
    """
    return lambda t, y: [alpha * y[0] - beta * y[0] * y[1],
                         -gamma * y[1] + delta * y[0] * y[1]]
