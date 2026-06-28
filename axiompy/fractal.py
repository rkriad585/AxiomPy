import math
from typing import Callable


def mandelbrot(width: int, height: int,
               x_range: tuple[float, float] = (-2.0, 1.0),
               y_range: tuple[float, float] = (-1.5, 1.5),
               max_iter: int = 100) -> list[list[int]]:
    """Compute the Mandelbrot set on a grid.

    Returns a 2D list of iteration counts (0 = in set).

    Args:
        width: Number of columns (x-axis pixels).
        height: Number of rows (y-axis pixels).
        x_range: (xmin, xmax) in the complex plane.
        y_range: (ymin, ymax) in the complex plane.
        max_iter: Maximum iterations per point.

    Returns:
        List[list[int]]: ``grid[y][x]`` = iteration count.
    """
    xmin, xmax = x_range
    ymin, ymax = y_range
    grid: list[list[int]] = []
    for py in range(height):
        y = ymin + (ymax - ymin) * py / (height - 1)
        row: list[int] = []
        for px in range(width):
            x = xmin + (xmax - xmin) * px / (width - 1)
            zx, zy = 0.0, 0.0
            i = 0
            while i < max_iter and zx * zx + zy * zy < 4.0:
                zx, zy = zx * zx - zy * zy + x, 2.0 * zx * zy + y
                i += 1
            row.append(i)
        grid.append(row)
    return grid


def julia(c: complex, width: int, height: int,
          x_range: tuple[float, float] = (-1.5, 1.5),
          y_range: tuple[float, float] = (-1.5, 1.5),
          max_iter: int = 100) -> list[list[int]]:
    """Compute the Julia set for parameter ``c`` on a grid.

    Args:
        c: Complex parameter.
        width, height: Grid dimensions.
        x_range, y_range: Coordinate ranges.
        max_iter: Maximum iterations.

    Returns:
        List[list[int]]: ``grid[y][x]`` = iteration count.
    """
    xmin, xmax = x_range
    ymin, ymax = y_range
    grid: list[list[int]] = []
    cr, ci = c.real, c.imag
    for py in range(height):
        y = ymin + (ymax - ymin) * py / (height - 1)
        row: list[int] = []
        for px in range(width):
            x = xmin + (xmax - xmin) * px / (width - 1)
            zx, zy = x, y
            i = 0
            while i < max_iter and zx * zx + zy * zy < 4.0:
                zx, zy = zx * zx - zy * zy + cr, 2.0 * zx * zy + ci
                i += 1
            row.append(i)
        grid.append(row)
    return grid


def logistic_map(r: float, x0: float, n: int) -> list[float]:
    """Iterate the logistic map ``x_{n+1} = r * x_n * (1 - x_n)``.

    Args:
        r: Growth parameter.
        x0: Initial condition in [0, 1].
        n: Number of iterations.

    Returns:
        List[float]: Orbit of length ``n + 1`` (including x0).
    """
    xs = [x0]
    x = x0
    for _ in range(n):
        x = r * x * (1.0 - x)
        xs.append(x)
    return xs


def bifurcation_diagram(r_start: float, r_end: float, x0: float = 0.5,
                        n_transient: int = 100, n_plot: int = 100,
                        r_steps: int = 500) -> list[tuple[float, float]]:
    """Compute points for a bifurcation diagram of the logistic map.

    Returns a list of ``(r, x)`` points to plot.

    Args:
        r_start: Minimum r value.
        r_end: Maximum r value.
        x0: Initial condition.
        n_transient: Transient iterations to discard.
        n_plot: Iterations to keep per r value.
        r_steps: Number of r values.

    Returns:
        List[(float, float)]: Bifurcation points.
    """
    points: list[tuple[float, float]] = []
    for i in range(r_steps):
        r = r_start + (r_end - r_start) * i / (r_steps - 1)
        x = x0
        for _ in range(n_transient):
            x = r * x * (1.0 - x)
        for _ in range(n_plot):
            x = r * x * (1.0 - x)
            points.append((r, x))
    return points


def lyapunov_exponent(f: Callable[[float], float], x0: float, n: int = 1000) -> float:
    """Estimate the Lyapunov exponent of a 1-D map via orbit.

    ``λ ≈ (1/n) Σ log|f'(x_k)|``

    Args:
        f: 1-D map function (callable returning next value).
        x0: Initial condition.
        n: Number of iterations.

    Returns:
        float: Estimated Lyapunov exponent.
    """
    x = x0
    lam = 0.0
    h = 1e-8
    for _ in range(n):
        fx = f(x)
        df = (f(x + h) - fx) / h
        if df != 0:
            lam += math.log(abs(df))
        x = fx
    return lam / n
