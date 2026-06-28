import numpy as np


class Visualization:
    """ASCII-based plotting utilities for data and vector fields."""

    @staticmethod
    def plot_ascii(x_data, y_data, width: int = 60, height: int = 15):
        """Render an ASCII scatter plot of x_data vs y_data.

        Args:
            x_data: Sequence of x-coordinates.
            y_data: Sequence of y-coordinates.
            width (int): Character width of the plot.
            height (int): Character height of the plot.
        """
        if not x_data or not y_data:
            return
        min_x, max_x = min(x_data), max(x_data)
        min_y, max_y = min(y_data), max(y_data)
        plot = [[' ' for _ in range(width)] for _ in range(height)]
        for x, y in zip(x_data, y_data):
            px = int((x - min_x) / (max_x - min_x) * (width - 1)) if max_x > min_x else 0
            py = int((y - min_y) / (max_y - min_y) * (height - 1)) if max_y > min_y else 0
            plot[height - 1 - py][px] = '*'
        lines = ["".join(row) for row in plot]
        y_axis = [
            f"{max_y:8.2f} |",
            " " * 8 + "|",
            f"{min_y:8.2f} |",
        ]
        for i, line in enumerate(lines):
            prefix = y_axis[0] if i == 0 else (y_axis[2] if i == height - 1 else y_axis[1])
            print(prefix + line)
        print(" " * 8 + "-" * (width + 1))
        print(f"{'':8} {min_x:<8.2f}{'':^{width-20}}{max_x:>8.2f}")

    @staticmethod
    def plot_field_ascii(field_fn, center, size, width=30, height=15):
        """Render an ASCII direction plot of a 2D vector field.

        Args:
            field_fn: Callable that takes a point and returns a vector.
            center: Center of the field region.
            size: Side length of the square region.
            width (int): Character width of the plot.
            height (int): Character height of the plot.
        """
        arrows = {(1, 0): '>', (-1, 0): '<', (0, 1): '^', (0, -1): 'v'}
        for j in range(height, 0, -1):
            line = ""
            for i in range(width):
                x = center[0] - size / 2 + i * size / width
                y = center[1] - size / 2 + j * size / height
                E = field_fn([x, y, 0] if len(center) == 3 else [x, y])
                norm_E = np.array(E) / (np.linalg.norm(E) + 1e-9)
                key = (round(norm_E[0]), round(norm_E[1]))
                line += arrows.get(key, '.')
            print(f"  {line}")
