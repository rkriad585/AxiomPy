from typing import ClassVar, Optional

import numpy as np


class Visualization:
    """ASCII-based plotting utilities for data and vector fields."""

    _MARKERS: ClassVar[list[str]] = ['*', 'o', '+', 'x', '#', '@', '%', '&']

    @staticmethod
    def plot_ascii(x_data, y_data, width: int = 60, height: int = 15,
                   title: Optional[str] = None, xlabel: Optional[str] = None,
                   ylabel: Optional[str] = None,
                   extra_series: Optional[list] = None):
        """Render an ASCII scatter plot of x_data vs y_data.

        Args:
            x_data: Sequence of x-coordinates.
            y_data: Sequence of y-coordinates.
            width (int): Character width of the plot.
            height (int): Character height of the plot.
            title (str, optional): Plot title printed above the chart.
            xlabel (str, optional): Label for the x-axis.
            ylabel (str, optional): Label for the y-axis.
            extra_series (list, optional): List of (x_vals, y_vals, label) tuples
                for additional series plotted with different markers.
        """
        if not x_data or not y_data:
            return
        if title:
            print(f"{title:^{width + 10}}")
        series = [(x_data, y_data, '', Visualization._MARKERS[0])]
        if extra_series:
            for i, (xs, ys, lbl) in enumerate(extra_series):
                series.append((xs, ys, lbl, Visualization._MARKERS[(i + 1) % len(Visualization._MARKERS)]))
        all_x = [v for s in series for v in s[0]]
        all_y = [v for s in series for v in s[1]]
        if not all_x or not all_y:
            return
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        if max_x == min_x:
            max_x = min_x + 1
        if max_y == min_y:
            max_y = min_y + 1
        plot = [[' ' for _ in range(width)] for _ in range(height)]
        for xs, ys, _, marker in series:
            for x, y in zip(xs, ys):
                px = int((x - min_x) / (max_x - min_x) * (width - 1))
                py = int((y - min_y) / (max_y - min_y) * (height - 1))
                plot[height - 1 - py][px] = marker
        lines = ["".join(row) for row in plot]
        y_axis = [f"{max_y:8.2f} |", " " * 8 + "|", f"{min_y:8.2f} |"]
        for i, line in enumerate(lines):
            prefix = y_axis[0] if i == 0 else (y_axis[2] if i == height - 1 else y_axis[1])
            print(prefix + line)
        x_offset = 8
        print(" " * x_offset + "-" * (width + 1))
        if xlabel:
            print(f"{'':8} {min_x:<8.2f}{'':^{width - 20}}{max_x:>8.2f}")
            print(f"{'':^{x_offset + width // 2}}{xlabel}")
        else:
            print(f"{'':8} {min_x:<8.2f}{'':^{width - 20}}{max_x:>8.2f}")
        if ylabel:
            print(f"{'':^{x_offset // 2}}{ylabel}")
        if extra_series:
            legend_parts = []
            for i, (_, _, lbl, _) in enumerate(series):
                if lbl:
                    legend_parts.append(f"{Visualization._MARKERS[i]} = {lbl}")
            if legend_parts:
                print(f"{'':8}{' | '.join(legend_parts)}")

    @staticmethod
    def plot_histogram(data, bins: int = 10, width: int = 40, title: Optional[str] = None):
        """Render an ASCII histogram of data.

        Args:
            data: Sequence of numeric values.
            bins (int): Number of bins.
            width (int): Character width of the longest bar.
            title (str, optional): Plot title.
        """
        if not data:
            return
        if title:
            print(f"{title:^{width + 10}}")
        counts, edges = np.histogram(data, bins=bins)
        max_count = max(counts) if counts.max() > 0 else 1
        for i, count in enumerate(counts):
            bar_len = int(count / max_count * width)
            bar = '#' * bar_len
            print(f"{edges[i]:8.2f}-{edges[i + 1]:>8.2f} |{bar} {count}")

    @staticmethod
    def plot_bar(labels, values, width: int = 40, title: Optional[str] = None):
        """Render an ASCII horizontal bar chart.

        Args:
            labels: Sequence of label strings.
            values: Sequence of numeric values.
            width (int): Character width of the longest bar.
            title (str, optional): Plot title.
        """
        if not labels or not values:
            return
        if title:
            print(f"{title:^{width + 20}}")
        max_val = max(values) if max(values) > 0 else 1
        max_label_len = max(len(str(l)) for l in labels)
        for label, val in zip(labels, values):
            bar_len = int(val / max_val * width)
            bar = '#' * bar_len
            print(f"{label!s:>{max_label_len}} |{bar} {val:.2f}")

    @staticmethod
    def plot_scatter(xs, ys, width: int = 40, height: int = 12, title: Optional[str] = None):
        """Render an ASCII scatter plot with compact dimensions.

        Args:
            xs: Sequence of x-coordinates.
            ys: Sequence of y-coordinates.
            width (int): Character width.
            height (int): Character height.
            title (str, optional): Plot title.
        """
        Visualization.plot_ascii(xs, ys, width=width, height=height, title=title)

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
