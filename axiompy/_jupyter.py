"""Jupyter notebook integration for AxiomPy.

Provides ``_repr_html_`` / ``_repr_latex_`` on core types and optional interactive widgets
for live parameter exploration in Jupyter notebooks.

Widgets require ``ipywidgets`` (installed separately).
"""

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional interactive widgets
# ---------------------------------------------------------------------------

def _require_ipywidgets():
    try:
        import ipywidgets  # noqa: F401
    except ImportError:
        raise ImportError(
            "Interactive widgets require 'ipywidgets'. Install it with: "
            "pip install ipywidgets"
        )


class PolynomialSliders:
    """Interactive polynomial explorer with coefficient sliders.

    Displays a polynomial and its value at a given ``x``, updated live as
    sliders are adjusted.  Requires ``ipywidgets`` and a Jupyter notebook.

    Args:
        coeffs: Initial coefficient values (constant through highest degree).
        x0: Initial evaluation point.
    """

    def __init__(self, coeffs: list[float] | None = None, x0: float = 0.0):
        _require_ipywidgets()
        import ipywidgets as widgets
        from IPython.display import display

        coeffs = coeffs or [1.0, 0.0, 0.0]

        self._sliders = {}
        slider_box = widgets.VBox([])
        children = []
        for i, c in enumerate(coeffs):
            s = widgets.FloatSlider(
                value=c,
                min=-10.0,
                max=10.0,
                step=0.1,
                description=f"a{i}",
                continuous_update=True,
            )
            self._sliders[i] = s
            children.append(s)
        slider_box.children = children

        self._x_slider = widgets.FloatSlider(
            value=x0,
            min=-10.0,
            max=10.0,
            step=0.1,
            description="x",
            continuous_update=True,
        )

        self._output = widgets.HTML()

        def _update(change=None):
            poly_str = self._format_poly()
            val = self._evaluate()
            self._output.value = (
                f"<div style='font-size:14px;'>"
                f"<b>Polynomial:</b> {poly_str}<br>"
                f"<b>Value at x={self._x_slider.value}:</b> {val:.4f}"
                f"</div>"
            )

        for s in self._sliders.values():
            s.observe(_update, "value")
        self._x_slider.observe(_update, "value")

        _update()

        display(widgets.VBox([slider_box, self._x_slider, self._output]))

    def _format_poly(self) -> str:
        terms = []
        for i in sorted(self._sliders):
            c = self._sliders[i].value
            if c == 0:
                continue
            if i == 0:
                terms.append(f"{c:.2f}")
            elif i == 1:
                terms.append(f"{c:.2f}x" if c != 1 else "x")
            else:
                terms.append(f"{c:.2f}x^{i}" if c != 1 else f"x^{i}")
        return " + ".join(reversed(terms)) if terms else "0"

    def _evaluate(self) -> float:
        x = self._x_slider.value
        result = 0.0
        for i in sorted(self._sliders):
            result += self._sliders[i].value * (x ** i)
        return result


class MatrixExplorer:
    """Interactive 2x2 matrix explorer with element sliders.

    Shows the matrix, its determinant, trace, and eigenvalues updated live.

    Requires ``ipywidgets`` and a Jupyter notebook.
    """

    def __init__(self):
        _require_ipywidgets()
        import ipywidgets as widgets
        import numpy as np
        from IPython.display import display

        self._sliders = {}
        labels = [("a", 0, 0), ("b", 0, 1), ("c", 1, 0), ("d", 1, 1)]
        slider_list = []
        for name, _, _ in labels:
            s = widgets.FloatSlider(
                value=1.0 if name in ("a", "d") else 0.0,
                min=-10.0,
                max=10.0,
                step=0.1,
                description=name,
                continuous_update=True,
            )
            self._sliders[name] = s
            slider_list.append(s)

        self._output = widgets.HTML()

        def _update(change=None):
            a = self._sliders["a"].value
            b = self._sliders["b"].value
            c = self._sliders["c"].value
            d = self._sliders["d"].value
            det = a * d - b * c
            trace = a + d
            ev = np.linalg.eigvals(np.array([[a, b], [c, d]]))
            html = (
                f"<div style='font-size:14px;'>"
                f"<b>Matrix:</b> [ [{a:.2f}, {b:.2f}] ; [{c:.2f}, {d:.2f}] ]<br>"
                f"<b>Det:</b> {det:.4f}<br>"
                f"<b>Trace:</b> {trace:.4f}<br>"
                f"<b>Eigenvalues:</b> ({ev[0]:.4f}, {ev[1]:.4f})"
                f"</div>"
            )
            self._output.value = html

        for s in slider_list:
            s.observe(_update, "value")

        _update()
        display(widgets.VBox([widgets.HBox(slider_list), self._output]))
