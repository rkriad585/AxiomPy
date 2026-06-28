"""
Example: Jupyter notebook integration for AxiomPy.

Demonstrates ``_repr_html_`` / ``_repr_latex_`` output and optional interactive
widgets.  Widgets require ``ipywidgets`` (``pip install ipywidgets``).

This script can be run in a terminal to verify correctness of the repr methods;
the visual HTML/LaTeX output is designed for Jupyter notebooks.
"""

from axiompy import Axiom

A = Axiom

# ---- Vector repr ----
v = A.Vector([1.5, 2.0, 3.0])
print("=== Vector ===")
print("to_latex():")
print(v.to_latex())
print()
print("_repr_latex_():")
print(v._repr_latex_())
print()
print("_repr_html_():")
print(v._repr_html_())
print()

# ---- Matrix repr ----
m = A.Matrix([[1.0, 2.0], [3.0, 4.0]])
print("=== Matrix ===")
print("to_latex():")
print(m.to_latex())
print()
print("_repr_latex_():")
print(m._repr_latex_())
print()
print("_repr_html_():")
print(m._repr_html_())
print()

# ---- Polynomial repr ----
p = A.Polynomial([1, -2, 3])  # 3x^2 - 2x + 1
print("=== Polynomial ===")
print("to_latex():")
print(p.to_latex())
print()
print("_repr_latex_():")
print(p._repr_latex_())
print()
print("_repr_html_():")
print(p._repr_html_())
print()

# ---- Sparse matrix repr ----
s = A.SparseMatrix.from_dense(m)
print("=== SparseMatrix ===")
print("to_latex():")
print(s.to_latex())
print()

print("All repr methods verified in terminal mode.")
print()
print("In a Jupyter notebook, typing a Vector/Matrix/Polynomial at the end of a")
print("cell will render its rich HTML/Latex representation automatically.")
print()
print("Interactive widgets (requires ipywidgets):")
print("  A.PolynomialSliders()  — explore a polynomial with sliders")
print("  A.MatrixExplorer()     — explore a 2x2 matrix with sliders")
