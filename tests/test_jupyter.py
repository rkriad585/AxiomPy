"""Tests for Jupyter integration (``_repr_html_``, ``_repr_latex_``, widgets)."""

import pytest

from axiompy import Axiom

A = Axiom


class TestReprLatex:
    def test_vector_latex(self):
        v = A.Vector([1.5, 2.0, 3.0])
        latex = v.to_latex()
        assert latex.startswith("\\begin{pmatrix}")
        assert latex.endswith("\\end{pmatrix}")
        assert "1.5" in latex
        assert "2" in latex

    def test_vector_repr_latex(self):
        v = A.Vector([1.0, 2.0])
        out = v._repr_latex_()
        assert out.startswith("$$")
        assert out.endswith("$$")

    def test_vector_repr_html(self):
        v = A.Vector([1.0, 2.0, 3.0])
        html = v._repr_html_()
        assert html.startswith("<table")
        assert "1.0" in html
        assert "2.0" in html
        assert "3.0" in html

    def test_matrix_latex(self):
        m = A.Matrix([[1.0, 2.0], [3.0, 4.0]])
        latex = m.to_latex()
        assert "begin{pmatrix}" in latex
        assert "1" in latex
        assert "4" in latex

    def test_matrix_repr_latex(self):
        m = A.Matrix([[1.0, 2.0], [3.0, 4.0]])
        out = m._repr_latex_()
        assert out.startswith("$$")
        assert "begin{pmatrix}" in out

    def test_matrix_repr_html(self):
        m = A.Matrix([[1.0, 2.0], [3.0, 4.0]])
        html = m._repr_html_()
        assert html.startswith("<table")
        assert "1.0" in html
        assert "4.0" in html

    def test_polynomial_latex(self):
        p = A.Polynomial([1, 2, 3])
        latex = p.to_latex()
        assert "x^{2}" in latex
        assert "3" in latex

    def test_polynomial_repr_latex(self):
        p = A.Polynomial([1, 2, 3])
        out = p._repr_latex_()
        assert out.startswith("$$")
        assert "x^{2}" in out

    def test_polynomial_repr_html(self):
        p = A.Polynomial([1, 2, 3])
        html = p._repr_html_()
        assert "x<sup>2</sup>" in html
        assert "sup" in html

    def test_polynomial_constant_latex(self):
        p = A.Polynomial([5.0])
        assert p.to_latex() == "5.0"

    def test_polynomial_zero_latex(self):
        p = A.Polynomial([0])
        assert p.to_latex() == "0"

    def test_vector_float_precision(self):
        # default precision is 6 digits → 1.23457
        v = A.Vector([1.23456789])
        latex = v.to_latex()
        assert "1.234568" in latex

    def test_sparse_latex(self):
        m = A.Matrix([[1.0, 0.0], [0.0, 0.0]])
        s = A.SparseMatrix.from_dense(m)
        latex = s.to_latex()
        assert "begin{pmatrix}" in latex
        assert "1" in latex

    def test_sparse_repr_latex(self):
        m = A.Matrix([[1.0, 0.0], [0.0, 0.0]])
        s = A.SparseMatrix.from_dense(m)
        out = s._repr_latex_()
        assert out.startswith("$$")

    def test_sparse_repr_html(self):
        m = A.Matrix([[1.0, 0.0], [0.0, 0.0]])
        s = A.SparseMatrix.from_dense(m)
        html = s._repr_html_()
        assert html.startswith("<table")
        assert "1.0" in html


class TestWidgets:
    def test_polynomial_sliders_requires_ipywidgets(self):
        with pytest.raises(ImportError, match="ipywidgets"):
            A.PolynomialSliders()

    def test_matrix_explorer_requires_ipywidgets(self):
        with pytest.raises(ImportError, match="ipywidgets"):
            A.MatrixExplorer()
