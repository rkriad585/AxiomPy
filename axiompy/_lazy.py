"""Lazy evaluation — deferred computation graphs for Vector and Matrix.

Usage::

    from axiompy import Axiom

    v = Axiom.Vector([1, 2, 3])
    w = Axiom.Vector([4, 5, 6])

    expr = Axiom.lazy(v) + Axiom.lazy(w) * 2.0
    result = expr.compute()     # → Vector([9.0, 12.0, 15.0])
"""
import logging
from numbers import Number
from typing import Optional, Union

import numpy as np

from .matrix import Matrix
from .vector import Vector

logger = logging.getLogger(__name__)

LazyValue = Union['LazyExpr', Vector, Matrix, float]


def _promote_scalar(val, like):
    """If val is a scalar, create a Vector/Matrix of the same type/shape as *like*."""
    if isinstance(val, (int, float)):
        if isinstance(like, Vector):
            return Vector(np.full(like._data.shape, val))
        if isinstance(like, Matrix):
            return Matrix(np.full(like._data.shape, val))
    return val


class _Op:
    """Operation type enum for expression nodes."""
    CONSTANT = 'constant'
    ADD = 'add'
    SUB = 'sub'
    MUL = 'mul'          # scalar mul or element-wise
    MATMUL = 'matmul'    # @
    NEG = 'neg'
    TRANSPOSE = 'T'
    POWER = 'pow'


class LazyExpr:
    """A node in a deferred computation graph for Vector/Matrix operations.

    Build an expression by applying operators::

        expr = LazyExpr.constant(v) + LazyExpr.constant(w)
        result = expr.compute()
    """

    def __init__(self, op: str, value=None,
                 left: Optional['LazyExpr'] = None,
                 right: Optional['LazyExpr'] = None):
        self._op = op
        self._value = value        # used only for CONSTANT and scalar args
        self._left = left
        self._right = right

    # ---- factories ---------------------------------------------------------

    @staticmethod
    def constant(value: Union[Vector, Matrix, float]) -> 'LazyExpr':
        """Wrap a concrete value as a leaf node."""
        return LazyExpr(_Op.CONSTANT, value=value)

    @staticmethod
    def add(a: 'LazyExpr', b: 'LazyExpr') -> 'LazyExpr':
        return LazyExpr(_Op.ADD, left=a, right=b)

    @staticmethod
    def sub(a: 'LazyExpr', b: 'LazyExpr') -> 'LazyExpr':
        return LazyExpr(_Op.SUB, left=a, right=b)

    @staticmethod
    def mul(a: 'LazyExpr', b: 'LazyExpr') -> 'LazyExpr':
        return LazyExpr(_Op.MUL, left=a, right=b)

    @staticmethod
    def matmul(a: 'LazyExpr', b: 'LazyExpr') -> 'LazyExpr':
        return LazyExpr(_Op.MATMUL, left=a, right=b)

    @staticmethod
    def neg(a: 'LazyExpr') -> 'LazyExpr':
        return LazyExpr(_Op.NEG, left=a)

    @staticmethod
    def transpose(a: 'LazyExpr') -> 'LazyExpr':
        return LazyExpr(_Op.TRANSPOSE, left=a)

    @staticmethod
    def power(a: 'LazyExpr', exponent: Number) -> 'LazyExpr':
        return LazyExpr(_Op.POWER, value=exponent, left=a)

    # ---- operators ---------------------------------------------------------

    def __add__(self, other):
        if isinstance(other, (Vector, Matrix, float, int)):
            other = LazyExpr.constant(other)
        return LazyExpr.add(self, other)

    def __radd__(self, other):
        return LazyExpr.add(LazyExpr.constant(other), self)

    def __sub__(self, other):
        if isinstance(other, (Vector, Matrix, float, int)):
            other = LazyExpr.constant(other)
        return LazyExpr.sub(self, other)

    def __rsub__(self, other):
        return LazyExpr.sub(LazyExpr.constant(other), self)

    def __mul__(self, other):
        if isinstance(other, (Vector, Matrix, float, int)):
            other = LazyExpr.constant(other)
        return LazyExpr.mul(self, other)

    def __rmul__(self, other):
        return LazyExpr.mul(LazyExpr.constant(other), self)

    def __matmul__(self, other):
        if isinstance(other, (Vector, Matrix)):
            other = LazyExpr.constant(other)
        return LazyExpr.matmul(self, other)

    def __neg__(self):
        return LazyExpr.neg(self)

    @property
    def T(self):
        return LazyExpr.transpose(self)

    def __pow__(self, exponent: Number):
        return LazyExpr.power(self, exponent)

    # ---- materialization ---------------------------------------------------

    def compute(self):
        """Walk the expression graph bottom-up and return the concrete result.

        Returns:
            Vector, Matrix, or float: The materialized value.
        """
        if self._op == _Op.CONSTANT:
            return self._value

        left_val = self._left.compute() if self._left else None
        right_val = self._right.compute() if self._right else None

        if self._op == _Op.ADD:
            left_val = _promote_scalar(left_val, right_val)
            right_val = _promote_scalar(right_val, left_val)
            return left_val + right_val
        elif self._op == _Op.SUB:
            left_val = _promote_scalar(left_val, right_val)
            right_val = _promote_scalar(right_val, left_val)
            return left_val - right_val
        elif self._op == _Op.MUL:
            return left_val * right_val
        elif self._op == _Op.MATMUL:
            return left_val @ right_val
        elif self._op == _Op.NEG:
            return -left_val
        elif self._op == _Op.TRANSPOSE:
            return left_val.T
        elif self._op == _Op.POWER:
            return left_val ** self._value
        else:
            raise ValueError(f"Unknown operation: {self._op}")

    def __repr__(self):
        if self._op == _Op.CONSTANT:
            val = self._value
            if isinstance(val, (Vector, Matrix)):
                return f"Lazy({type(val).__name__}({val._data.shape}))"
            return f"Lazy({val!r})"
        if self._op in (_Op.NEG, _Op.TRANSPOSE):
            return f"({self._op}{self._left!r})"
        if self._op == _Op.POWER:
            return f"({self._left!r} ** {self._value})"
        return f"({self._left!r} {self._op} {self._right!r})"


# ---- Convenience wrapper ---------------------------------------------------


class LazyScope:
    """Namespace exposing :func:`lazy` and :class:`LazyExpr`."""

    @staticmethod
    def lazy(value: Union[Vector, Matrix, float]) -> LazyExpr:
        """Wrap a Vector, Matrix, or scalar for lazy evaluation.

        Args:
            value: The value to wrap.

        Returns:
            LazyExpr: A leaf node in the computation graph.
        """
        return LazyExpr.constant(value)

    @staticmethod
    def compute(expr: LazyExpr):
        """Materialize a lazy expression.

        Args:
            expr: The lazy expression to evaluate.

        Returns:
            Vector, Matrix, or float: The computed result.
        """
        return expr.compute()


# Singleton for facade
_lazy_scope = LazyScope()
