import logging
import math
from numbers import Number
from typing import Optional

logger = logging.getLogger(__name__)


class AutoDiff:
    """Automatic differentiation engine with a Variable type and common functions."""

    class Variable:
        """A scalar variable that tracks its value and gradient for reverse-mode autodiff."""

        def __init__(self, value: float, _children: Optional[set] = None, _op: str = ''):
            """Initialize a Variable with a value and optional computational graph links.

            Args:
                value (float): The scalar value.
                _children (set, optional): Set of parent Variables in the computation graph.
                _op (str): The operation that produced this variable.
            """
            self.value = value
            self.grad = 0.0
            self._backward = lambda: None
            self._prev = _children if _children is not None else set()

        def __repr__(self):
            """Return a string representation of the Variable.

            Returns:
                str: A string showing value and gradient.
            """
            return f"Variable(value={self.value:.4f}, grad={self.grad:.4f})"

        def __add__(self, other):
            """Add two Variables or a Variable and a scalar.

            Args:
                other (AutoDiff.Variable or Number): The value to add.

            Returns:
                AutoDiff.Variable: The result of the addition.
            """
            other = other if isinstance(other, AutoDiff.Variable) else AutoDiff.Variable(other)
            out = AutoDiff.Variable(self.value + other.value, {self, other})
            def _backward():
                self.grad += out.grad
                other.grad += out.grad
            out._backward = _backward
            return out

        def __mul__(self, other):
            """Multiply two Variables or a Variable and a scalar.

            Args:
                other (AutoDiff.Variable or Number): The value to multiply.

            Returns:
                AutoDiff.Variable: The result of the multiplication.
            """
            other = other if isinstance(other, AutoDiff.Variable) else AutoDiff.Variable(other)
            out = AutoDiff.Variable(self.value * other.value, {self, other})
            def _backward():
                self.grad += other.value * out.grad
                other.grad += self.value * out.grad
            out._backward = _backward
            return out

        def __pow__(self, other: Number):
            """Raise the Variable to a scalar power.

            Args:
                other (Number): The exponent.

            Returns:
                AutoDiff.Variable: The result of the power operation.
            """
            out = AutoDiff.Variable(self.value ** other, {self})
            def _backward():
                self.grad += (other * self.value ** (other - 1)) * out.grad
            out._backward = _backward
            return out

        def __neg__(self):
            """Negate the Variable.

            Returns:
                AutoDiff.Variable: The negated Variable.
            """
            return self * -1

        def __sub__(self, other):
            """Subtract another value from this Variable.

            Args:
                other (AutoDiff.Variable or Number): The value to subtract.

            Returns:
                AutoDiff.Variable: The result of the subtraction.
            """
            return self + (-other)

        def __truediv__(self, other):
            """Divide this Variable by another value.

            Args:
                other (AutoDiff.Variable or Number): The divisor.

            Returns:
                AutoDiff.Variable: The result of the division.
            """
            return self * other ** -1

        def __radd__(self, other):
            """Handle scalar + Variable.

            Args:
                other (Number): The scalar value.

            Returns:
                AutoDiff.Variable: The result of the addition.
            """
            return self + other

        def __rmul__(self, other):
            """Handle scalar * Variable.

            Args:
                other (Number): The scalar value.

            Returns:
                AutoDiff.Variable: The result of the multiplication.
            """
            return self * other

        def __rsub__(self, other):
            """Handle scalar - Variable.

            Args:
                other (Number): The scalar value.

            Returns:
                AutoDiff.Variable: The result of the subtraction.
            """
            return other + (-self)

        def __rtruediv__(self, other):
            """Handle scalar / Variable.

            Args:
                other (Number): The scalar value.

            Returns:
                AutoDiff.Variable: The result of the division.
            """
            return other * self ** -1

        def backward(self):
            """Compute gradients via reverse-mode automatic differentiation."""
            topo, visited = [], set()
            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for c in v._prev:
                        build_topo(c)
                    topo.append(v)
            build_topo(self)
            self.grad = 1.0
            for v in reversed(topo):
                v._backward()

        def tanh(self):
            """Compute the hyperbolic tangent (instance method).

            Returns:
                AutoDiff.Variable: The hyperbolic tangent.
            """
            return AutoDiff.tanh(self)

        def sigmoid(self):
            """Compute the sigmoid (instance method).

            Returns:
                AutoDiff.Variable: The sigmoid.
            """
            return AutoDiff.sigmoid(self)

        def to_ascii(self, indent: str = "") -> str:
            """Render the computational graph as ASCII.

            Args:
                indent (str): Indentation prefix (default "").

            Returns:
                str: ASCII representation of the graph.
            """
            lines = []
            visited = set()
            def dfs(v, d):
                if v in visited:
                    return
                visited.add(v)
                prefix = indent + "  " * d
                lines.append(f"{prefix}Variable(value={v.value:.4f}, grad={v.grad:.4f})")
                for child in v._prev:
                    dfs(child, d + 1)
            dfs(self, 0)
            return "\n".join(reversed(lines))

    @staticmethod
    def sin(v: 'AutoDiff.Variable'):
        """Compute the sine of a Variable.

        Args:
            v (AutoDiff.Variable): The input Variable.

        Returns:
            AutoDiff.Variable: The sine of v.
        """
        out = AutoDiff.Variable(math.sin(v.value), {v})
        def _backward():
            v.grad += math.cos(v.value) * out.grad
        out._backward = _backward
        return out

    @staticmethod
    def exp(v: 'AutoDiff.Variable'):
        """Compute the exponential of a Variable.

        Args:
            v (AutoDiff.Variable): The input Variable.

        Returns:
            AutoDiff.Variable: e raised to the power of v.
        """
        out = AutoDiff.Variable(math.exp(v.value), {v})
        def _backward():
            v.grad += out.value * out.grad
        out._backward = _backward
        return out

    @staticmethod
    def log(v: 'AutoDiff.Variable'):
        """Compute the natural logarithm of a Variable.

        Args:
            v (AutoDiff.Variable): The input Variable (must be positive).

        Returns:
            AutoDiff.Variable: The natural logarithm of v.
        """
        out = AutoDiff.Variable(math.log(v.value), {v})
        def _backward():
            v.grad += (1 / v.value) * out.grad
        out._backward = _backward
        return out

    @staticmethod
    def tanh(v: 'AutoDiff.Variable'):
        """Compute the hyperbolic tangent of a Variable.

        Args:
            v (AutoDiff.Variable): The input Variable.

        Returns:
            AutoDiff.Variable: The hyperbolic tangent of v.
        """
        t = math.tanh(v.value)
        out = AutoDiff.Variable(t, {v})
        def _backward():
            v.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    @staticmethod
    def sigmoid(v: 'AutoDiff.Variable'):
        """Compute the sigmoid of a Variable.

        Args:
            v (AutoDiff.Variable): The input Variable.

        Returns:
            AutoDiff.Variable: The sigmoid of v.
        """
        s = 1 / (1 + math.exp(-v.value))
        out = AutoDiff.Variable(s, {v})
        def _backward():
            v.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    @staticmethod
    def sqrt(v: 'AutoDiff.Variable'):
        """Compute the square root of a Variable.

        Args:
            v (AutoDiff.Variable): The input Variable (must be non-negative).

        Returns:
            AutoDiff.Variable: The square root of v.
        """
        s = math.sqrt(v.value)
        out = AutoDiff.Variable(s, {v})
        def _backward():
            v.grad += (0.5 / s) * out.grad
        out._backward = _backward
        return out

    @staticmethod
    def gradient_descent(fn, start: float, lr: float = 0.01,
                         steps: int = 100) -> float:
        """Minimize a scalar function using gradient descent.

        Args:
            fn (callable): A function mapping a Variable to a scalar Variable.
            start (float): The initial value.
            lr (float): The learning rate (default 0.01).
            steps (int): Number of gradient descent steps (default 100).

        Returns:
            float: The value that approximately minimizes the function.
        """
        x = AutoDiff.Variable(start)
        for _ in range(steps):
            loss = fn(x)
            loss.backward()
            x.value -= lr * x.grad
            x.grad = 0.0
        return x.value

    @staticmethod
    def adam(fn, start: float, lr: float = 0.01,
             steps: int = 100, beta1: float = 0.9,
             beta2: float = 0.999, eps: float = 1e-8) -> float:
        """Minimize a scalar function using the Adam optimizer.

        Args:
            fn (callable): A function mapping a Variable to a scalar Variable.
            start (float): The initial value.
            lr (float): The learning rate (default 0.01).
            steps (int): Number of optimization steps (default 100).
            beta1 (float): Exponential decay for first moment (default 0.9).
            beta2 (float): Exponential decay for second moment (default 0.999).
            eps (float): Small constant for numerical stability (default 1e-8).

        Returns:
            float: The value that approximately minimizes the function.
        """
        x = AutoDiff.Variable(start)
        m, v = 0.0, 0.0
        for t in range(1, steps + 1):
            loss = fn(x)
            loss.backward()
            g = x.grad
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g * g
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            x.value -= lr * m_hat / (math.sqrt(v_hat) + eps)
            x.grad = 0.0
        return x.value

    @staticmethod
    def hessian(fn, x0: float, h: float = 1e-5) -> float:
        """Compute the second derivative (Hessian) of a scalar function.

        Uses a forward-over-reverse approach: computes the gradient via autodiff,
        then differentiates numerically.

        Args:
            fn (callable): A function mapping a Variable to a scalar Variable.
            x0 (float): The point at which to evaluate the Hessian.
            h (float): Step size for the finite difference (default 1e-5).

        Returns:
            float: The second derivative at x0.
        """
        def grad_at(x_val):
            x = AutoDiff.Variable(x_val)
            loss = fn(x)
            loss.backward()
            return x.grad
        return (grad_at(x0 + h) - grad_at(x0 - h)) / (2 * h)
