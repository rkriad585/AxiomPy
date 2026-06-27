import math
from numbers import Number

class AutoDiff:
    class Variable:
        def __init__(self, value: float, _children: set = None, _op: str = ''):
            self.value = value
            self.grad = 0.0
            self._backward = lambda: None
            self._prev = _children if _children is not None else set()

        def __repr__(self):
            return f"Variable(value={self.value:.4f}, grad={self.grad:.4f})"

        def __add__(self, other):
            other = other if isinstance(other, AutoDiff.Variable) else AutoDiff.Variable(other)
            out = AutoDiff.Variable(self.value + other.value, {self, other})
            def _backward():
                self.grad += out.grad
                other.grad += out.grad
            out._backward = _backward
            return out

        def __mul__(self, other):
            other = other if isinstance(other, AutoDiff.Variable) else AutoDiff.Variable(other)
            out = AutoDiff.Variable(self.value * other.value, {self, other})
            def _backward():
                self.grad += other.value * out.grad
                other.grad += self.value * out.grad
            out._backward = _backward
            return out

        def __pow__(self, other: Number):
            out = AutoDiff.Variable(self.value ** other, {self})
            def _backward():
                self.grad += (other * self.value ** (other - 1)) * out.grad
            out._backward = _backward
            return out

        def __neg__(self):
            return self * -1

        def __sub__(self, other):
            return self + (-other)

        def __truediv__(self, other):
            return self * other ** -1

        def __radd__(self, other):
            return self + other

        def __rmul__(self, other):
            return self * other

        def __rsub__(self, other):
            return other + (-self)

        def __rtruediv__(self, other):
            return other * self ** -1

        def backward(self):
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

    @staticmethod
    def sin(v: 'AutoDiff.Variable'):
        out = AutoDiff.Variable(math.sin(v.value), {v})
        def _backward():
            v.grad += math.cos(v.value) * out.grad
        out._backward = _backward
        return out

    @staticmethod
    def exp(v: 'AutoDiff.Variable'):
        out = AutoDiff.Variable(math.exp(v.value), {v})
        def _backward():
            v.grad += out.value * out.grad
        out._backward = _backward
        return out

    @staticmethod
    def log(v: 'AutoDiff.Variable'):
        out = AutoDiff.Variable(math.log(v.value), {v})
        def _backward():
            v.grad += (1 / v.value) * out.grad
        out._backward = _backward
        return out

    @staticmethod
    def tanh(v: 'AutoDiff.Variable'):
        t = math.tanh(v.value)
        out = AutoDiff.Variable(t, {v})
        def _backward():
            v.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    @staticmethod
    def sigmoid(v: 'AutoDiff.Variable'):
        s = 1 / (1 + math.exp(-v.value))
        out = AutoDiff.Variable(s, {v})
        def _backward():
            v.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    @staticmethod
    def sqrt(v: 'AutoDiff.Variable'):
        s = math.sqrt(v.value)
        out = AutoDiff.Variable(s, {v})
        def _backward():
            v.grad += (0.5 / s) * out.grad
        out._backward = _backward
        return out

    @staticmethod
    def gradient_descent(fn, start: float, lr: float = 0.01,
                         steps: int = 100) -> float:
        x = AutoDiff.Variable(start)
        for _ in range(steps):
            loss = fn(x)
            loss.backward()
            x.value -= lr * x.grad
            x.grad = 0.0
        return x.value
