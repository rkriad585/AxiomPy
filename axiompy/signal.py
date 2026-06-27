import math
import cmath
from typing import List, Sequence


class Signal:
    @staticmethod
    def dft(x: Sequence) -> List[complex]:
        n = len(x)
        result = []
        for k in range(n):
            s = 0j
            for t in range(n):
                angle = -2j * cmath.pi * k * t / n
                s += x[t] * cmath.exp(angle)
            result.append(s)
        return result

    @staticmethod
    def idft(X: Sequence[complex]) -> List[float]:
        n = len(X)
        result = []
        for t in range(n):
            s = 0j
            for k in range(n):
                angle = 2j * cmath.pi * k * t / n
                s += X[k] * cmath.exp(angle)
            result.append(s.real / n)
        return result

    @staticmethod
    def fft(x: Sequence) -> List[complex]:
        n = len(x)
        if n <= 1:
            return [complex(v) for v in x]
        if n & (n - 1) != 0:
            return Signal.dft([complex(v) for v in x])
        even = Signal.fft(x[0::2])
        odd = Signal.fft(x[1::2])
        result = [0j] * n
        for k in range(n // 2):
            w = cmath.exp(-2j * cmath.pi * k / n)
            result[k] = even[k] + w * odd[k]
            result[k + n // 2] = even[k] - w * odd[k]
        return result

    @staticmethod
    def ifft(X: Sequence[complex]) -> List[float]:
        n = len(X)
        conjugated = [x.conjugate() for x in X]
        forward = Signal.fft(conjugated) if n & (n - 1) == 0 else Signal.dft(conjugated)
        return [v.real / n for v in forward]

    @staticmethod
    def convolve(a: Sequence[float], b: Sequence[float]) -> List[float]:
        n = len(a) + len(b) - 1
        result = [0.0] * n
        for i, av in enumerate(a):
            for j, bv in enumerate(b):
                result[i + j] += av * bv
        return result

    @staticmethod
    def moving_average(data: Sequence[float], window: int) -> List[float]:
        result = []
        for i in range(len(data) - window + 1):
            result.append(sum(data[i:i + window]) / window)
        return result
