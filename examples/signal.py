import math
from axiompy import Axiom
from axiompy.signal import Signal

# DFT of a simple sinusoid
n = 8
x = [math.sin(2 * math.pi * 2 * k / n) for k in range(n)]
X = Signal.dft(x)
recovered = Signal.idft(X)
print("Original:  ", [f"{v:.2f}" for v in x])
print("Recovered: ", [f"{v:.2f}" for v in recovered])
print("DFT magnitude (first 4):", [f"{abs(v):.2f}" for v in X[:4]])

# FFT (power-of-2 length)
n2 = 16
x2 = [math.cos(2 * math.pi * 3 * k / n2) for k in range(n2)]
X2 = Signal.fft(x2)
rec2 = Signal.ifft(X2)
print(f"\nFFT (n={n2}): match = {all(abs(a - b) < 1e-10 for a, b in zip(x2, rec2))}")

# Convolution
a = [1, 2, 3]
b = [0, 1, 0.5]
conv = Signal.convolve(a, b)
print(f"\nConvolution of {a} * {b}: {conv}")
print(f"  Length: {len(conv)} (expected: {len(a) + len(b) - 1})")

# Moving average
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ma = Signal.moving_average(data, window=3)
print(f"\nMoving average (window=3) of {data}:")
print(f"  {ma}")
print(f"  Length: {len(ma)} (expected: {len(data) - 3 + 1})")
