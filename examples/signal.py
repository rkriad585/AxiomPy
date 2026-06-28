import math

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

# --- Phase 5.7 features ---

# Pad to next power of 2
small = [1, 2, 3]
padded = Signal.pad_next_power_of_two(small)
print(f"\nPad {small} to next power of 2: {padded} (len={len(padded)})")

# Autocorrelation
ac = Signal.autocorrelation(x2)
print(f"Autocorrelation (first 3): {[f'{v:.4f}' for v in ac[:3]]}")

# Cross-correlation
xcorr = Signal.cross_correlation(x2, x2)
print(f"Cross-correlation (self, first 3): {[f'{v:.4f}' for v in xcorr[:3]]}")

# Sinc low-pass filter
coeffs = Signal.sinc_filter(cutoff=100, fs=1000, taps=31)
print(f"\nSinc filter taps (first 5): {[f'{c:.4f}' for c in coeffs[:5]]}")
print(f"  Number of taps: {len(coeffs)}")

# Downsample / upsample
orig = [1, 2, 3, 4, 5, 6]
down = Signal.downsample(orig, factor=2)
up = Signal.upsample(orig, factor=2)
print(f"\nOriginal:       {orig}")
print(f"Downsample (2): {down}")
print(f"Upsample (2):   {up}")

# Spectrogram
fs = 1000
t = [k / fs for k in range(512)]
chirp = [math.sin(2 * math.pi * (100 + 200 * tk) * tk) for tk in t]
spec = Signal.spectrogram(chirp, window_size=64, hop_size=32)
print(f"\nSpectrogram shape: {len(spec)} x {len(spec[0])}")
print("  (time frames x frequency bins)")
