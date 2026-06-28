import cmath
import logging
import math
from collections.abc import Sequence
from typing import Optional

logger = logging.getLogger(__name__)


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


class Signal:
    """Digital signal processing operations including transforms and filtering."""

    @staticmethod
    def dft(x: Sequence) -> list[complex]:
        """Compute the Discrete Fourier Transform of a sequence.

        Args:
            x (Sequence): Input signal samples.

        Returns:
            List[complex]: Frequency-domain representation.
        """
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
    def idft(X: Sequence[complex]) -> list[float]:
        """Compute the Inverse Discrete Fourier Transform of a frequency sequence.

        Args:
            X (Sequence[complex]): Frequency-domain coefficients.

        Returns:
            List[float]: Reconstructed time-domain signal.
        """
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
    def fft(x: Sequence) -> list[complex]:
        """Compute the Fast Fourier Transform using the Cooley-Tukey algorithm.

        Falls back to DFT when the input length is not a power of two.

        Args:
            x (Sequence): Input signal samples.

        Returns:
            List[complex]: Frequency-domain representation.
        """
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
    def ifft(X: Sequence[complex]) -> list[float]:
        """Compute the Inverse Fast Fourier Transform.

        Args:
            X (Sequence[complex]): Frequency-domain coefficients.

        Returns:
            List[float]: Reconstructed time-domain signal.
        """
        n = len(X)
        conjugated = [x.conjugate() for x in X]
        forward = Signal.fft(conjugated) if n & (n - 1) == 0 else Signal.dft(conjugated)
        return [v.real / n for v in forward]

    @staticmethod
    def convolve(a: Sequence[float], b: Sequence[float]) -> list[float]:
        """Compute the discrete convolution of two sequences.

        Args:
            a (Sequence[float]): First input sequence.
            b (Sequence[float]): Second input sequence.

        Returns:
            List[float]: Convolution result of length len(a) + len(b) - 1.
        """
        n = len(a) + len(b) - 1
        result = [0.0] * n
        for i, av in enumerate(a):
            for j, bv in enumerate(b):
                result[i + j] += av * bv
        return result

    @staticmethod
    def moving_average(data: Sequence[float], window: int) -> list[float]:
        """Apply a moving-average filter to a data sequence.

        Args:
            data (Sequence[float]): Input data.
            window (int): Window size.

        Returns:
            List[float]: Smoothed signal.
        """
        result = []
        for i in range(len(data) - window + 1):
                result.append(sum(data[i:i + window]) / window)
        return result

    @staticmethod
    def pad_next_power_of_two(x: Sequence[float]) -> list[float]:
        """Zero-pad a sequence to the next power of two length.

        Args:
            x (Sequence[float]): Input signal.

        Returns:
            List[float]: Zero-padded signal.
        """
        n = _next_power_of_two(len(x))
        return list(x) + [0.0] * (n - len(x))

    @staticmethod
    def autocorrelation(x: Sequence[float]) -> list[float]:
        """Compute the autocorrelation of a signal via FFT.

        Args:
            x (Sequence[float]): Input signal.

        Returns:
            List[float]: Autocorrelation (same length as input).
        """
        n = len(x)
        padded = Signal.pad_next_power_of_two(x)
        X = Signal.fft(padded)
        psd = [abs(v) ** 2 for v in X]
        acf = Signal.ifft(psd)
        return [acf[k] / acf[0] for k in range(n)]

    @staticmethod
    def cross_correlation(x: Sequence[float],
                          y: Sequence[float]) -> list[float]:
        """Compute the cross-correlation of two signals via FFT.

        Args:
            x (Sequence[float]): First signal.
            y (Sequence[float]): Second signal.

        Returns:
            List[float]: Cross-correlation (length ``len(x) + len(y) - 1``).
        """
        n = len(x) + len(y) - 1
        fft_len = _next_power_of_two(n)
        x_pad = list(x) + [0.0] * (fft_len - len(x))
        y_pad = list(y) + [0.0] * (fft_len - len(y))
        X = Signal.fft(x_pad)
        Y = Signal.fft(y_pad)
        XY = [X[k] * Y[k].conjugate() for k in range(fft_len)]
        cc = Signal.ifft(XY)
        return cc[:n]

    @staticmethod
    def sinc_filter(cutoff: float, fs: float,
                     taps: int = 51) -> list[float]:
        """Design a low-pass FIR filter using a windowed sinc.

        Uses a Hamming window to reduce Gibbs phenomenon.

        Args:
            cutoff (float): Cutoff frequency in Hz.
            fs (float): Sampling frequency in Hz.
            taps (int): Number of filter taps (default 51, must be odd).

        Returns:
            List[float]: Filter coefficients.
        """
        if taps % 2 == 0:
            taps += 1
        fc = cutoff / fs
        half = taps // 2
        kernel = [0.0] * taps
        for i in range(taps):
            n = i - half
            if n == 0:
                val = 2 * fc
            else:
                val = math.sin(2 * math.pi * fc * n) / (math.pi * n)
            window = 0.54 - 0.46 * math.cos(2 * math.pi * i / (taps - 1))
            kernel[i] = val * window
        total = sum(kernel)
        return [v / total for v in kernel]

    @staticmethod
    def downsample(x: Sequence[float], factor: int) -> list[float]:
        """Downsample a signal by an integer factor (decimate).

        Args:
            x (Sequence[float]): Input signal.
            factor (int): Downsampling factor.

        Returns:
            List[float]: Downsampled signal.
        """
        return [x[i] for i in range(0, len(x), factor)]

    @staticmethod
    def upsample(x: Sequence[float], factor: int) -> list[float]:
        """Upsample a signal by an integer factor (insert zeros).

        Args:
            x (Sequence[float]): Input signal.
            factor (int): Upsampling factor.

        Returns:
            List[float]: Upsampled signal with zeros inserted.
        """
        result = []
        for v in x:
            result.append(v)
            result.extend([0.0] * (factor - 1))
        return result

    @staticmethod
    def spectrogram(x: Sequence[float], window_size: int = 256,
                     hop_size: Optional[int] = None) -> list[list[float]]:
        """Compute a magnitude spectrogram via STFT.

        Args:
            x (Sequence[float]): Input signal.
            window_size (int): FFT window size (default 256).
            hop_size (int, optional): Hop size between windows (default window_size // 2).

        Returns:
            List[List[float]]: Spectrogram as a list of frequency magnitude vectors.
        """
        if hop_size is None:
            hop_size = window_size // 2
        window = [0.54 - 0.46 * math.cos(2 * math.pi * i / (window_size - 1))
                  for i in range(window_size)]
        n_fft = _next_power_of_two(window_size)
        spectrogram_data = []
        for start in range(0, len(x) - window_size + 1, hop_size):
            segment = [x[start + i] * window[i] for i in range(window_size)]
            padded = segment + [0.0] * (n_fft - window_size)
            X = Signal.fft(padded)
            mags = [abs(X[k]) for k in range(n_fft // 2 + 1)]
            spectrogram_data.append(mags)
        return spectrogram_data
