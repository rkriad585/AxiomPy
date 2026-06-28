import math
import pytest
from axiompy import Axiom


class TestSignal:
    def test_dft_idft(self):
        x = [math.sin(2 * math.pi * 2 * k / 8) for k in range(8)]
        X = Axiom.signal.dft(x)
        recovered = Axiom.signal.idft(X)
        for orig, rec in zip(x, recovered):
            assert rec == pytest.approx(orig, abs=1e-10)

    def test_fft_ifft(self):
        x = [1.0, 2.0, 3.0, 4.0]
        X = Axiom.signal.fft(x)
        recovered = Axiom.signal.ifft(X)
        for orig, rec in zip(x, recovered):
            assert rec == pytest.approx(orig, abs=1e-10)

    def test_fft_non_power_of_two(self):
        x = [1.0, 2.0, 3.0]
        X = Axiom.signal.fft(x)
        recovered = Axiom.signal.ifft(X)
        for orig, rec in zip(x, recovered):
            assert rec == pytest.approx(orig, abs=1e-10)

    def test_fft_single(self):
        X = Axiom.signal.fft([5.0])
        assert X[0] == pytest.approx(5.0)

    def test_convolve(self):
        result = Axiom.signal.convolve([1, 2, 3], [0, 1, 0.5])
        assert result == [0.0, 1.0, 2.5, 4.0, 1.5]

    def test_moving_average(self):
        result = Axiom.signal.moving_average([1, 2, 3, 4, 5], window=3)
        assert result == [2.0, 3.0, 4.0]

    def test_pad_next_power_of_two(self):
        result = Axiom.signal.pad_next_power_of_two([1, 2, 3])
        assert len(result) == 4
        assert result[:3] == [1.0, 2.0, 3.0]
        assert result[3] == 0.0

    def test_autocorrelation(self):
        x = [1.0, 2.0, 3.0, 4.0]
        acf = Axiom.signal.autocorrelation(x)
        assert len(acf) == len(x)
        assert acf[0] == pytest.approx(1.0, abs=1e-10)

    def test_cross_correlation(self):
        x = [1.0, 2.0, 3.0]
        y = [0.0, 1.0, 0.5]
        cc = Axiom.signal.cross_correlation(x, y)
        assert len(cc) == len(x) + len(y) - 1

    def test_sinc_filter(self):
        coeffs = Axiom.signal.sinc_filter(cutoff=100, fs=1000, taps=51)
        assert len(coeffs) == 51
        assert abs(sum(coeffs) - 1.0) < 1e-6

    def test_downsample(self):
        result = Axiom.signal.downsample([1, 2, 3, 4, 5, 6], factor=2)
        assert result == [1.0, 3.0, 5.0]

    def test_upsample(self):
        result = Axiom.signal.upsample([1, 2, 3], factor=2)
        assert result == [1.0, 0.0, 2.0, 0.0, 3.0, 0.0]

    def test_spectrogram(self):
        x = [math.sin(2 * math.pi * 5 * k / 100) for k in range(1000)]
        spec = Axiom.signal.spectrogram(x, window_size=64, hop_size=32)
        assert len(spec) > 0
        assert len(spec[0]) == 33  # n_fft/2 + 1 for n_fft=64
