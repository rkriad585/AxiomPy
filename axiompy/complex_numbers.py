import math
from typing import Union

Number = Union[int, float, complex]


class ComplexNumber:
    """A complex number with polar/rect conversion and arithmetic operations."""

    def __init__(self, real: float = 0.0, imag: float = 0.0):
        if isinstance(real, complex):
            self._z = real
        else:
            self._z = complex(real, imag)

    @classmethod
    def from_polar(cls, r: float, phi: float) -> 'ComplexNumber':
        """Create a complex number from polar coordinates.

        Args:
            r (float): Magnitude.
            phi (float): Phase angle in radians.

        Returns:
            ComplexNumber: The complex number r * exp(i * phi).
        """
        return cls(r * math.cos(phi), r * math.sin(phi))

    @classmethod
    def from_complex(cls, z: complex) -> 'ComplexNumber':
        """Create from a Python complex number.

        Args:
            z (complex): A Python complex number.

        Returns:
            ComplexNumber: Wrapping z.
        """
        return cls(z.real, z.imag)

    @staticmethod
    def roots_of_unity(n: int) -> list['ComplexNumber']:
        """Compute the n-th roots of unity.

        Args:
            n (int): The order (>= 1).

        Returns:
            List[ComplexNumber]: The n complex roots of unity.
        """
        return [ComplexNumber.from_polar(1.0, 2 * math.pi * k / n) for k in range(n)]

    @property
    def real(self) -> float:
        return self._z.real

    @property
    def imag(self) -> float:
        return self._z.imag

    @property
    def conjugate(self) -> 'ComplexNumber':
        """Return the complex conjugate."""
        return ComplexNumber(self._z.real, -self._z.imag)

    def modulus(self) -> float:
        """Return the magnitude (absolute value)."""
        return abs(self._z)

    def argument(self) -> float:
        """Return the phase angle in radians."""
        return math.atan2(self._z.imag, self._z.real)

    def power(self, n: int) -> 'ComplexNumber':
        """Raise to integer power n via De Moivre.

        Args:
            n (int): The exponent.

        Returns:
            ComplexNumber: self ** n.
        """
        r = self.modulus() ** n
        phi = self.argument() * n
        return ComplexNumber.from_polar(r, phi)

    def __add__(self, other: 'ComplexNumber') -> 'ComplexNumber':
        if isinstance(other, ComplexNumber):
            return ComplexNumber(self._z + other._z)
        return NotImplemented

    def __sub__(self, other: 'ComplexNumber') -> 'ComplexNumber':
        if isinstance(other, ComplexNumber):
            return ComplexNumber(self._z - other._z)
        return NotImplemented

    def __mul__(self, other: 'ComplexNumber') -> 'ComplexNumber':
        if isinstance(other, ComplexNumber):
            return ComplexNumber(self._z * other._z)
        return NotImplemented

    def __truediv__(self, other: 'ComplexNumber') -> 'ComplexNumber':
        if isinstance(other, ComplexNumber):
            return ComplexNumber(self._z / other._z)
        return NotImplemented

    def __eq__(self, other) -> bool:
        if isinstance(other, ComplexNumber):
            return self._z == other._z
        return NotImplemented

    def __repr__(self) -> str:
        if self._z.imag >= 0:
            return f"({self._z.real:.6f} + {self._z.imag:.6f}i)"
        return f"({self._z.real:.6f} - {abs(self._z.imag):.6f}i)"

    def __str__(self) -> str:
        return repr(self)

    def to_complex(self) -> complex:
        """Convert to a plain Python complex number."""
        return self._z

    @staticmethod
    def cfft(x: list) -> list:
        """Alias for :func:`axiompy.signal.Signal.fft`.

        Args:
            x: Input signal (power-of-2 length recommended).

        Returns:
            list: FFT result.
        """
        from .signal import Signal
        return Signal.fft(x)


class ComplexVector:
    """A vector of complex numbers."""

    def __init__(self, data: list[Union[complex, ComplexNumber]]):
        self._data = [z if isinstance(z, ComplexNumber) else ComplexNumber.from_complex(z) for z in data]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> ComplexNumber:
        return self._data[idx]

    def __repr__(self) -> str:
        return f"ComplexVector({self._data})"

    def to_list(self) -> list[complex]:
        return [z.to_complex() for z in self._data]


class ComplexMatrix:
    """A matrix of complex numbers."""

    def __init__(self, data: list[list[Union[complex, ComplexNumber]]]):
        self._data = [
            [z if isinstance(z, ComplexNumber) else ComplexNumber.from_complex(z) for z in row]
            for row in data
        ]
        self._shape = (len(data), len(data[0]) if data else 0)

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def __repr__(self) -> str:
        return f"ComplexMatrix(shape={self._shape})"

    def to_list(self) -> list[list[complex]]:
        return [[z.to_complex() for z in row] for row in self._data]
