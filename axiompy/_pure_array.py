from typing import Union

Shape = tuple[int, ...]
ArrayLike = Union[list, 'PureArray']


class PureArray:
    """Lightweight array wrapper backed by nested Python lists.

    Supports 1-D (vector) and 2-D (matrix) layouts.  All operations
    are pure Python — no external dependencies.
    """

    def __init__(self, data: list):
        self._data = data
        self._shape = self._infer_shape(data)

    @staticmethod
    def _infer_shape(data: list) -> tuple[int, ...]:
        if not isinstance(data, list):
            raise TypeError(f"Expected list, got {type(data).__name__}")
        if not data:
            return (0,)
        if isinstance(data[0], list):
            return (len(data), len(data[0]))
        return (len(data),)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __len__(self):
        return self._shape[0]

    def copy(self) -> 'PureArray':
        return PureArray([row[:] if isinstance(row, list) else row for row in self._data])

    @property
    def T(self) -> 'PureArray':
        if self.ndim != 2:
            return self.copy()
        return PureArray([[self._data[j][i] for j in range(self._shape[0])]
                          for i in range(self._shape[1])])

    def tolist(self) -> list:
        return self._data

    def __repr__(self):
        return f"PureArray(shape={self._shape})"

    # Arithmetic helpers used by the backend
    def __add__(self, other):
        if isinstance(other, PureArray):
            return PureArray(_add(self._data, other._data))
        return PureArray(_add_scalar(self._data, other))

    def __sub__(self, other):
        if isinstance(other, PureArray):
            return PureArray(_sub(self._data, other._data))
        return PureArray(_sub_scalar(self._data, other))

    def __mul__(self, other):
        if isinstance(other, PureArray):
            return PureArray(_mul(self._data, other._data))
        return PureArray(_mul_scalar(self._data, other))

    def __matmul__(self, other):
        return PureArray(_matmul(self._data, other._data))

    def __neg__(self):
        return PureArray(_neg(self._data))

    def __abs__(self):
        return PureArray(_abs(self._data))

    def sum(self):
        return _sum_all(self._data)

    def mean(self):
        return _mean_all(self._data)

    def max(self):
        return _max_all(self._data)

    def min(self):
        return _min_all(self._data)


# ---- internal arithmetic helpers (operate on raw list-of-lists / lists) -----


def _is_2d(x):
    return isinstance(x, list) and x and isinstance(x[0], list)


def _map2(f, a, b):
    if _is_2d(a):
        return [[f(ai, bi) for ai, bi in zip(ra, rb)] for ra, rb in zip(a, b)]
    return [f(ai, bi) for ai, bi in zip(a, b)]


def _map1(f, a):
    if _is_2d(a):
        return [[f(x) for x in row] for row in a]
    return [f(x) for x in a]


def _add(a, b):
    return _map2(lambda x, y: x + y, a, b)


def _sub(a, b):
    return _map2(lambda x, y: x - y, a, b)


def _mul(a, b):
    return _map2(lambda x, y: x * y, a, b)


def _add_scalar(a, s):
    return _map1(lambda x: x + s, a)


def _sub_scalar(a, s):
    return _map1(lambda x: x - s, a)


def _mul_scalar(a, s):
    return _map1(lambda x: x * s, a)


def _neg(a):
    return _map1(lambda x: -x, a)


def _abs(a):
    return _map1(abs, a)


def _sum_all(a):
    if _is_2d(a):
        return sum(sum(row) for row in a)
    return sum(a)


def _mean_all(a):
    if _is_2d(a):
        total = sum(sum(row) for row in a)
        n = sum(len(row) for row in a)
    else:
        total = sum(a)
        n = len(a)
    return total / n if n else 0.0


def _max_all(a):
    if _is_2d(a):
        return max(max(row) for row in a)
    return max(a)


def _min_all(a):
    if _is_2d(a):
        return min(min(row) for row in a)
    return min(a)


def _matmul(a, b):
    """Matrix multiply a @ b (2-D only)."""
    if not (a and b):
        return []
    m, k1 = len(a), len(a[0])
    k2, n = len(b), len(b[0])
    if k1 != k2:
        raise ValueError(f"Incompatible shapes: ({m},{k1}) @ ({k2},{n})")
    return [[sum(a[i][k] * b[k][j] for k in range(k1)) for j in range(n)]
            for i in range(m)]
