import cmath
import math

from ._backend import Backend
from ._pure_array import PureArray, _matmul


class PurePythonBackend(Backend):
    """Pure-Python backend that implements all :class:`Backend` methods
    using only standard library types (lists and floats).  No numpy dependency.
    """

    def array(self, data, dtype=None):
        if isinstance(data, PureArray):
            return data.copy()
        if isinstance(data, list):
            return PureArray(data)
        if hasattr(data, 'tolist'):
            return PureArray(data.tolist())
        return PureArray(list(data))

    # ---- linear algebra ----------------------------------------------------

    def dot(self, a, b):
        a = self._unwrap(a)
        b = self._unwrap(b)
        if _is_2d(a) and _is_2d(b):
            return PureArray(_matmul(a, b))
        # 1-D dot → scalar
        return sum(ai * bi for ai, bi in zip(a, b))

    def inv(self, a):
        a = self._unwrap(a)
        n = len(a)
        aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
               for i, row in enumerate(a)]
        # Forward elimination
        for col in range(n):
            pivot = _find_pivot(aug, col, col)
            if pivot is None:
                raise ValueError("Singular matrix")
            aug[col], aug[pivot] = aug[pivot], aug[col]
            piv_val = aug[col][col]
            for j in range(2 * n):
                aug[col][j] /= piv_val
            for row in range(n):
                if row != col:
                    factor = aug[row][col]
                    for j in range(2 * n):
                        aug[row][j] -= factor * aug[col][j]
        return PureArray([row[n:] for row in aug])

    def solve(self, A, b):
        A = self._unwrap(A)
        b = self._unwrap(b)
        n = len(A)
        aug = [[*A[i][:], b[i]] for i in range(n)]
        for col in range(n):
            pivot = _find_pivot(aug, col, col)
            if pivot is None:
                raise ValueError("Singular matrix")
            aug[col], aug[pivot] = aug[pivot], aug[col]
            piv_val = aug[col][col]
            for j in range(n + 1):
                aug[col][j] /= piv_val
            for row in range(n):
                if row != col:
                    factor = aug[row][col]
                    for j in range(n + 1):
                        aug[row][j] -= factor * aug[col][j]
        return PureArray([row[n] for row in aug])

    def det(self, a):
        a = self._unwrap(a)
        n = len(a)
        if n == 0:
            return 1.0
        mat = [row[:] for row in a]
        det = 1.0
        for col in range(n):
            pivot = _find_pivot(mat, col, col)
            if pivot is None:
                return 0.0
            if pivot != col:
                mat[col], mat[pivot] = mat[pivot], mat[col]
                det = -det
            det *= mat[col][col]
            for row in range(col + 1, n):
                factor = mat[row][col] / mat[col][col]
                for j in range(col, n):
                    mat[row][j] -= factor * mat[col][j]
        return det

    def matrix_power(self, a, power):
        a = self._unwrap(a)
        n = len(a)
        if power == 0:
            return PureArray([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)])
        result = a
        for _ in range(power - 1):
            result = _matmul(result, a)
        return PureArray(result)

    def norm(self, a, ord=None):
        a = self._unwrap(a)
        flat = _flatten(a)
        if ord is None:
            # Frobenius norm for matrices, 2-norm for vectors
            if _is_2d(a):
                return math.sqrt(sum(x * x for x in flat))
            return math.sqrt(sum(x * x for x in a))
        if ord == 1:
            if _is_2d(a):
                return max(sum(abs(a[i][j]) for i in range(len(a))) for j in range(len(a[0])))
            return sum(abs(x) for x in a)
        if ord == 2:
            return math.sqrt(sum(x * x for x in flat))
        if ord == float('inf') or ord == 'inf':
            if _is_2d(a):
                return max(sum(abs(x) for x in row) for row in a)
            return max(abs(x) for x in a)
        if _is_2d(a):
            return math.sqrt(sum(x * x for x in flat))
        return sum(abs(x) ** ord for x in a) ** (1.0 / ord)

    def qr(self, a):
        """QR decomposition via classical Gram-Schmidt."""
        a = self._unwrap(a)
        m, n = len(a), len(a[0])
        q = [[0.0] * n for _ in range(m)]
        r = [[0.0] * n for _ in range(n)]
        for j in range(n):
            v = [a[i][j] for i in range(m)]
            for i in range(j):
                r[i][j] = sum(q[k][i] * a[k][j] for k in range(m))
                for k in range(m):
                    v[k] -= r[i][j] * q[k][i]
            r[j][j] = math.sqrt(sum(x * x for x in v))
            if r[j][j] != 0:
                for k in range(m):
                    q[k][j] = v[k] / r[j][j]
            else:
                for k in range(m):
                    q[k][j] = 0.0
        return PureArray(q), PureArray(r)

    def cholesky(self, a):
        """Cholesky decomposition A = L @ L^T for SPD matrix."""
        a = self._unwrap(a)
        n = len(a)
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    val = a[i][i] - s
                    if val <= 0:
                        raise ValueError("Matrix not positive definite")
                    L[i][j] = math.sqrt(val)
                else:
                    L[i][j] = (a[i][j] - s) / L[j][j]
        return PureArray(L)

    def eigvals(self, a):
        """Compute eigenvalues via QR iteration (symmetric matrices)."""
        a = self._unwrap(a)
        n = len(a)
        if n == 0:
            return PureArray([])
        if n == 1:
            return PureArray([a[0][0]])
        # Symmetrize for stability
        mat = [[a[i][j] for j in range(n)] for i in range(n)]
        max_iter = 1000
        tol = 1e-10
        for _ in range(max_iter):
            q, r_mat = self.qr(PureArray(mat))
            q_data = self._unwrap(q)
            r_data = self._unwrap(r_mat)
            mat = _matmul(r_data, q_data)
            # Check off-diagonal convergence
            off = sum(abs(mat[i][j]) for i in range(n) for j in range(n) if i != j)
            if off < tol:
                break
        return PureArray([mat[i][i] for i in range(n)])

    def matrix_rank(self, a):
        """Compute matrix rank via Gaussian elimination with a tolerance."""
        a_data = self._unwrap(a)
        m, n = len(a_data), len(a_data[0])
        mat = [row[:] for row in a_data]
        rank = 0
        col = 0
        for row in range(m):
            if col >= n:
                break
            pivot = _find_pivot(mat, row, col)
            if pivot is None:
                continue
            mat[row], mat[pivot] = mat[pivot], mat[row]
            piv_val = mat[row][col]
            for r in range(row + 1, m):
                factor = mat[r][col] / piv_val
                for c in range(col, n):
                    mat[r][c] -= factor * mat[row][c]
            rank += 1
            col += 1
        return rank

    # ---- FFT (naive O(n²) DFT fallback) ------------------------------------

    def fft(self, x):
        """Naive O(n²) DFT.

        Args:
            x: Input list or PureArray of complex/float values.

        Returns:
            PureArray: Complex DFT coefficients.
        """
        x = self._unwrap(x)
        n = len(x)
        result = []
        for k in range(n):
            s = 0.0 + 0.0j
            for t in range(n):
                angle = -2 * math.pi * k * t / n
                s += x[t] * cmath.exp(complex(0, angle))
            result.append(s)
        return PureArray(result)

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _unwrap(a):
        """Return raw nested-list representation."""
        if isinstance(a, PureArray):
            return a._data
        if hasattr(a, 'tolist'):
            return a.tolist()
        return list(a)


def _find_pivot(mat, row, col):
    """Find pivot row index for Gaussian elimination (partial pivoting)."""
    n = len(mat)
    max_val = abs(mat[row][col]) if row < n else 0.0
    pivot = row if row < n else None
    for i in range(row + 1, n):
        val = abs(mat[i][col])
        if val > max_val:
            max_val = val
            pivot = i
    return pivot if max_val > 1e-15 else None


def _dot_1d(a, b):
    """1-D dot product."""
    return [sum(a[i][j] * b[i] for i in range(len(a))) for j in range(len(a[0]))] if _is_2d(a) else \
           sum(ai * bi for ai, bi in zip(a, b))


def _is_2d(x):
    return isinstance(x, list) and x and isinstance(x[0], list)


def _flatten(a):
    if _is_2d(a):
        return [x for row in a for x in row]
    return list(a)
