"""Accelerated backends example — JAX (JIT) and CuPy (CUDA GPU).

Both are optional dependencies.  If the library is not installed,
the corresponding section is skipped.
"""
from axiompy import get_backend, set_backend


def try_jax():
    try:
        from axiompy._jax_backend import _HAS_JAX
        if not _HAS_JAX:
            print("\n=== JAX backend: skipping (jax not installed) ===")
            return
    except ImportError:
        print("\n=== JAX backend: skipping (jax not installed) ===")
        return

    set_backend("jax")
    print(f"\n=== JAX backend: {get_backend().__class__.__name__} ===")

    b = get_backend()
    a = b.array([[1, 2], [3, 4]])
    print(f"Array:  {a}")

    r = b.dot(a, a)
    print(f"A @ A:  {r}")

    d = b.det(a)
    print(f"det(A): {d:.4f}")

    x = b.solve([[2, 1], [1, 3]], [5, 6])
    print(f"Solve:  x = {x}")

    ev = b.eigvals([[4, 1], [1, 3]])
    print(f"Eigenvalues: {sorted(ev.tolist(), reverse=True)}")

    set_backend("numpy")


def try_cupy():
    try:
        from axiompy._cupy_backend import _HAS_CUPY
        if not _HAS_CUPY:
            print("\n=== CuPy backend: skipping (cupy not installed) ===")
            return
    except ImportError:
        print("\n=== CuPy backend: skipping (cupy not installed) ===")
        return

    set_backend("cupy")
    print(f"\n=== CuPy backend: {get_backend().__class__.__name__} ===")

    b = get_backend()
    a = b.array([[1.0, 2.0], [3.0, 4.0]])
    print(f"Array:  {a}")

    r = b.dot(a, a)
    print(f"A @ A:  {r}")

    d = b.det(a)
    print(f"det(A): {d:.4f}")

    x = b.solve([[2, 1], [1, 3]], [5, 6])
    print(f"Solve:  x = {x}")

    set_backend("numpy")


try_jax()
try_cupy()

print(f"\nBackend restored to: {get_backend().__class__.__name__}")
