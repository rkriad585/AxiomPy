"""Example: Memory-mapped arrays (out-of-core) and dtype support."""

import tempfile

import numpy as np

from axiompy import Axiom


def main():
    # --- Dtype support ---
    print("=== Dtype Support ===")
    v = Axiom.Vector([1.5, 2.5, 3.5])
    print(f"Default dtype: {v.dtype}")

    v32 = v.astype("float32")
    print(f"float32 vector: {v32.dtype} → {v32.to_list()}")

    vi = v.astype("int32")
    print(f"int32 vector:   {vi.dtype} → {vi.to_list()}")

    M = Axiom.Matrix([[1, 2], [3, 4]])
    m32 = M.astype("float32")
    print(f"Matrix float32: {m32.dtype}")

    # --- Out-of-core with MmapArray ---
    print("\n=== Out-of-core MmapArray ===")
    tmpdir = tempfile.mkdtemp()
    path = f"{tmpdir}/data.mmap"

    # Create a zero-initialized mmap array
    mm = Axiom.MmapArray.zeros((100, 50), path=path)
    mm.open()
    print(f"Created: {mm}")
    print(f"  shape={mm.shape}, dtype={mm.dtype}, size={mm.size}")

    # Set some values
    mm[0, 0] = 42.0
    mm[1, :] = np.arange(50.0)
    print(f"  mm[0,0] = {mm[0,0]}")
    print(f"  mm[1,0] = {mm[1,0]}, mm[1,1] = {mm[1,1]}")

    # Chunked matmul with a vector
    v2 = Axiom.Vector(np.ones(50))
    result = mm.matmul(v2)
    print(f"  matmul with ones vector: first 3 = {result.to_list()[:3]}")

    # Chunked scalar addition
    mm2_path = f"{tmpdir}/data2.mmap"
    arr = np.ones((10, 5))
    mm2 = Axiom.MmapArray.from_array(arr, path=mm2_path)
    mm2.add(10.0)
    print(f"  add(10): mm2[0,0] = {mm2[0,0]}")  # should be 11

    # Chunked scalar multiplication
    mm2.mul(3.0)
    print(f"  mul(3): mm2[0,0] = {mm2[0,0]}")  # should be 33

    # Convert back to dense
    dense = mm2.to_dense()
    print(f"  dense: shape={dense.shape}, [0,0]={dense.to_list()[0][0]}")

    # Load from mmap into Matrix
    M2 = Axiom.Matrix.from_mmap(mm2)
    print(f"  Matrix.from_mmap: {M2.shape}, dtype={M2.dtype}")

    # Context manager pattern
    path3 = f"{tmpdir}/data3.mmap"
    with Axiom.MmapArray.zeros((3, 3), path=path3) as mm3:
        mm3[0, :] = [1, 2, 3]
        mm3[1, :] = [4, 5, 6]
        mm3[2, :] = [7, 8, 9]
        v3 = Axiom.Vector([1.0, 2.0, 3.0])
        res = mm3.matmul(v3)
        print(f"\n  Context manager matmul: {res.to_list()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
