"""Example: Sparse matrix operations with SparseMatrix."""

from axiompy import Axiom


def main():
    # Create a dense matrix with mostly zeros
    M = Axiom.Matrix([
        [5, 0, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 0],
        [0, 9, 0, 7],
    ])
    print("Dense matrix:")
    print(M.to_list())

    # Convert to sparse
    sp = M.to_sparse()
    print(f"\nSparse: shape={sp.shape}, nnz={sp.nnz}")
    print(f"  density = {sp.density:.4g}")

    # Convert back to dense
    dense = sp.to_dense()
    print(f"Round-trip matches: {dense == M}")

    # Identity in sparse
    I = Axiom.SparseMatrix.identity(4)
    print(f"\nSparse identity: shape={I.shape}, nnz={I.nnz}")

    # Sparse-sparse addition
    sp2 = Axiom.SparseMatrix([0, 1], [0, 1], [1.0, 1.0], (4, 4))
    summed = sp + sp2
    print(f"\nSum: shape={summed.shape}, nnz={summed.nnz}")
    print(summed.to_dense().to_list())

    # Scalar multiplication
    scaled = sp * 2.0
    print(f"\nScaled * 2.0:\n{scaled.to_dense().to_list()}")

    # Matrix-vector product
    v = Axiom.Vector([1.0, 2.0, 3.0, 4.0])
    result = sp @ v
    print(f"\nMatrix-vector product: {result.to_list()}")

    # Sparse-sparse matrix product
    A = Axiom.SparseMatrix([0, 1], [0, 1], [2.0, 3.0], (2, 2))
    B = Axiom.SparseMatrix([0, 1], [0, 1], [4.0, 5.0], (2, 2))
    C = A @ B
    print(f"\nSparse @ sparse:\n{C.to_dense().to_list()}")

    # COO extraction
    rows, cols, data = sp.to_coo()
    print(f"\nCOO: rows={rows.tolist()}, cols={cols.tolist()}, data={data.tolist()}")


if __name__ == "__main__":
    main()
