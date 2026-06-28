from axiompy import Axiom

# Create tensors
t0 = Axiom.Tensor([1, 2, 3])
t1 = Axiom.Tensor([[1, 2], [3, 4]])

print(f"1-D tensor: {t0}  shape={t0.shape}")
print(f"2-D tensor: {t1}  shape={t1.shape}")

# Arithmetic with broadcasting
t2 = Axiom.Tensor([[1, 2, 3], [4, 5, 6]])
t3 = Axiom.Tensor([10, 20, 30])
print(f"\nBroadcast addition:\n{(t2 + t3).to_list()}")

# Factory methods
z = Axiom.Tensor.zeros(2, 3)
o = Axiom.Tensor.ones(4)
e = Axiom.Tensor.eye(3)
print(f"\nZeros:\n{z.to_list()}")
print(f"Ones: {o.to_list()}")
print(f"Eye:\n{e.to_list()}")

# Reductions
data = Axiom.Tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"\nSum: {data.sum().to_list()}")
print(f"Mean: {data.mean().to_list()}")
print(f"Max: {data.max().to_list()}")

# Tensor operations
a = Axiom.Tensor([[1, 2], [3, 4]])
b = Axiom.Tensor([[5, 6], [7, 8]])
c = Axiom.Tensor.contract(a, b, axes=1)
print(f"\nContract (matmul):\n{c.to_list()}")

o = Axiom.Tensor.outer(Axiom.Tensor([1, 2]), Axiom.Tensor([3, 4, 5]))
print(f"\nOuter product:\n{o.to_list()}")

k = Axiom.Tensor.kronecker(a, b)
print(f"\nKronecker product shape: {k.shape}")

r = Axiom.Tensor.einsum("ij,jk->ik", a, b)
print(f"\nEinsum:\n{r.to_list()}")

# Reshape / transpose
t = Axiom.Tensor.arange(0, 6).reshape(2, 3)
print(f"\nReshaped:\n{t.to_list()}")
print(f"Transposed:\n{t.T.to_list()}")
