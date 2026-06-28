
from axiompy import Axiom
from axiompy.stats import Statistics

data = Axiom.Matrix([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])
cov = Statistics.covariance_matrix(data)
print("Data:\n", data)
print("\nCovariance matrix:\n", cov)

# Each variable's values
x = [1, 4, 7]
y = [2, 5, 8]
x_bar = sum(x) / len(x)
y_bar = sum(y) / len(y)
manual_cov = sum((xi - x_bar) * (yi - y_bar) for xi, yi in zip(x, y)) / (len(x) - 1)
print(f"\nManual check: cov(x,y) = {manual_cov:.2f}")
