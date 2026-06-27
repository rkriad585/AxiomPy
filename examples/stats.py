from axiompy import Axiom
from axiompy.stats import Statistics

data = [1, 2, 2, 3, 4, 5, 5, 5, 6]

print("Data:", data)
print("mean  =", Statistics.mean(data))
print("median =", Statistics.median(data))
print("mode  =", Statistics.mode(data))
print("variance (sample) =", Statistics.variance(data))
print("std (sample) =", Statistics.std(data))
print("variance (pop)   =", Statistics.variance(data, ddof=0))
print("std (pop)   =", Statistics.std(data, ddof=0))
print("25th percentile =", Statistics.percentile(data, 25))
print("quartiles =", Statistics.quartiles(data))
print("IQR       =", Statistics.iqr(data))

x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]
print("\nCorrelation (x, y) =", Statistics.correlation_coefficient(x, y))

reg = Statistics.linear_regression(x, y)
print(f"Linear regression: y = {reg['slope']:.2f}x + {reg['intercept']:.2f}")
print(f"R² = {reg['r_squared']:.2f}")

# Covariance matrix
M = Axiom.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\nCovariance matrix:\n", Statistics.covariance_matrix(M))

print("\nNormal PDF at 0:", f"{Statistics.normal_pdf(0):.4f}")
print("Normal CDF at 0:", f"{Statistics.normal_cdf(0):.4f}")
print("Normal CDF at 1.96:", f"{Statistics.normal_cdf(1.96):.4f}")

print("\nRandom uniform:", f"{Statistics.random_sample('uniform'):.3f}")
