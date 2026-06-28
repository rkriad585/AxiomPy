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
print(f"R^2 = {reg['r_squared']:.2f}")

# Covariance matrix
M = Axiom.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\nCovariance matrix:\n", Statistics.covariance_matrix(M))

print("\nNormal PDF at 0:", f"{Statistics.normal_pdf(0):.4f}")
print("Normal CDF at 0:", f"{Statistics.normal_cdf(0):.4f}")
print("Normal CDF at 1.96:", f"{Statistics.normal_cdf(1.96):.4f}")

print("\nRandom uniform:", f"{Statistics.random_sample('uniform'):.3f}")

# --- Phase 5.3 features ---

# Hypothesis tests
a = [1, 2, 3, 4, 5]
b = [2, 3, 4, 5, 6]
print("\nOne-sample t-test (mu=3):", Statistics.ttest_1samp(a, 3))
print("Independent t-test:", Statistics.ttest_ind(a, b))
print("Paired t-test:", Statistics.ttest_paired(a, b))

# Chi-square
observed = [20, 30, 25, 25]
expected = [25, 25, 25, 25]
print("Chi-square:", Statistics.chisquare(observed, expected))

# One-way ANOVA
print("One-way ANOVA:", Statistics.f_oneway(a, [2, 3, 4], [3, 4, 5]))

# Distributions
print("\nUniform PDF (0.5):", Statistics.uniform_pdf(0.5))
print("Uniform CDF (0.5):", Statistics.uniform_cdf(0.5))
print("Exponential PDF (1.0, rate=2):", Statistics.exponential_pdf(1.0, rate=2.0))
print("Binomial PMF (3 successes, n=5, p=0.5):", Statistics.binomial_pmf(3, 5, 0.5))

# Z-score and MAD
print("\nZ-scores:", [f"{z:.2f}" for z in Statistics.zscore([1, 2, 3, 4, 5])])
print("MAD:", Statistics.mad([1, 2, 3, 4, 5]))

# Covariance with raw arrays + ddof
print("\nCovariance (arrays, ddof=0):\n", Statistics.covariance_matrix([[1, 2], [3, 4], [5, 6]], ddof=0))
