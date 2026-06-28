import random

from axiompy import Axiom

x = [i * 0.4 for i in range(20)]
y = [v**2 for v in x]

print("y = x^2:")
Axiom.viz.plot_ascii(x, y, title="Quadratic", xlabel="x", ylabel="y")

# Multi-series
x2 = [i * 0.4 for i in range(20)]
y2 = [v for v in x2]
print("\nTwo series:")
Axiom.viz.plot_ascii(
    x, y,
    title="Comparison",
    extra_series=[(x2, y2, "linear")]
)

# Histogram
random.seed(42)
data = [random.gauss(0, 1) for _ in range(200)]
print("\nHistogram:")
Axiom.viz.plot_histogram(data, bins=8, title="Normal distribution")

# Bar chart
print("\nBar chart:")
Axiom.viz.plot_bar(["Apples", "Bananas", "Cherries", "Dates"], [30, 55, 20, 45],
                    title="Fruit counts")

# Scatter
print("\nScatter plot:")
Axiom.viz.plot_scatter(x, y, title="Scatter")
