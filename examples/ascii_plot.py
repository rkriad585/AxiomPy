from axiompy import Axiom

x = [i * 0.4 for i in range(20)]
y = [v**2 for v in x]

print("y = x^2:")
Axiom.viz.plot_ascii(x, y)
