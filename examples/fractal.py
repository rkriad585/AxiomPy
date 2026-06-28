from axiompy import Axiom

# ---- Mandelbrot set ----
grid = Axiom.mandelbrot(40, 20, max_iter=50)
print("Mandelbrot (40x20):")
for row in grid:
    line = "".join("*" if c == 50 else " " if c > 10 else "." for c in row)
    print(f"  {line}")

# ---- Julia set ----
grid = Axiom.julia(-0.7 + 0.27j, 40, 20, max_iter=50)
print("\nJulia set (-0.7+0.27i, 40x20):")
for row in grid:
    line = "".join("*" if c == 50 else " " if c > 10 else "." for c in row)
    print(f"  {line}")

# ---- Logistic map ----
orbit = Axiom.logistic_map(3.8, 0.5, 20)
print("\nLogistic map (r=3.8, x0=0.5, n=20):")
print(f"  {[f'{x:.4f}' for x in orbit]}")

# ---- Bifurcation diagram points ----
points = Axiom.bifurcation_diagram(2.5, 4.0, r_steps=20, n_transient=20, n_plot=10)
print(f"\nBifurcation points: {len(points)} points generated")

# ---- Lyapunov exponent ----
def f_stable(x):
    return 2.0 * x * (1 - x)

def f_chaos(x):
    return 4.0 * x * (1 - x)

lam_s = Axiom.lyapunov_exponent(f_stable, 0.5, n=500)
lam_c = Axiom.lyapunov_exponent(f_chaos, 0.5, n=1000)
print("\nLyapunov exponents:")
print(f"  r=2.0 (stable):   lambda = {lam_s:.4f}")
print(f"  r=4.0 (chaotic):  lambda = {lam_c:.4f}")
