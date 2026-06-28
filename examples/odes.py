import math

from axiompy import Axiom


# ---- solve_ivp with various methods ----
def f(t, y):
    return [y[0]]

for method in ('euler', 'rk4', 'rk45', 'adams_bashforth'):
    ts, ys = Axiom.solve_ivp(f, [1.0], (0, 1), method=method, dt=0.01)
    print(f"{method:18s}  y(1) = {ys[-1][0]:.6f}  (exact: {math.e:.6f})")

# ---- 2D system: simple harmonic oscillator ----
def sho(t, y):
    return [-y[1], y[0]]

ts, ys = Axiom.solve_ivp(sho, [1.0, 0.0], (0, math.pi / 2), method='rk4', dt=0.01)
print(f"\nSHO at pi/2: y = ({ys[-1][0]:.4f}, {ys[-1][1]:.4f})  (expected ~(0, 1))")

# ---- BVP: y'' = -y, y(0)=0, y(pi/2)=1 => y(x) = sin(x) ----
def bvp_f(t, y):
    return [y[1], -y[0]]

def bvp_bc(y_end):
    return [y_end[0] - 1.0]

ts, ys = Axiom.solve_bvp(bvp_f, bvp_bc, (0, math.pi / 2), guess=[0.0, 1.0], dt=0.05)
print(f"\nBVP y(pi/2) = {ys[-1][0]:.4f}  (expected 1.0)")

# ---- Pendulum ----
pend = Axiom.pendulum_odes(L=1.0, b=0.2)
ts, ys = Axiom.solve_ivp(pend, [0.5, 0.0], (0, 10), method='rk4', dt=0.01)
print(f"\nPendulum at t=10: theta={ys[-1][0]:.4f}, omega={ys[-1][1]:.4f}")

# ---- Lotka-Volterra ----
lv = Axiom.lotka_volterra_odes()
ts, ys = Axiom.solve_ivp(lv, [10.0, 2.0], (0, 50), method='rk4', dt=0.05)
print(f"Lotka-Volterra at t=50: prey={ys[-1][0]:.1f}, pred={ys[-1][1]:.1f}")
