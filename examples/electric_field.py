from axiompy import Axiom
from axiompy.electromagnetism import Electromagnetism

q = 1e-9
charges = [
    Electromagnetism.Charge(q, (0, 0, 0)),
    Electromagnetism.Charge(-q, (1, 0, 0)),
]

point = (0.5, 0.5, 0)
E = Axiom.electromagnetism.calculate_electric_field(charges, point)
print(f"Electric field at {point}: {E}")
print(f"|E| = {sum(v**2 for v in E.to_list())**0.5:.2f} N/C")

# Electric potential
V = Electromagnetism.electric_potential(charges, point)
print(f"Electric potential at {point}: {V:.2f} V")

# Magnetic field from moving charges
velocities = [(1e6, 0, 0), (-1e6, 0, 0)]
B = Electromagnetism.calculate_magnetic_field(charges, (0, 0, 1), velocities)
print(f"Magnetic field at (0,0,1): {B}")

# Dipole moment
p = Electromagnetism.dipole_moment(charges)
print(f"Dipole moment: {p} C.m")

# Field superposition
combined = Electromagnetism.combine_fields([E, B])
print(f"Combined E + B: {combined}")
