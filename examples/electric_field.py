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
