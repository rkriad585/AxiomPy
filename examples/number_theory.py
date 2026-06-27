from axiompy import Axiom

n = [3, 5, 7]
a = [2, 3, 2]
solution = Axiom.number_theory.chinese_remainder_theorem(n, a)
print(f"x ≡ {a[0]} (mod {n[0]})")
print(f"x ≡ {a[1]} (mod {n[1]})")
print(f"x ≡ {a[2]} (mod {n[2]})")
print(f"Solution: x = {solution}")
print(f"Verify: {solution} % 3 = {solution % 3}")
print(f"Verify: {solution} % 5 = {solution % 5}")
print(f"Verify: {solution} % 7 = {solution % 7}")

a, m = 7, 26
inv = Axiom.number_theory.mod_inverse(a, m)
print(f"\n{1}/{a} mod {m} = {inv}  (check: {a} × {inv} mod {m} = {(a * inv) % m})")
