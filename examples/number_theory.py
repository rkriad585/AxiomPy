from axiompy import Axiom

n = [3, 5, 7]
a = [2, 3, 2]
solution = Axiom.number_theory.chinese_remainder_theorem(n, a)
print(f"x ≡ {a[0]} (mod {n[0]})")
print(f"x ≡ {a[1]} (mod {n[1]})")
print(f"x ≡ {a[2]} (mod {n[2]})")
print(f"Solution: x = {solution}")

inv = Axiom.number_theory.mod_inverse(7, 26)
print(f"\n1/7 mod 26 = {inv}  (check: {7 * inv % 26})")

# Prime utilities
primes = Axiom.number_theory.sieve_of_eratosthenes(50)
print(f"\nPrimes up to 50: {primes}")

for n in [17, 18, 19, 7919]:
    print(f"is_prime({n}) = {Axiom.number_theory.is_prime(n)}")

print(f"\nPrime factors of 84: {Axiom.number_theory.prime_factors(84)}")
print(f"Prime factors of 100: {Axiom.number_theory.prime_factors(100)}")

print(f"\nEuler totient φ(10) = {Axiom.number_theory.euler_totient(10)}")
print(f"Euler totient φ(17) = {Axiom.number_theory.euler_totient(17)}")
