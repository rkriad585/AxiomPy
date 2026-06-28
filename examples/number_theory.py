from axiompy import Axiom

n = [3, 5, 7]
a = [2, 3, 2]
solution = Axiom.number_theory.chinese_remainder_theorem(n, a)
print(f"x = {a[0]} (mod {n[0]})")
print(f"x = {a[1]} (mod {n[1]})")
print(f"x = {a[2]} (mod {n[2]})")
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

print(f"\nEuler totient phi(10) = {Axiom.number_theory.euler_totient(10)}")
print(f"Euler totient phi(17) = {Axiom.number_theory.euler_totient(17)}")

# --- Phase 5.9 features ---

# Miller-Rabin
print(f"\nMiller-Rabin: 7919 prime? {Axiom.number_theory.miller_rabin(7919)}")
print(f"Miller-Rabin: 7920 prime? {Axiom.number_theory.miller_rabin(7920)}")

# Next / Nth prime
print(f"Next prime >= 100: {Axiom.number_theory.next_prime(100)}")
print(f"10th prime: {Axiom.number_theory.nth_prime(10)}")

# Legendre / Jacobi symbols
print(f"\nLegendre (2|7) = {Axiom.number_theory.legendre_symbol(2, 7)}")
print(f"Jacobi (7|15) = {Axiom.number_theory.jacobi_symbol(7, 15)}")

# Discrete log (2^x = 8 mod 13)
x = Axiom.number_theory.discrete_log(2, 8, 13)
print(f"\nDiscrete log: 2^{x} = 8 (mod 13)")

# Fibonacci / Lucas
print(f"\nFibonacci(10) = {Axiom.number_theory.fibonacci(10)}  (expected: 55)")
print(f"Fibonacci(20) = {Axiom.number_theory.fibonacci(20)}  (expected: 6765)")
print(f"Lucas(0) = {Axiom.number_theory.lucas(0)}  (expected: 2)")
print(f"Lucas(10) = {Axiom.number_theory.lucas(10)}  (expected: 123)")
