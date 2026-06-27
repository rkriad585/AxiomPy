import math
from functools import reduce
from typing import List
from ._base import AxiomError

class NumberTheory:
    @staticmethod
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        g, x1, y1 = NumberTheory.extended_gcd(b % a, a)
        return g, y1 - (b // a) * x1, x1

    @staticmethod
    def mod_inverse(a, m):
        g, x, _ = NumberTheory.extended_gcd(a, m)
        if g != 1:
            raise AxiomError(f'Modular inverse does not exist for {a} and {m}')
        return x % m

    @staticmethod
    def chinese_remainder_theorem(n: List[int], a: List[int]) -> int:
        prod = reduce(lambda x, y: x * y, n)
        result = 0
        for n_i, a_i in zip(n, a):
            p = prod // n_i
            result += a_i * NumberTheory.mod_inverse(p, n_i) * p
        return result % prod

    @staticmethod
    def sieve_of_eratosthenes(n: int) -> List[int]:
        if n < 2:
            return []
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if sieve[i]:
                step = i
                start = i * i
                sieve[start:n + 1:step] = [False] * ((n - start) // step + 1)
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    @staticmethod
    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    @staticmethod
    def prime_factors(n: int) -> List[int]:
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1 if d == 2 else 2
        if n > 1:
            factors.append(n)
        return factors

    @staticmethod
    def euler_totient(n: int) -> int:
        result = n
        p = 2
        temp = n
        while p * p <= temp:
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
            p += 1 if p == 2 else 2
        if temp > 1:
            result -= result // temp
        return result
