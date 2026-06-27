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
