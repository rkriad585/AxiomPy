import logging
import math
import random
from functools import reduce

from ._base import AxiomError

logger = logging.getLogger(__name__)


class NumberTheory:
    """Number theory utilities including GCD, primality, and modular arithmetic."""

    @staticmethod
    def extended_gcd(a, b):
        """Compute the extended greatest common divisor of a and b.

        Args:
            a (int): First integer.
            b (int): Second integer.

        Returns:
            tuple: A tuple (g, x, y) such that g = gcd(a, b) = a*x + b*y.
        """
        if a == 0:
            return b, 0, 1
        g, x1, y1 = NumberTheory.extended_gcd(b % a, a)
        return g, y1 - (b // a) * x1, x1

    @staticmethod
    def mod_inverse(a, m):
        """Compute the modular inverse of a modulo m.

        Args:
            a (int): The number to invert.
            m (int): The modulus.

        Returns:
            int: The modular inverse of a modulo m.

        Raises:
            AxiomError: If the modular inverse does not exist.
        """
        g, x, _ = NumberTheory.extended_gcd(a, m)
        if g != 1:
            raise AxiomError(f'Modular inverse does not exist for {a} and {m}')
        return x % m

    @staticmethod
    def chinese_remainder_theorem(n: list[int], a: list[int]) -> int:
        """Solve a system of congruences using the Chinese Remainder Theorem.

        Args:
            n (List[int]): List of pairwise coprime moduli.
            a (List[int]): List of residues.

        Returns:
            int: The unique solution modulo the product of n.
        """
        prod = reduce(lambda x, y: x * y, n)
        result = 0
        for n_i, a_i in zip(n, a):
            p = prod // n_i
            result += a_i * NumberTheory.mod_inverse(p, n_i) * p
        return result % prod

    @staticmethod
    def sieve_of_eratosthenes(n: int) -> list[int]:
        """Generate all prime numbers up to n using the Sieve of Eratosthenes.

        Args:
            n (int): The upper bound (inclusive).

        Returns:
            List[int]: List of primes up to n.
        """
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
        """Test whether a number is prime.

        Args:
            n (int): The number to test.

        Returns:
            bool: True if n is prime, False otherwise.
        """
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
    def prime_factors(n: int) -> list[int]:
        """Compute the prime factors of a positive integer.

        Args:
            n (int): The integer to factor.

        Returns:
            List[int]: List of prime factors (with repetition).
        """
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
        """Compute Euler's totient function phi(n).

        Args:
            n (int): A positive integer.

        Returns:
            int: The number of integers between 1 and n that are coprime to n.
        """
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

    @staticmethod
    def miller_rabin(n: int, k: int = 10) -> bool:
        """Probabilistic primality test using Miller-Rabin.

        Args:
            n (int): The number to test.
            k (int): Number of random witnesses (default 10).

        Returns:
            bool: True if n is probably prime, False if composite.
        """
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0:
            return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for _ in range(k):
            a = random.randrange(2, n - 1)
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    @staticmethod
    def next_prime(n: int) -> int:
        """Find the smallest prime greater than or equal to n.

        Args:
            n (int): The lower bound.

        Returns:
            int: The smallest prime >= n.
        """
        if n <= 2:
            return 2
        cand = n if n % 2 == 1 else n + 1
        while True:
            if NumberTheory.is_prime(cand):
                return cand
            cand += 2

    @staticmethod
    def nth_prime(n: int) -> int:
        """Return the n-th prime (1-indexed).

        Args:
            n (int): The index (>= 1).

        Returns:
            int: The n-th prime.
        """
        if n < 1:
            raise AxiomError("n must be >= 1")
        if n == 1:
            return 2
        count = 1
        cand = 3
        while count < n:
            if NumberTheory.is_prime(cand):
                count += 1
            cand += 2
        return cand - 2

    @staticmethod
    def legendre_symbol(a: int, p: int) -> int:
        """Compute the Legendre symbol (a|p) for odd prime p.

        Args:
            a (int): The integer.
            p (int): An odd prime.

        Returns:
            int: 1 if a is quadratic residue mod p, -1 if non-residue, 0 if a % p == 0.
        """
        if a % p == 0:
            return 0
        return 1 if pow(a, (p - 1) // 2, p) == 1 else -1

    @staticmethod
    def jacobi_symbol(a: int, n: int) -> int:
        """Compute the Jacobi symbol (a|n) for odd positive n.

        Args:
            a (int): The integer.
            n (int): An odd positive integer.

        Returns:
            int: The Jacobi symbol value.
        """
        if n <= 0 or n % 2 == 0:
            raise AxiomError("n must be odd and positive")
        t = 1
        while a != 0:
            while a % 2 == 0:
                a //= 2
                r = n % 8
                if r in (3, 5):
                    t = -t
            a, n = n, a
            if a % 4 == 3 and n % 4 == 3:
                t = -t
            a %= n
        return t if n == 1 else 0

    @staticmethod
    def discrete_log(g: int, h: int, p: int) -> int | None:
        """Solve g^x ≡ h (mod p) using baby-step giant-step.

        Args:
            g (int): The generator.
            h (int): The target value.
            p (int): The prime modulus.

        Returns:
            Optional[int]: The discrete logarithm x, or None if not found.
        """
        m = math.isqrt(p) + 1
        baby = {}
        cur = 1
        for j in range(m):
            if cur not in baby:
                baby[cur] = j
            cur = (cur * g) % p
        factor = pow(g, -m, p)
        cur = h
        for i in range(m):
            if cur in baby:
                return i * m + baby[cur]
            cur = (cur * factor) % p
        return None

    @staticmethod
    def fibonacci(n: int) -> int:
        """Compute the n-th Fibonacci number (F_0 = 0, F_1 = 1) via fast doubling.

        Args:
            n (int): Non-negative index.

        Returns:
            int: The n-th Fibonacci number.
        """
        if n < 0:
            raise AxiomError("n must be >= 0")

        def fib_pair(k):
            if k == 0:
                return 0, 1
            a, b = fib_pair(k >> 1)
            c = a * (2 * b - a)
            d = a * a + b * b
            if k & 1:
                return d, c + d
            return c, d

        return fib_pair(n)[0]

    @staticmethod
    def lucas(n: int) -> int:
        """Compute the n-th Lucas number (L_0 = 2, L_1 = 1) via L(n) = F(n-1) + F(n+1).

        Args:
            n (int): Non-negative index.

        Returns:
            int: The n-th Lucas number.
        """
        if n < 0:
            raise AxiomError("n must be >= 0")
        if n == 0:
            return 2
        return NumberTheory.fibonacci(n - 1) + NumberTheory.fibonacci(n + 1)
