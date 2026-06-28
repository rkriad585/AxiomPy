import pytest
from axiompy import Axiom
from axiompy._base import AxiomError


class TestNumberTheory:
    def test_extended_gcd(self):
        g, x, y = Axiom.number_theory.extended_gcd(30, 20)
        assert g == 10
        assert 30 * x + 20 * y == g

    def test_mod_inverse(self):
        inv = Axiom.number_theory.mod_inverse(7, 26)
        assert (7 * inv) % 26 == 1

    def test_mod_inverse_error(self):
        with pytest.raises(AxiomError):
            Axiom.number_theory.mod_inverse(2, 4)

    def test_chinese_remainder(self):
        result = Axiom.number_theory.chinese_remainder_theorem([3, 5, 7], [2, 3, 2])
        assert result == 23

    def test_sieve(self):
        primes = Axiom.number_theory.sieve_of_eratosthenes(30)
        assert primes == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    def test_sieve_empty(self):
        assert Axiom.number_theory.sieve_of_eratosthenes(1) == []

    def test_is_prime(self):
        assert Axiom.number_theory.is_prime(2) is True
        assert Axiom.number_theory.is_prime(17) is True
        assert Axiom.number_theory.is_prime(1) is False
        assert Axiom.number_theory.is_prime(4) is False

    def test_prime_factors(self):
        assert Axiom.number_theory.prime_factors(12) == [2, 2, 3]

    def test_prime_factors_prime(self):
        assert Axiom.number_theory.prime_factors(7) == [7]

    def test_euler_totient(self):
        assert Axiom.number_theory.euler_totient(10) == 4
        assert Axiom.number_theory.euler_totient(7) == 6
        assert Axiom.number_theory.euler_totient(1) == 1

    def test_miller_rabin(self):
        assert Axiom.number_theory.miller_rabin(7) is True
        assert Axiom.number_theory.miller_rabin(4) is False
        assert Axiom.number_theory.miller_rabin(7919) is True
        assert Axiom.number_theory.miller_rabin(1) is False

    def test_next_prime(self):
        assert Axiom.number_theory.next_prime(2) == 2
        assert Axiom.number_theory.next_prime(10) == 11
        assert Axiom.number_theory.next_prime(100) == 101

    def test_nth_prime(self):
        assert Axiom.number_theory.nth_prime(1) == 2
        assert Axiom.number_theory.nth_prime(10) == 29
        with pytest.raises(AxiomError):
            Axiom.number_theory.nth_prime(0)

    def test_legendre_symbol(self):
        assert Axiom.number_theory.legendre_symbol(1, 7) == 1
        assert Axiom.number_theory.legendre_symbol(2, 7) == 1
        assert Axiom.number_theory.legendre_symbol(3, 7) == -1
        assert Axiom.number_theory.legendre_symbol(7, 7) == 0

    def test_jacobi_symbol(self):
        assert Axiom.number_theory.jacobi_symbol(1, 7) == 1
        assert Axiom.number_theory.jacobi_symbol(2, 15) == 1
        assert Axiom.number_theory.jacobi_symbol(7, 15) == -1
        with pytest.raises(AxiomError):
            Axiom.number_theory.jacobi_symbol(1, 4)

    def test_discrete_log(self):
        x = Axiom.number_theory.discrete_log(2, 8, 13)
        assert x is not None
        assert pow(2, x, 13) == 8
        assert Axiom.number_theory.discrete_log(2, 7, 13) is not None

    def test_fibonacci(self):
        assert Axiom.number_theory.fibonacci(0) == 0
        assert Axiom.number_theory.fibonacci(1) == 1
        assert Axiom.number_theory.fibonacci(10) == 55
        with pytest.raises(AxiomError):
            Axiom.number_theory.fibonacci(-1)

    def test_lucas(self):
        assert Axiom.number_theory.lucas(0) == 2
        assert Axiom.number_theory.lucas(1) == 1
        assert Axiom.number_theory.lucas(10) == 123
        with pytest.raises(AxiomError):
            Axiom.number_theory.lucas(-1)
