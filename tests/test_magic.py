from axiompy import Axiom


def test_timer_decorator(capsys):
    @Axiom.magic.timer
    def work():
        return 42

    assert work() == 42
    captured = capsys.readouterr()
    assert "took" in captured.out


def test_pipe():
    result = Axiom.magic.pipe(5, lambda x: x * 2, lambda x: x + 1)
    assert result == 11


def test_compose():
    add1 = lambda x: x + 1
    double = lambda x: x * 2
    f = Axiom.magic.compose(add1, double)
    assert f(5) == 11  # add1(double(5)) = 11


def test_magic_memoize():
    call_count = 0

    @Axiom.magic.magic(memoize=True)
    def fn(x):
        nonlocal call_count
        call_count += 1
        return x * 2

    assert fn(3) == 6
    assert call_count == 1
    assert fn(3) == 6
    assert call_count == 1  # memoized


# ---- Number basics ----

def test_digit_sum():
    assert Axiom.magic.digit_sum(123) == 6


def test_digital_root():
    assert Axiom.magic.digital_root(12345) == 6


def test_is_palindrome():
    assert Axiom.magic.is_palindrome(121) is True
    assert Axiom.magic.is_palindrome(123) is False


def test_reverse_number():
    assert Axiom.magic.reverse_number(123) == 321
    assert Axiom.magic.reverse_number(-123) == -321


def test_collatz():
    seq = Axiom.magic.collatz(6)
    assert seq[0] == 6
    assert seq[-1] == 1


def test_happy_numbers():
    happy = Axiom.magic.happy_numbers(20)
    assert 1 in happy
    assert 7 in happy
    assert 10 in happy


def test_armstrong_number():
    assert Axiom.magic.armstrong_number(153) is True
    assert Axiom.magic.armstrong_number(10) is False


def test_perfect_number():
    assert Axiom.magic.perfect_number(28) is True
    assert Axiom.magic.perfect_number(10) is False


def test_friendly_numbers():
    assert Axiom.magic.friendly_numbers(6, 28) is True
    assert Axiom.magic.friendly_numbers(6, 10) is False


# ---- New sieve ----

def test_sieve_of_eratosthenes():
    assert Axiom.magic.sieve_of_eratosthenes(20) == [2, 3, 5, 7, 11, 13, 17, 19]
    assert Axiom.magic.sieve_of_eratosthenes(1) == []


# ---- Kaprekar ----

def test_kaprekar_routine():
    seq = Axiom.magic.kaprekar_routine(3524)
    assert seq == [3524, 3087, 8352, 6174]
    assert seq[-1] == 6174


# ---- Look and say ----

def test_look_and_say():
    assert Axiom.magic.look_and_say(5) == [1, 11, 21, 1211, 111221]
    assert Axiom.magic.look_and_say(1) == [1]


# ---- Ulam spiral ----

def test_ulam_spiral():
    spiral = Axiom.magic.ulam_spiral(10)
    assert len(spiral) == 10
    # first point is (0, 0, 1)
    assert spiral[0] == (0, 0, 1)


# ---- Narcissistic numbers ----

def test_narcissistic_numbers():
    nums = Axiom.magic.narcissistic_numbers(200)
    assert 153 in nums
    assert 10 not in nums


# ---- Smith numbers ----

def test_smith_numbers():
    smiths = Axiom.magic.smith_numbers(30)
    assert smiths == [4, 22, 27]


# ---- Emirp numbers ----

def test_emirp_numbers():
    emirps = Axiom.magic.emirp_numbers(50)
    assert 13 in emirps
    assert 17 in emirps
    assert 11 not in emirps  # palindrome, excluded


# ---- Goldbach ----

def test_goldbach_conjecture():
    parts = Axiom.magic.goldbach_conjecture(20)
    assert (3, 17) in parts
    assert (7, 13) in parts
    # odd numbers return empty
    assert Axiom.magic.goldbach_conjecture(21) == []


# ---- Twin primes ----

def test_twin_primes():
    twins = Axiom.magic.twin_primes(30)
    assert (3, 5) in twins
    assert (5, 7) in twins
    assert (11, 13) in twins


# ---- Circular primes ----

def test_circular_primes():
    circ = Axiom.magic.circular_primes(100)
    assert 2 in circ
    assert 13 in circ
    assert 37 in circ


# ---- Number to words ----

def test_number_to_words():
    assert Axiom.magic.number_to_words(0) == "zero"
    assert Axiom.magic.number_to_words(42) == "forty-two"
    assert Axiom.magic.number_to_words(2024) == "two thousand twenty-four"
    assert Axiom.magic.number_to_words(-7) == "negative seven"


# ---- Roman numeral ----

def test_roman_numeral():
    assert Axiom.magic.roman_numeral(42) == "XLII"
    assert Axiom.magic.roman_numeral(2024) == "MMXXIV"
    assert Axiom.magic.roman_numeral(1) == "I"
    assert Axiom.magic.roman_numeral(3999) == "MMMCMXCIX"
    import pytest
    with pytest.raises(ValueError):
        Axiom.magic.roman_numeral(0)


# ---- Factorial digit sum ----

def test_factorial_digit_sum():
    assert Axiom.magic.factorial_digit_sum(10) == 27


# ---- Fibonacci spiral ----

def test_fibonacci_spiral():
    assert Axiom.magic.fibonacci_spiral(10) == [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    assert Axiom.magic.fibonacci_spiral(1) == [1]
    assert Axiom.magic.fibonacci_spiral(0) == []


# ---- Automorphic numbers ----

def test_automorphic_numbers():
    assert Axiom.magic.automorphic_numbers(100) == [1, 5, 6, 25, 76]


# ---- Harshad numbers ----

def test_harshad_numbers():
    harshad = Axiom.magic.harshad_numbers(30)
    assert 1 in harshad
    assert 10 in harshad
    assert 12 in harshad
    assert 13 not in harshad


# ---- Visualize number ----

def test_visualize_number():
    viz = Axiom.magic.visualize_number(8128)
    assert "Number: 8128" in viz
    assert "Digit sum:" in viz
    assert "Roman:" in viz
    assert "Binary:" in viz
