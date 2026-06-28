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
