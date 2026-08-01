import pytest

from cpp_python_project import core


def test_add():
    assert core.add(2, 3) == 5
    assert core.add(-1, 1) == 0


def test_fibonacci():
    assert core.fibonacci(0) == 0
    assert core.fibonacci(1) == 1
    assert core.fibonacci(10) == 55
    assert core.fibonacci(20) == 6765


def test_scale_vector():
    result = core.scale_vector([1.0, 2.0, 3.0], 2.0)
    assert result == [2.0, 4.0, 6.0]


def test_counter():
    counter = core.Counter(5)
    assert counter.value() == 5

    counter.increment()
    assert counter.value() == 6

    counter.increment(4)
    assert counter.value() == 10

    counter.reset()
    assert counter.value() == 0