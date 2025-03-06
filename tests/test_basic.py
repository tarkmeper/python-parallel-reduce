import pytest
from functools import reduce
import parallelreduce
import math


def _test_func(x, y):
    return x + y


def test_empty():
    with pytest.raises(ValueError):
        parallelreduce.parallel_reduce(_test_func, [])


def test_single_element():
    assert parallelreduce.parallel_reduce(_test_func, [1]) == 1


def test_smaller_than_chunk():
    assert parallelreduce.parallel_reduce(_test_func, [1, 2, 3], chunk_size=4) == 6


def test_large_data_set():
    assert parallelreduce.parallel_reduce(_test_func, list(range(1000))) == 499500


def test_large_data_set_init():
    assert parallelreduce.parallel_reduce(_test_func, list(range(1000)), 0) == 499500


def _prime_factors(x):
    # find all the  prime factors of x
    divisor = 2
    while x > 1:
        while x % divisor == 0:
            yield divisor
            x //= divisor
        divisor += 1


def _complex_test_func(x, y):
    # A more compute intesive function since a trivial function means that the algorithm to break to parallel
    # we are going to count how many prime factors exist for all the numbers in the range.
    if isinstance(y, dict):
        for k in y:
            x[k] = y[k] if k not in x else x[k] + y[k]
    else:
        for divisor in _prime_factors(y):
            if divisor not in x:
                x[divisor] = 1
            else:
                x[divisor] += 1
    return x


@pytest.mark.benchmark
@pytest.mark.parametrize('mode', ['parallel', 'single'])
def test_performance(benchmark, mode):
    lst = list(range(50000))

    def parallel(execution_mode):
        if execution_mode == 'single':
            return reduce(_complex_test_func, lst, {})
        else:
            return parallelreduce.parallel_reduce(_complex_test_func, lst, {})

    benchmark.pedantic(parallel, (mode,), rounds=1)
