import multiprocessing
from functools import reduce
import concurrent.futures
import math
import itertools

TARGET_PARTITIONS_PER_CPU = 10


def __chunk_list(lst, chunk_size):
    """ Simple helper to divide a list into chunks as an iterator. """
    it = iter(lst)
    return iter(lambda: list(itertools.islice(it, chunk_size)), [])


# Replicate the initial treatment in the Python reduce function.
__initial_missing = object()


def parallel_reduce(fn, data, initial=__initial_missing, chunk_size=None, max_workers=None):
    """
    Parallel reduce algoirthm, interface is same as functools reduce with one exception related to initial value.  The
    initital value will be supplied to all processes.

    :param fn:  Function to execute on each element.
    :param data:  Full data list
    :param initial:  Initial value.  Note that if supplied this value will be used on _each_ processor which could
    results in including in multiple times.
    :param chunk_size: Size of the chunks to use for processing.  If not supplied will use a heuristic to determine
    an appropriate size.  Running with a very small number can be slow if overhead is a concern.
    :param max_workers: Total number of processes to spawn.
    :return:
    """
    workers = max_workers if max_workers else multiprocessing.cpu_count()

    # exception cases if the length of the data is not sufficient to start the parallel procssing.
    if len(data) == 0:
        raise ValueError("data must not be empty")
    elif len(data) == 1:
        return data[0]

    # heuristic to determine chunk size.  Chunk-size of 1 is extremely slow in most cases since the time to start
    # and stop executors can be a problem.  However, we do want enough chunks that it will somewhat average out any
    # long-running and faster sections of the code.  We aim for a reasonable number of chunks per CPU to help average
    # out if one of the chunks is significantly slower than the others.
    if chunk_size is None:
        chunk_size = math.ceil(max(math.sqrt(len(data)), len(data) / (TARGET_PARTITIONS_PER_CPU * workers)))

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers - 1) as executor:
        if initial is __initial_missing:
            result_futures = [executor.submit(reduce, fn, sublst) for sublst in __chunk_list(data, chunk_size)]
        else:
            result_futures = [executor.submit(reduce, fn, sublst, initial) for sublst in __chunk_list(data, chunk_size)]

        it = concurrent.futures.as_completed(result_futures)
        final_initial = next(it).result()
        result = reduce(lambda x, y: fn(x, y.result()), it, final_initial)

    return result
