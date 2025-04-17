import functools
import itertools

# FUNCTIONS ####################################################################

compose = lambda __l: (lambda __x: functools.reduce(lambda __e, __f: __f(__e), __l, __x))

distribute = lambda __f: (lambda *__t: tuple(map(__f, __t)))

# SPLIT ########################################################################

def chunk(seq: list, size: int, repeats: bool=True) -> list:
    __chunks = (seq[__i:__i + size] for __i in range(0, len(seq), size))
    return list(__chunks if repeats else set(__chunks))

def merge(chunks: list) -> list:
    return list(itertools.chain.from_iterable(chunks))

# PERMUTATION ##################################################################

def rotate(sequence: list, ticks: int) -> list:
    __n = ticks % len(sequence)
    return sequence[__n:] + sequence[:__n] # shift left if ticks > 0 right otherwise
