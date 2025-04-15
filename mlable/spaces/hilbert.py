"""Map a flat dimension with points of rank N, according to the Hilbert curve."""

import functools

# BINARY #######################################################################

def encode_binary(number: int, width: int) -> str:
    return format(number, 'b').zfill(width)[:width] # truncated at width

# SHAPING ######################################################################

def transpose_axes(number: int, order: int, rank: int) -> list:
    __bits = encode_binary(number, width=rank * order)
    return [int(__bits[__i::rank], 2) for __i in range(rank)]

def flatten_axes(coords: list, order: int, rank: int) -> int:
    __coords = [encode_binary(__c, width=order) for __c in coords]
    return int(''.join([__y[__i] for __i in range(order) for __y in __coords]), 2)

# GRAY CODES ###################################################################

def encode_gray(number: int) -> int:
    return number ^ (number >> 1)

def decode_gray(number: int) -> int:
    return functools.reduce(
        lambda __a, __b: __a ^ __b,
        [number >> __i for __i in range(len(format(number, 'b')))])

# TANGLE #######################################################################

def entangle(coords: list, order: int, rank: int) -> list:
    __coords = list(coords)
    # undo the extra rotations
    for __j in range(order - 1, 0, -1):
        # q is a single bit mask and (q -1) is a string of ones
        __q = 2 ** __j
        for __i in range(0, rank):
            # invert the least significant bits
            if __coords[__i] & __q:
                __coords[0] ^= __q - 1
            # exchange the least significant bits
            else:
                __t = (__coords[0] ^ __coords[__i]) & (__q - 1)
                __coords[0] ^= __t
                __coords[__i] ^= __t
    # list of rank coordinates
    return __coords

def untangle(coords: list, order: int, rank: int) -> list:
    __coords = list(coords)
    # undo the extra rotations
    for __j in range(1, order):
        # q is a single bit mask and (q -1) is a string of ones
        __q = 2 ** __j
        for __i in range(rank - 1, -1, -1):
            # invert the least significant bits
            if __coords[__i] & __q:
                __coords[0] ^= __q - 1
            # exchange the least significant bits
            else:
                __t = (__coords[0] ^ __coords[__i]) & (__q - 1)
                __coords[0] ^= __t
                __coords[__i] ^= __t
    # list of rank coordinates
    return __coords

# 1D => 2D #####################################################################

def point(position: int, order: int, rank: int) -> list:
    # gray encoding H ^ (H/2)
    __gray = encode_gray(position)
    # approximate the curve
    __coords = transpose_axes(__gray, order=order, rank=rank)
    # Undo excess work
    return untangle(__coords, order=order, rank=rank)

# 2D => 1D #####################################################################

def position(coords: list, order: int, rank: int) -> int:
    # entangle the positions back
    __coords = entangle(coords, order=order, rank=rank)
    # flatten the coordinate
    __position = flatten_axes(__coords, order=order, rank=rank)
    # decode the gray encodings
    return decode_gray(__position)
