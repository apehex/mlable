import mlable.utils

# DIMS ########################################################################

def normalize_dim(dim: int) -> int:
    return 0 if (dim is None) else dim

def symbolic_dim(dim: int) -> int:
    return None if (dim == 0) else dim

def multiply_dim(dim_l: int, dim_r: int) -> int:
    return -1 if (dim_l == -1 or dim_r == -1) else dim_l * dim_r

def divide_dim(dim_l: int, dim_r: int) -> int:
    return -1 if (dim_l == -1 or dim_r == -1 or dim_r == 0) else dim_l // dim_r

# NORMALIZE ###################################################################

def normalize(shape: list) -> list:
    return [normalize_dim(dim=__d) for __d in list(shape)]

def symbolic(shape: list) -> list:
    return [symbolic_dim(dim=__d) for __d in list(shape)]

def filter(shape: list, axes: list) -> list:
    __shape = normalize(shape)
    __axes = [__a % len(__shape) for __a in axes] # interpret negative indexes
    return [__d if __i in __axes else 1 for __i, __d in enumerate(__shape)]

# DIVIDE ######################################################################

def divide(shape: list, axis: int, factor: int, insert: bool=False, right: bool=True) -> list:
    # cast all dims to int
    __shape = normalize(shape)
    # the source axis is positioned according to the original shape
    __axis0 = (axis % len(__shape)) + int(insert) * int(not right)
    # the destination axis is right before or right after
    __axis1 = __axis0 + int(right) - int(not right)
    # option to group data on a new axis
    if insert: __shape.insert(__axis1, 1)
    # move data from axis 0 to axis 1
    __shape[__axis0] = divide_dim(__shape[__axis0], factor)
    __shape[__axis1] = multiply_dim(__shape[__axis1], factor)
    # list of ints
    return __shape

# MERGE #######################################################################

def merge(shape: list, axis: int, right: bool=True) -> list:
    # copy
    __shape = normalize(shape)
    __rank = len(__shape)
    # avoid negative indexes for comparisons
    __axis0 = axis % __rank
    # the destination axis is right beside the origin
    __axis1 = (__axis0 + int(right) - int(not right)) % __rank
    # new dimension
    __dim = multiply_dim(__shape[__axis0], __shape[__axis1])
    # new shape
    __shape[min(__axis0, __axis1)] = __dim
    __shape.pop(max(__axis0, __axis1))
    # list of ints
    return __shape

# SWAP #########################################################################

def swap(shape: list, left: int, right: int) -> list:
    __shape = normalize(shape)
    __left, __right = left % len(__shape), right % len(__shape)
    __shape[__left], __shape[__right] = __shape[__right], __shape[__left]
    return __shape

# MOVE #########################################################################

def move(shape: list, before: int, after: int) -> list:
    __shape = normalize(shape)
    # indexes
    __from = before % len(__shape)
    __to = after % len(__shape)
    # rotate left to right if from < to and vice-versa
    __dir = 1 if __from < __to else -1
    # split the sequence
    __left = __shape[:min(__from, __to)]
    __shift = mlable.utils.rotate(__shape[min(__from, __to):max(__from, __to) + 1], ticks=__dir)
    __right = __shape[max(__from, __to) + 1:]
    # recompose
    return __left + __shift + __right
