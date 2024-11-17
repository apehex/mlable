import tensorflow as tf

# PERMUTATION ##################################################################

def rotate(sequence: list, ticks: int) -> list:
    __n = ticks % len(sequence)
    return sequence[__n:] + sequence[:__n] # shift left if ticks > 0 right otherwise

# AXES ########################################################################

def swap_axes(rank: int, left: int, right: int, perm: list=[]) -> list:
    __perm = perm if perm else list(range(rank))
    __left, __right = left % rank, right % rank
    __perm[__left], __perm[__right] = __perm[__right], __perm[__left]
    return __perm

def move_axis(rank: int, before: int, after: int, perm: list=[]) -> list:
    __perm = perm if perm else list(range(rank))
    # indexes
    __from = before % len(__perm)
    __to = after % len(__perm)
    # rotate left to right if from < to and vice-versa
    __dir = 1 if __from < __to else -1
    # split the sequence
    __left = __perm[:min(__from, __to)]
    __shift = rotate(__perm[min(__from, __to):max(__from, __to) + 1], ticks=__dir)
    __right = __perm[max(__from, __to) + 1:]
    # recompose
    return __left + __shift + __right

# DIMS ########################################################################

def normalize_dim(dim: int) -> int:
    return -1 if (dim is None or dim < 0) else dim

def multiply_dim(dim_l: int, dim_r: int) -> int:
    return -1 if (dim_l == -1 or dim_r == -1) else dim_l * dim_r

def divide_dim(dim_l: int, dim_r: int) -> int:
    return -1 if (dim_l == -1 or dim_r == -1) else dim_l // dim_r

# NORMALIZE ###################################################################

def normalize_shape(shape: list) -> list:
    return [normalize_dim(dim=__d) for __d in list(shape)]

def filter_shape(shape: list, axes: list) -> list:
    __shape = normalize_shape(shape)
    __axes = [__a % len(__shape) for __a in axes] # interpret negative indexes
    return [__d if __i in __axes else 1 for __i, __d in enumerate(__shape)]

# DIVIDE ######################################################################

def divide_shape(shape: list, input_axis: int, output_axis: int, factor: int, insert: bool=False) -> list:
    # copy
    __shape = normalize_shape(shape)
    # rank, according to the new shape
    __rank = len(__shape) + int(insert)
    # axes, taken from the new shape
    __axis0 = input_axis % __rank
    __axis1 = output_axis % __rank
    # option to group data on a new axis
    if insert: __shape.insert(__axis1, 1)
    # move data from axis 0 to axis 1
    __shape[__axis0] = divide_dim(__shape[__axis0], factor)
    __shape[__axis1] = multiply_dim(__shape[__axis1], factor)
    # return
    return __shape

def divide(data: tf.Tensor, input_axis: int, output_axis: int, factor: int, insert: bool=False) -> tf.Tensor:
    # move data from input axis to output axis
    __shape = divide_shape(shape=list(data.shape), input_axis=input_axis, output_axis=output_axis, factor=factor, insert=insert)
    # actually reshape
    return tf.reshape(tensor=data, shape=__shape)

# MERGE #######################################################################

def merge_shape(shape: list, left_axis: int, right_axis: int, left: bool=True) -> list:
    # copy
    __shape = normalize_shape(shape)
    __rank = len(__shape)
    # normalize (negative indexes)
    __axis_l = left_axis % __rank
    __axis_r = right_axis % __rank
    # new dimension
    __dim = multiply_dim(__shape[__axis_l], __shape[__axis_r])
    # select axes
    __axis_k = __axis_l if left else __axis_r # kept axis
    __axis_d = __axis_r if left else __axis_l # deleted axis
    # new shape
    __shape[__axis_k] = __dim
    __shape.pop(__axis_d)
    # return
    return __shape

def merge_to_same_rank(x1: tf.Tensor, x2: tf.Tensor) -> tuple:
    # init
    __x1, __x2 = x1, x2
    __s1, __s2 = list(__x1.shape), list(__x2.shape)
    __r1, __r2 = len(__s1), len(__s2)
    # x1 has one more axis
    if __r1 == __r2 + 1:
        __s1 = merge_shape(shape=__s1, left_axis=-2, right_axis=-1, left=True)
        __x1 = tf.reshape(__x1, shape=__s1)
    # x2 has one more axis
    if __r2 == __r1 + 1:
        __s2 = merge_shape(shape=__s2, left_axis=-2, right_axis=-1, left=True)
        __x2 = tf.reshape(__x2, shape=__s2)
    # return both
    return __x1, __x2

def merge(data: tf.Tensor, left_axis: int, right_axis: int, left: bool=True) -> tf.Tensor:
    # new shape
    __shape = merge_shape(shape=list(data.shape), left_axis=left_axis, right_axis=right_axis, left=left)
    # actually merge the two axes
    return tf.reshape(tensor=data, shape=__shape)
