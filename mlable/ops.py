import functools

import tensorflow as tf

import mlable.shaping

# REDUCE ######################################################################

def _reduce(data: tf.Tensor, operation: callable, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    # original shape
    __shape = mlable.shaping.normalize_shape(shape=list(data.shape))
    # reduction factor on each axis
    __axes = list(range(len(__shape))) if axis is None else [axis % len(__shape)]
    __repeats = mlable.shaping.filter_shape(shape=__shape, axes=__axes)
    # actually reduce
    __data = operation(data, axis=axis, keepdims=keepdims)
    # repeat the value along the reduced axis
    return tf.tile(input=__data, multiples=__repeats) if keepdims else __data

def _reduce_any(data: tf.Tensor, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    return _reduce(data=data, operation=tf.reduce_any, axis=axis, keepdims=keepdims)

def _reduce_all(data: tf.Tensor, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    return _reduce(data=data, operation=tf.reduce_all, axis=axis, keepdims=keepdims)

# GROUP #######################################################################

def _reduce_group_by_group(data: tf.Tensor, operation: callable, group: int, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    # original shape
    __shape = mlable.shaping.normalize_shape(data.shape)
    # interpret negative axis index / orginal shape
    __axis = axis % len(__shape)
    # split the last axis
    __data = mlable.shaping.divide(data=data, input_axis=__axis, output_axis=__axis + 1, factor=group, insert=True)
    # repeat values to keep the same shape as the original tensor
    __data = _reduce(data=__data, operation=operation, axis=__axis + 1, keepdims=keepdims)
    # merge the new axis back
    return mlable.shaping.merge(data=__data, left_axis=__axis, right_axis=__axis + 1, left=True) if keepdims else __data

def _reduce_group_by_group_any(data: tf.Tensor, group: int, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    return _reduce_group_by_group(data=data, operation=tf.reduce_any, group=group, axis=axis, keepdims=keepdims)

def _reduce_group_by_group_all(data: tf.Tensor, group: int, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    return _reduce_group_by_group(data=data, operation=tf.reduce_all, group=group, axis=axis, keepdims=keepdims)

# BASE ########################################################################

def _reduce_base(data: tf.Tensor, base: int, axis: int=-1, keepdims: bool=False, bigendian: bool=True) -> tf.Tensor:
    # select the dimension of the given axis
    __shape = mlable.shaping.filter_shape(shape=data.shape, axes=[axis])
    # exponents
    __exp = range(__shape[axis])[::-1] if bigendian else range(__shape[axis])
    # base, in big endian
    __base = tf.convert_to_tensor([base ** __e for __e in __exp], dtype=data.dtype)
    # match the input shape
    __base = tf.reshape(__base, shape=__shape)
    # recompose the number
    return tf.reduce_sum(data * __base, axis=axis, keepdims=keepdims)

def expand_base(data: tf.Tensor, base: int, depth: int, bigendian: bool=True) -> tf.Tensor:
    __shape = len(list(data.shape)) * [1] + [depth]
    # base indexes
    __idx = range(depth)[::-1] if bigendian else range(depth)
    # base divisor and modulus
    __div = tf.convert_to_tensor([base ** __e for __e in __idx], dtype=data.dtype)
    __mod = tf.convert_to_tensor([base ** (__e + 1) for __e in __idx], dtype=data.dtype)
    # match the input shape
    __div = tf.reshape(__div, shape=__shape)
    __mod = tf.reshape(__mod, shape=__shape)
    # Euclidean algorithm
    __digits = tf.math.floordiv(x=tf.math.floormod(x=tf.expand_dims(data, axis=-1), y=__mod), y=__div)
    # format
    return tf.cast(__digits, dtype=data.dtype)

# API #########################################################################

def reduce(data: tf.Tensor, operation: callable, group: int=0, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    if isinstance(axis, int) and isinstance(group, int) and group > 0:
        return _reduce_group_by_group(data=data, operation=operation, group=group, axis=axis, keepdims=keepdims)
    else:
        return _reduce(data=data, operation=operation, axis=axis, keepdims=keepdims)

def reduce_any(data: tf.Tensor, group: int=0, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    return reduce(data=data, operation=tf.reduce_any, group=group, axis=axis, keepdims=keepdims)

def reduce_all(data: tf.Tensor, group: int=0, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    return reduce(data=data, operation=tf.reduce_all, group=group, axis=axis, keepdims=keepdims)

def reduce_base(data: tf.Tensor, base: int, group: int=0, axis: int=-1, keepdims: bool=False) -> tf.Tensor:
    __operation = functools.partial(_reduce_base, base=base)
    return reduce(data=data, operation=__operation, group=group, axis=axis, keepdims=keepdims)
