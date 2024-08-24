import functools

import tensorflow as tf

import mlable.utils

# REDUCE ######################################################################

def _reduce(data: tf.Tensor, operation: callable, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    # original shape
    __shape = mlable.utils.normalize_shape(shape=list(data.shape))
    # reduction factor on each axis
    __axes = list(range(len(__shape))) if axis is None else [axis % len(__shape)]
    __repeats = mlable.utils.filter_shape(shape=__shape, axes=__axes)
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
    __shape = mlable.utils.normalize_shape(data.shape)
    # normalize axis / orginal shape
    __axis = axis % len(__shape)
    # axes are indexed according to the new shape
    __shape = mlable.utils.divide_shape(shape=__shape, input_axis=__axis, output_axis=-1, factor=group, insert=True)
    # split the last axis
    __data = tf.reshape(data, shape=__shape)
    # repeat values to keep the same shape as the original tensor
    __data = _reduce(data=__data, operation=operation, axis=-1, keepdims=keepdims)
    # match the original shape
    __shape = mlable.utils.merge_shape(shape=__shape, left_axis=__axis, right_axis=-1, left=True)
    # merge the new axis back
    return tf.reshape(__data, shape=__shape) if keepdims else __data

def _reduce_group_by_group_any(data: tf.Tensor, group: int, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    return _reduce_group_by_group(data=data, operation=tf.reduce_any, group=group, axis=axis, keepdims=keepdims)

def _reduce_group_by_group_all(data: tf.Tensor, group: int, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    return _reduce_group_by_group(data=data, operation=tf.reduce_all, group=group, axis=axis, keepdims=keepdims)

# BASE ########################################################################

def _reduce_base(data: tf.Tensor, base: int, axis: int=-1, keepdims: bool=False) -> tf.Tensor:
    # select the dimension of the given axis
    __shape = mlable.utils.filter_shape(shape=data.shape, axes=[axis])
    # exponents
    __exp = range(__shape[axis])
    # base, in big endian
    __base = tf.convert_to_tensor([base ** __e for __e in __exp[::-1]], dtype=data.dtype)
    # match the input shape
    __base = tf.reshape(__base, shape=__shape)
    # recompose the number
    return tf.reduce_sum(data * __base, axis=axis, keepdims=keepdims)

def expand_base(data: tf.Tensor, base: int, depth: int) -> tf.Tensor:
    __shape = len(list(data.shape)) * [1] + [depth]
    # base indexes, in big endian
    __idx = range(depth)[::-1]
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

# RESHAPING ###################################################################

def divide(data: tf.Tensor, input_axis: int, output_axis: int, factor: int, insert: bool=False) -> tf.Tensor:
    # move data from input axis to output axis
    __shape = mlable.utils.divide_shape(shape=list(data.shape), input_axis=input_axis, output_axis=output_axis, factor=factor, insert=insert)
    # actually reshape
    return tf.reshape(tensor=data, shape=__shape)

def merge(data: tf.Tensor, left_axis: int, right_axis: int, left: bool=True) -> tf.Tensor:
    # new shape
    __shape = mlable.utils.merge_shape(shape=list(data.shape), left_axis=left_axis, right_axis=right_axis, left=left)
    # actually merge the two axes
    return tf.reshape(tensor=data, shape=__shape)
