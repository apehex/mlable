import functools

import tensorflow as tf

import mlable.utils

# REDUCE ######################################################################

def _reduce(tensor: tf.Tensor, operation: callable, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    # original shape
    __shape = mlable.utils.normalize_shape(shape=list(tensor.shape))
    # reduction factor on each axis
    __axes = list(range(len(__shape))) if axis is None else [axis % len(__shape)]
    __repeats = mlable.utils.filter_shape(shape=__shape, axes=__axes)
    # actually reduce
    __tensor = operation(tensor, axis=axis, keepdims=keepdims)
    # repeat the value along the reduced axis
    return tf.tile(input=__tensor, multiples=__repeats) if keepdims else __tensor

def _reduce_any(tensor: tf.Tensor, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    return _reduce(tensor=tensor, operation=tf.reduce_any, axis=axis, keepdims=keepdims)

def _reduce_all(tensor: tf.Tensor, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    return _reduce(tensor=tensor, operation=tf.reduce_all, axis=axis, keepdims=keepdims)

# GROUP #######################################################################

def _reduce_group_by_group(tensor: tf.Tensor, operation: callable, group: int, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    # original shape
    __shape = mlable.utils.normalize_shape(tensor.shape)
    # normalize axis / orginal shape
    __axis = axis % len(__shape)
    # axes are indexed according to the new shape
    __shape = mlable.utils.divide_shape(shape=__shape, input_axis=__axis, output_axis=-1, factor=group, insert=True)
    # split the last axis
    __tensor = tf.reshape(tensor, shape=__shape)
    # repeat values to keep the same shape as the original tensor
    __tensor = _reduce(tensor=__tensor, operation=operation, axis=-1, keepdims=keepdims)
    # match the original shape
    __shape = mlable.utils.merge_shape(shape=__shape, left_axis=__axis, right_axis=-1, left=True)
    # merge the new axis back
    return tf.reshape(__tensor, shape=__shape) if keepdims else __tensor

def _reduce_group_by_group_any(tensor: tf.Tensor, group: int, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    return _reduce_group_by_group(tensor=tensor, operation=tf.reduce_any, group=group, axis=axis, keepdims=keepdims)

def _reduce_group_by_group_all(tensor: tf.Tensor, group: int, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    return _reduce_group_by_group(tensor=tensor, operation=tf.reduce_all, group=group, axis=axis, keepdims=keepdims)

# BASE ########################################################################

def _reduce_base(tensor: tf.Tensor, base: int, axis: int=-1, keepdims: bool=False) -> tf.Tensor:
    # select the dimension of the given axis
    __shape = mlable.utils.filter_shape(shape=tensor.shape, axes=[axis])
    # exponents
    __exp = range(__shape[axis])
    # base, in big endian
    __base = tf.convert_to_tensor([base ** __e for __e in __exp[::-1]], dtype=tensor.dtype)
    # match the input shape
    __base = tf.reshape(__base, shape=__shape)
    # recompose the number
    return tf.reduce_sum(tensor * __base, axis=axis, keepdims=keepdims)

# API #########################################################################

def reduce(tensor: tf.Tensor, operation: callable, group: int=0, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    if isinstance(axis, int) and isinstance(group, int) and group > 0:
        return _reduce_group_by_group(tensor=tensor, operation=operation, group=group, axis=axis, keepdims=keepdims)
    else:
        return _reduce(tensor=tensor, operation=operation, axis=axis, keepdims=keepdims)

def reduce_any(tensor: tf.Tensor, group: int=0, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    return reduce(tensor=tensor, operation=tf.reduce_any, group=group, axis=axis, keepdims=keepdims)

def reduce_all(tensor: tf.Tensor, group: int=0, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    return reduce(tensor=tensor, operation=tf.reduce_all, group=group, axis=axis, keepdims=keepdims)

def reduce_base(tensor: tf.Tensor, base: int, group: int=0, axis: int=-1, keepdims: bool=False) -> tf.Tensor:
    __operation = functools.partial(_reduce_base, base=base)
    return reduce(tensor=tensor, operation=__operation, group=group, axis=axis, keepdims=keepdims)