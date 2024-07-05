import functools

import tensorflow as tf

import mlable.utils

# REDUCE ######################################################################

def _reduce(mask: tf.Tensor, operation: callable, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    # original shape
    __shape = mlable.utils.normalize_shape(shape=list(mask.shape))
    # reduction factor on each axis
    __axes = list(range(len(__shape))) if axis is None else [axis % len(__shape)]
    __repeats = mlable.utils.filter_shape(shape=__shape, axes=__axes)
    # actually reduce
    __mask = operation(mask, axis=axis, keepdims=keepdims)
    # repeat the value along the reduced axis
    return tf.tile(input=__mask, multiples=__repeats) if keepdims else __mask

def _reduce_any(mask: tf.Tensor, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    return _reduce(mask=mask, operation=tf.reduce_any, axis=axis, keepdims=keepdims)

def _reduce_all(mask: tf.Tensor, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    return _reduce(mask=mask, operation=tf.reduce_all, axis=axis, keepdims=keepdims)

# GROUP #######################################################################

def _reduce_group_by_group(mask: tf.Tensor, operation: callable, group: int, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    # original shape
    __shape = mlable.utils.normalize_shape(mask.shape)
    # normalize axis / orginal shape
    __axis = axis % len(__shape)
    # axes are indexed according to the new shape
    __shape = mlable.utils.divide_shape(shape=__shape, input_axis=__axis, output_axis=-1, factor=group, insert=True)
    # split the last axis
    __mask = tf.reshape(mask, shape=__shape)
    # repeat values to keep the same shape as the original mask
    __mask = _reduce(mask=__mask, operation=operation, axis=-1, keepdims=keepdims)
    # match the original shape
    __shape = mlable.utils.merge_shape(shape=__shape, left_axis=__axis, right_axis=-1, left=True)
    # merge the new axis back
    return tf.reshape(__mask, shape=__shape) if keepdims else __mask

def _reduce_group_by_group_any(mask: tf.Tensor, group: int, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    return _reduce_group_by_group(mask=mask, operation=tf.reduce_any, group=group, axis=axis, keepdims=keepdims)

def _reduce_group_by_group_all(mask: tf.Tensor, group: int, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    return _reduce_group_by_group(mask=mask, operation=tf.reduce_all, group=group, axis=axis, keepdims=keepdims)

# API #########################################################################

def reduce(mask: tf.Tensor, operation: callable, group: int=0, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    if isinstance(axis, int) and isinstance(group, int) and group > 0:
        return _reduce_group_by_group(mask=mask, operation=operation, group=group, axis=axis, keepdims=keepdims)
    else:
        return _reduce(mask=mask, operation=operation, axis=axis, keepdims=keepdims)

def reduce_any(mask: tf.Tensor, group: int=0, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    return reduce(mask=mask, operation=tf.reduce_any, group=group, axis=axis, keepdims=keepdims)

def reduce_all(mask: tf.Tensor, group: int=0, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    return reduce(mask=mask, operation=tf.reduce_all, group=group, axis=axis, keepdims=keepdims)
