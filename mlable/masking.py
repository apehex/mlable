import functools

import tensorflow as tf

import mlable.utils

# REDUCE ######################################################################

def reduce_mask(mask: tf.Tensor, operation: callable, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    # original shape
    __shape = mlable.utils.normalize_shape(shape=list(mask.shape))
    # reduction factor on each axis
    __axes = list(range(len(__shape))) if axis is None else [axis % len(__shape)]
    __repeats = mlable.utils.filter_shape(shape=__shape, axes=__axes)
    # actually reduce
    __mask = operation(mask, axis=axis, keepdims=keepdims)
    # repeat the value along the reduced axis
    return tf.tile(input=__mask, multiples=__repeats) if keepdims else __mask

def reduce_any_mask(mask: tf.Tensor, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    return reduce_mask(mask=mask, operation=tf.reduce_any, axis=axis, keepdims=keepdims)

def reduce_all_mask(mask: tf.Tensor, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    return reduce_mask(mask=mask, operation=tf.reduce_all, axis=axis, keepdims=keepdims)

# GROUP #######################################################################

def group_mask(mask: tf.Tensor, operation: callable, group: int, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    # original shape
    __shape = mlable.utils.normalize_shape(mask.shape)
    # normalize axis / orginal shape
    __axis = axis % len(__shape)
    # axes are indexed according to the new shape
    __shape = mlable.utils.divide_shape(shape=__shape, input_axis=__axis, output_axis=-1, factor=group, insert=True)
    # split the last axis
    __mask = tf.reshape(mask, shape=__shape)
    # repeat values to keep the same shape as the original mask
    __mask = reduce_mask(mask=__mask, operation=operation, axis=-1, keepdims=keepdims)
    # match the original shape
    __shape = mlable.utils.merge_shape(shape=__shape, left_axis=__axis, right_axis=-1, left=True)
    # merge the new axis back
    return tf.reshape(__mask, shape=__shape) if keepdims else __mask

def group_any_mask(mask: tf.Tensor, group: int, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    return group_mask(mask=mask, operation=tf.reduce_any, group=group, axis=axis, keepdims=keepdims)

def group_all_mask(mask: tf.Tensor, group: int, axis: int=-1, keepdims: bool=True) -> tf.Tensor:
    return group_mask(mask=mask, operation=tf.reduce_all, group=group, axis=axis, keepdims=keepdims)
