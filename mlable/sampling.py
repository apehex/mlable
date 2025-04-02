import numpy as np
import tensorflow as tf

import mlable.masking
import mlable.ops
import mlable.shaping

# FILTER #######################################################################

def filter_top_k(logits: tf.Tensor, count: int) -> tf.Tensor:
    __dim = int(tuple(logits.shape)[-1])
    # meaningful candidate count
    __count = tf.clip_by_value(count, clip_value_min=1, clip_value_max=__dim)
    # filter and sort the top k values
    __values, __indices = tf.math.top_k(logits, k=__count)
    # select the smallest logits
    __lower = tf.gather(__values, axis=-1, indices=[__count - 1])
    # mask the logits to remove
    __mask = logits < __lower
    # set the filtered logits to -inf
    return mlable.masking.choose(left=logits, right=-np.inf, mask=__mask)

def filter_top_p(logits: tf.Tensor, threshold: tf.Tensor) -> tf.Tensor:
    __dim = int(tuple(logits.shape)[-1])
    # sort the logits descending
    __values, __indices = tf.math.top_k(logits, k=__dim)
    # compute the cumulative probabilities
    __probs = tf.math.cumsum(tf.nn.softmax(__values, axis=-1), axis=-1)
    # identify the probabilities to remove, sorted
    __mask = __probs > threshold
    # always keep at least one token (eg. set the first column to False)
    __mask = tf.concat([tf.zeros_like(__mask[..., :1], dtype=tf.bool), __mask[..., 1:]], axis=-1)
    # lower bound (included) of the logits to keep
    __lower = tf.reduce_min(
        tf.where(__mask, tf.fill(tf.shape(__values), __values.dtype.max), __values),
        axis=-1,
        keepdims=True)
    # mask the logits to remove, in the original (scattered) order
    __mask = logits < __lower
    # set filtered logits to -inf
    return mlable.masking.choose(left=logits, right=-np.inf, mask=__mask)

# CATEGORICAL ##################################################################

def categorical(logits: tf.Tensor, topk: int, topp: float, depth: int=-1, random: bool=False) -> tf.Tensor:
    # isolate each one-hot vector
    __logits = logits if (depth < 2) else mlable.shaping.divide(logits, input_axis=-2, output_axis=-1, factor=depth, insert=True)
    # return the index with the highest probability
    return tf.argmax(input=__logits, axis=-1, output_type=tf.int32)

# BINARY #######################################################################

def _combine(logits: tf.Tensor, depth: int=8) -> tf.Tensor:
    __bin_rank = int(tf.rank(logits))
    __cat_dim = 2 ** depth # D
    # actually group the bits together
    __logits = mlable.shaping.divide(logits, input_axis=-2, output_axis=-1, factor=depth, insert=True)
    # reshape to allow broadcasting: add an axis for the categories (..., N, 1, D)
    __logits = tf.expand_dims(__logits, axis=-2)
    # enumerate all possible binary combinations for the given depth
    __categories = tf.range(__cat_dim, dtype=tf.int32)
    # decompose each category in binary bits
    __categories = mlable.ops.expand_binary(__categories, depth=depth, bigendian=False)
    # match the shape of the logits (..., 1, C, D)
    __categories = tf.reshape(__categories, shape=__bin_rank * (1,) + (__cat_dim, depth))
    # select the logits depending on the bit decomposition
    __joint = tf.where(__categories == 1, __logits, -__logits)
    # compute the joint log probabilities for each category (probability that the decomposition match on each bit)
    return tf.reduce_sum(__joint, axis=-1, keepdims=False)

def binary(logits: tf.Tensor, depth: int=8, random: bool=False) -> tf.Tensor:
    # combine the bits by logical unit (typically 8 bit to sample from bytes)
    __logits = _combine(logits=logits, depth=depth)
    # apply categorical sampling on the units
    return categorical(logits=__logits, depth=-1, random=random)

# RAW ##########################################################################

def raw(logits: tf.Tensor, factor: float=256., dtype: tf.dtypes.DType=tf.int32) -> tf.Tensor:
    return tf.cast(tf.round(tf.cast(factor, logits.dtype) * logits), dtype)
