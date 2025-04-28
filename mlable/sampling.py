import numpy as np
import tensorflow as tf

import mlable.masking
import mlable.maths.ops
import mlable.shaping.axes

# SHAPING ######################################################################

def _group(logits: tf.Tensor, depth: int=-1) -> tf.Tensor:
    return (
        logits if (depth < 2)
        else mlable.shaping.axes.divide(logits, axis=-1, factor=depth, insert=True, right=True))

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
    return tf.where(__mask, x=tf.cast(-np.inf, dtype=logits.dtype), y=logits)

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
    return tf.where(__mask, x=tf.cast(-np.inf, dtype=logits.dtype), y=logits)

# CATEGORICAL ##################################################################

def _categorical(logits: tf.Tensor, num_samples: int=1, seed: int=None, name: str=None, dtype: tf.DType=None) -> tf.Tensor:
    # save the original shape
    __shape = tuple(logits.shape)
    # flatten all the axes except the categories
    __logits = tf.reshape(logits, shape=(-1, int(__shape[-1])))
    # take random samples (requires a 2D tensor, hence the wrapper)
    __samples = tf.random.categorical(__logits, num_samples=num_samples, seed=seed, name=name, dtype=dtype)
    # return to the original shape
    return tf.reshape(__samples, shape=__shape[:-1] + (num_samples,))

def categorical(logits: tf.Tensor, temp: float=1.0, topp: float=0.0, topk: int=0, depth: int=-1, seed: int=None, dtype: tf.DType=tf.int32) -> tf.Tensor:
    # isolate each one-hot vector
    __logits = _group(logits=logits, depth=depth)
    # greedy sampling by default (deterministic)
    __samples = tf.argmax(input=__logits, axis=-1, output_type=dtype)
    # tweak the distribution
    if temp > 0.0:
        __logits = tf.cast(1. / temp, dtype=__logits.dtype) * __logits
    # nucleus sampling
    if topp > 0.0:
        __logits = filter_top_p(logits=__logits, threshold=topp)
        __samples = _categorical(logits=__logits, num_samples=1, seed=seed, dtype=dtype)
        __samples = tf.squeeze(__samples, axis=-1)
    # limit to the top probabilities
    if topk > 0: # top-p and top-k can be combined
        __logits = filter_top_k(logits=__logits, count=topk)
        __samples = _categorical(logits=__logits, num_samples=1, seed=seed, dtype=dtype)
        __samples = tf.squeeze(__samples, axis=-1)
    # tensor of integer indexes
    return __samples

# BINARY #######################################################################

def _combine(logits: tf.Tensor, bigendian: bool=True) -> tf.Tensor:
    # parse the shape
    __bin_shape = tuple(logits.shape)
    __bin_rank = len(__bin_shape)
    __bin_dim = int(__bin_shape[-1])
    __cat_dim = 2 ** __bin_dim # B
    # reshape to allow broadcasting: add an axis for the categories (..., N, 1, B)
    __logits = tf.expand_dims(logits, axis=-2)
    # enumerate all possible binary combinations for the given depth
    __categories = tf.range(__cat_dim, dtype=tf.int32)
    # decompose each category in binary bits
    __categories = mlable.maths.ops.expand_binary(__categories, depth=__bin_dim, bigendian=bigendian)
    # match the shape of the logits (..., 1, C, B)
    __categories = tf.reshape(__categories, shape=(__bin_rank - 1) * (1,) + (__cat_dim, __bin_dim))
    # select the logits depending on the bit decomposition
    __joint = tf.where(__categories == 1, __logits, -__logits)
    # compute the joint log probabilities for each category (probability that the decomposition match on each bit)
    return tf.reduce_sum(__joint, axis=-1, keepdims=False)

def _binary_bit_by_bit(logits: tf.Tensor, threshold: float=0.0, bigendian: bool=True, dtype: tf.DType=tf.int32) -> tf.Tensor:
    # convert the probabilities to bits
    __bits = tf.cast(logits > tf.cast(threshold, dtype=logits.dtype), dtype=dtype)
    # combine the bits into numbers
    return mlable.maths.ops.reduce_base(__bits, base=2, axis=-1, group=-1, keepdims=False, bigendian=bigendian)

def _binary_group_by_group(logits: tf.Tensor, temp: float=1.0, topp: float=-1.0, topk: int=-1, seed: int=None, bigendian: bool=True, dtype: tf.DType=tf.int32) -> tf.Tensor:
    # combine the bits by logical unit (typically 8 bit to sample from bytes)
    __logits = _combine(logits=logits, bigendian=bigendian)
    # no need to split the tensor further, it already has a depth of 2 ** depth
    return categorical(logits=__logits, temp=temp, topp=topp, topk=topk, depth=-1, seed=seed, dtype=dtype)

def binary(logits: tf.Tensor, threshold: float=0.0, temp: float=1.0, topp: float=-1.0, topk: int=-1, depth: int=-1, seed: int=None, bigendian: bool=True, dtype: tf.DType=tf.int32) -> tf.Tensor:
    # group the bits together if necessary
    __logits = _group(logits=logits, depth=depth)
    # greedy sampling bit by bit by default
    __samples = _binary_bit_by_bit(logits=__logits, threshold=threshold, bigendian=bigendian, dtype=dtype)
    # group the bit predictions by categories
    if (topp > 0.0) or (topk > 0):
        __samples = _binary_group_by_group(__logits, temp=temp, topp=topp, topk=topk, seed=seed, bigendian=bigendian, dtype=dtype)
    # index predictions
    return __samples

# RAW ##########################################################################

def raw(data: tf.Tensor, factor: float=256., dtype: tf.DType=tf.int32) -> tf.Tensor:
    return tf.cast(tf.round(tf.cast(factor, data.dtype) * data), dtype)
