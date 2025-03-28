import numpy as np
import tensorflow as tf

import mlable.masking
import mlable.ops
import mlable.shaping

# FILTER ###############################################################

def filter_top_k(logits: tf.Tensor, count: int) -> tf.Tensor:
    __dim = int(tuple(logits.shape)[-1])
    # meaningful candidate count
    __count = tf.clip_by_value(count, clip_value_min=1, clip_value_max=__dim)
    # filter the top k
    __values, __indices = tf.math.top_k(logits, k=__count)
    # select the smallest logits
    __lower = tf.gather(__values, axis=-1, indices=[__count - 1])
    # mask the logits to remove
    __mask = logits < __lower
    # set the filtered logits to -inf
    return mlable.masking.choose(left=logits, right=-np.inf, mask=__mask)

# BINARY ###############################################################

def binary(prediction: tf.Tensor, depth: int=-1, threshold: float=0.5, random: bool=False) -> tf.Tensor:
    # meta
    __threshold = tf.cast(threshold, prediction.dtype)
    # group the bits of each encoded value
    __prediction = prediction if (depth < 2) else mlable.shaping.divide(prediction, input_axis=-2, output_axis=-1, factor=depth, insert=True)
    # binary tensor
    __bits = tf.cast(__prediction > __threshold, dtype=tf.int32)
    # expand to match the input rank
    return mlable.ops._reduce_base(data=__bits, base=2, axis=-1, keepdims=False)

# CATEGORICAL #################################################################

def categorical(prediction: tf.Tensor, depth: int=-1, random: bool=False) -> tf.Tensor:
    # isolate each one-hot vector
    __prediction = prediction if (depth < 2) else mlable.shaping.divide(prediction, input_axis=-2, output_axis=-1, factor=depth, insert=True)
    # return the index with the highest probability
    return tf.argmax(input=__prediction, axis=-1, output_type=tf.int32)

# RAW #########################################################################

def raw(prediction: tf.Tensor, factor: float=256., dtype: tf.dtypes.DType=tf.int32, random: bool=False) -> tf.Tensor:
    return tf.cast(tf.round(tf.cast(factor, prediction.dtype) * prediction), dtype)
