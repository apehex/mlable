import tensorflow as tf

import mlable.ops

# BINARY ###############################################################

def binary(prediction: tf.Tensor, threshold: float=0.5, random: bool=False) -> tf.Tensor:
    # meta
    __threshold = tf.cast(threshold, prediction.dtype)
    # binary tensor
    __bits = tf.cast(prediction > __threshold, dtype=tf.int32)
    # expand to match the input rank
    return mlable.ops._reduce_base(data=__bits, base=2, axis=-1, keepdims=False)

# CATEGORICAL #################################################################

def categorical(prediction: tf.Tensor, random: bool=False) -> tf.Tensor:
    return tf.argmax(input=prediction, axis=-1, output_type=tf.int32)

# RAW #########################################################################

def raw(prediction: tf.Tensor, factor: float=256., random: bool=False) -> tf.Tensor:
    return tf.cast(tf.round(tf.cast(factor, prediction.dtype) * prediction), tf.int32)
