import math

import tensorflow as tf

# PROBABILITIES ################################################################

def log_normal_pdf(sample: tf.Tensor, mean: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
    __log2pi = tf.cast(tf.math.log(2. * math.pi), dtype=sample.dtype)
    __delta2 = (sample - tf.cast(mean, dtype=sample.dtype)) ** 2.
    __logvar = tf.cast(logvar, dtype=sample.dtype)
    return -0.5 * (__delta2 * tf.exp(-__logvar) + __logvar + __log2pi)
