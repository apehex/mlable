import functools

import tensorflow as tf

# SCHEDULING ###################################################################

def linear_schedule(step: int, step_min: int, step_max: int, rate_min: float=0.0, rate_max: float=1.0) -> float:
    __cast = functools.partial(tf.cast, dtype=tf.float32)
    __delta_rate = __cast(rate_max) - __cast(rate_min)
    __delta_step_cur = tf.maximum(__cast(0.0), __cast(step) - __cast(step_min))
    __delta_step_max = tf.maximum(__cast(1.0), __cast(step_max) - __cast(step_min))
    return rate_min + tf.minimum(__cast(1.0), __delta_step_cur / __delta_step_max) * __delta_rate
