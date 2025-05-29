import functools

import tensorflow as tf

# SCHEDULING ###################################################################

def linear_rate(step: int, start_step: int, end_step: int, start_rate: float=0.0, end_rate: float=1.0) -> float:
    __cast = functools.partial(tf.cast, dtype=tf.float32)
    # signed delta (could go either up or down)
    __delta_rate = __cast(end_rate - start_rate)
    # enforce ascending step order
    __start_step = min(start_step, end_step)
    __end_step = max(start_step, end_step)
    __delta_step = tf.maximum(__cast(1.0), __cast(__end_step - __start_step))
    __delta_step_cur = tf.maximum(__cast(0.0), __cast(step - __start_step))
    return __cast(start_rate) + tf.minimum(__cast(1.0), __delta_step_cur / __delta_step) * __delta_rate

# COSINE #######################################################################

def cosine_angles(angle_rates: float, start_rate: float=1.0, end_rate: float=0.0, dtype: tf.DType=None) -> tf.Tensor:
    __dtype = dtype or getattr(angle_rates, 'dtype', tf.float32)
    __angle_s = tf.cast(tf.math.acos(start_rate), dtype=__dtype)
    __angle_e = tf.cast(tf.math.acos(end_rate), dtype=__dtype)
    # linear progression in the angle space => cosine progression for the signal and noise
    return __angle_s + tf.cast(angle_rates, dtype=__dtype) * (__angle_e - __angle_s)

def cosine_rates(angle_rates: float, start_rate: float=1.0, end_rate: float=0.0, dtype: tf.DType=None) -> tuple:
    __angles = cosine_angles(start_rate=start_rate, end_rate=end_rate, angle_rates=angle_rates, dtype=dtype)
    return tf.math.sin(__angles), tf.math.cos(__angles) # noise rate, signal rate
