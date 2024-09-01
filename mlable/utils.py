import functools

import tensorflow as tf

import mlable.shaping

# FUNCTIONS ###################################################################

compose = lambda __l: (lambda __x: functools.reduce(lambda __e, __f: __f(__e), __l, __x))

distribute = lambda __f: (lambda *__t: tuple(map(__f, __t)))

# CACHE #######################################################################

def create_cache(batch_dim: int, cache_dim: int, head_dim: int, num_heads: int=None) -> tf.Tensor:
    __shape = [2, batch_dim, cache_dim, num_heads, head_dim] if num_heads else [2, batch_dim, cache_dim, head_dim]
    return tf.zeros(__shape, dtype=tf.float32)

def update_cache(tensor: tf.Tensor, cache: tf.Tensor, axis: int=1, step: int=None) -> tf.Tensor:
    if step is not None:
    	# expand the sequence axis with 1-dim axes
        __shape = mlable.shaping.filter_shape(shape=list(cache.shape), axes=[axis])
        # index of the updated row
        __indices = tf.reshape(tf.one_hot(indices=step, depth=__shape[axis], dtype=tensor.dtype), shape=__shape)
        # updated cache
        __tensor = cache + tensor * __indices
    else:
        __tensor = tf.concat(values=[tf.cast(cache, tensor.dtype), tensor], axis=axis)
    # past + current values
    return __tensor