import functools

import tensorflow as tf

# FUNCTIONS ###################################################################

compose = lambda __l: (lambda __x: functools.reduce(lambda __e, __f: __f(__e), __l, __x))

# SHAPES ######################################################################

def normalize_dim(dim: int) -> int:
    return -1 if (dim is None or dim < 0) else dim

def multiply_dim(dim_l: int, dim_r: int) -> int:
    return -1 if (dim_l == -1 or dim_r == -1) else dim_l * dim_r

def divide_dim(dim_l: int, dim_r: int) -> int:
    return -1 if (dim_l == -1 or dim_r == -1) else dim_l // dim_r

def filter_shape(shape: list, axes: list) -> list:
    return [__d if __i in axes else 1 for __i, __d in enumerate(shape)]

def normalize_shape(shape: list) -> list:
    return [-1 if __d is None else __d for __d in shape]

# CACHE #######################################################################

def create_cache(batch_dim: int, cache_dim: int, head_dim: int, num_heads: int=None) -> tf.Tensor:
    __shape = [2, batch_dim, cache_dim, num_heads, head_dim] if num_heads else [2, batch_dim, cache_dim, head_dim]
    return tf.zeros(__shape, dtype=tf.float32)

def update_cache(tensor: tf.Tensor, cache: tf.Tensor, axis: int=1, step: int=None) -> tf.Tensor:
    if step is not None:
    	# expand the sequence axis with 1-dim axes
        __shape = filter_shape(shape=list(cache.shape), axes=[axis])
        # index of the updated row
        __indices = tf.reshape(tf.one_hot(indices=step, depth=__shape[axis], dtype=tensor.dtype), shape=__shape)
        # updated cache
        __tensor = cache + tensor * __indices
    else:
        __tensor = tf.concat(values=[tf.cast(cache, tensor.dtype), tensor], axis=axis)
    # past + current values
    return __tensor