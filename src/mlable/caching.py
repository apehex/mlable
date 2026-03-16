import tensorflow as tf

import mlable.shapes

# CREATE #######################################################################

def create(batch_dim: int, cache_dim: int, head_dim: int, num_heads: int=None) -> tf.Tensor:
    __shape = [2, batch_dim, cache_dim, num_heads, head_dim] if num_heads else [2, batch_dim, cache_dim, head_dim]
    return tf.zeros(__shape, dtype=tf.float32)

# UPDATE #######################################################################

def update(tensor: tf.Tensor, cache: tf.Tensor, axis: int=1, step: int=None) -> tf.Tensor:
    if step is not None:
    	# expand the sequence axis with 1-dim axes
        __shape = mlable.shapes.filter(shape=list(cache.shape), axes=[axis])
        # index of the updated row
        __indices = tf.reshape(tf.one_hot(indices=step, depth=__shape[axis], dtype=tensor.dtype), shape=__shape)
        # updated cache
        __tensor = cache + tensor * __indices
    else:
        __tensor = tf.concat(values=[tf.cast(cache, tensor.dtype), tensor], axis=axis)
    # past + current values
    return __tensor
