import tensorflow as tf

# CONTRAST #####################################################################

def contrast(left: tf.Tensor, right: tf.Tensor, weight: float=1.0, dtype: tf.dtypes.DType=tf.float32) -> tf.Tensor:
    # identify the positions that differ between left and right tensors
    __mask = tf.not_equal(left, right)
    # cast from bool to allow multiplications
    __mask = tf.cast(__mask, dtype=dtype)
    # highlight the differences
    return weight * __mask + (1.0 - weight) * (1.0 - __mask)

# REPLACE ######################################################################

def choose(left: tf.Tensor, right: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    # expand the right values to match the shape of the left
    __right = tf.zeros_like(left) + tf.cast(right, dtype=left.dtype)
    # choose the right values when mask is true, left otherwise
    return tf.where(mask, x=__right, y=left)
