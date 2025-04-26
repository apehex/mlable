import tensorflow as tf

import mlable.shapes

# DIVIDE ######################################################################

def divide(data: tf.Tensor, axis: int, factor: int, insert: bool=False, right: bool=True) -> tf.Tensor:
    # move data from the source axis to its neighbor
    __shape = mlable.shapes.divide(shape=list(data.shape), axis=axis, factor=factor, insert=insert, right=right)
    # actually reshape
    return tf.reshape(tensor=data, shape=__shape)

# MERGE #######################################################################

def merge(data: tf.Tensor, axis: int, right: bool=True) -> tf.Tensor:
    # new shape
    __shape = mlable.shapes.merge(shape=list(data.shape), axis=axis, right=right)
    # actually merge the two axes
    return tf.reshape(tensor=data, shape=__shape)

# SWAP #########################################################################

def swap(data: tf.Tensor, left_axis: int, right_axis: int) -> tf.Tensor:
    # mapping from the new axis indices to the old indices
    __perm = mlable.shapes.swap(shape=range(len(data.shape)), left=left_axis, right=right_axis)
    # transpose the data instead of just reshaping
    return tf.transpose(data, perm=__perm, conjugate=False)

# MOVE #########################################################################

def move(data: tf.Tensor, from_axis: int, to_axis: int) -> tf.Tensor:
    # mapping from the new axis indices to the old indices
    __perm = mlable.shapes.move(shape=range(len(data.shape)), before=from_axis, after=to_axis)
    # transpose the data instead of just reshaping
    return tf.transpose(data, perm=__perm, conjugate=False)
