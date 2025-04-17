import tensorflow as tf

import mlable.shapes

# DIVIDE ######################################################################

def divide(data: tf.Tensor, input_axis: int, output_axis: int, factor: int, insert: bool=False) -> tf.Tensor:
    # move data from input axis to output axis
    __shape = mlable.shapes.divide(shape=list(data.shape), input_axis=input_axis, output_axis=output_axis, factor=factor, insert=insert)
    # actually reshape
    return tf.reshape(tensor=data, shape=__shape)

# MERGE #######################################################################

def merge(data: tf.Tensor, left_axis: int, right_axis: int, left: bool=True) -> tf.Tensor:
    # new shape
    __shape = mlable.shapes.merge(shape=list(data.shape), left_axis=left_axis, right_axis=right_axis, left=left)
    # actually merge the two axes
    return tf.reshape(tensor=data, shape=__shape)
