import functools

import tensorflow as tf

import mlable.utils

# ACCURACY ####################################################################

def group_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor, group: int=4) -> tuple:
    # category indexes
    __yt = tf.argmax(y_true, axis=-1)
    __yp = tf.argmax(y_pred, axis=-1)
    # matching
    __match = tf.equal(__yt, __yp)
    # group all the predictions for a given token
    if group and group > 1:
        __shape = mlable.utils.normalize_shape(list(__match.shape))
        # split
        __shape[-1] = mlable.utils.divide_dim(dim_l=__shape[-1], dim_r=group)
        __shape.append(group)
        # reshape
        __match = tf.reshape(__match, shape=__shape)
        # the token prediction is right if ALL its byte predictions are right
        __match = tf.reduce_all(__match, axis=-1)
    # cast
    return tf.cast(__match, dtype=tf.dtypes.float32)

class CategoricalGroupAccuracy(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, group: int=4, name: str='categorical_group_accuracy', dtype: tf.dtypes.DType=tf.dtypes.float32, **kwargs):
        # adapt the measure
        __fn = functools.partial(group_accuracy, group=group)
        # init
        super(CategoricalGroupAccuracy, self).__init__(fn=__fn, name=name, dtype=dtype, **kwargs)
        # group predictions
        self._group = group
        # sould be maximized
        self._direction = 'up'

    def get_config(self) -> dict:
        __config = super(CategoricalGroupAccuracy, self).get_config()
        __config.update({'group': self._group})
        return __config

# LOSS ########################################################################
