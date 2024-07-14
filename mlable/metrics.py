import functools

import keras as ks
import tensorflow as tf

import mlable.masking

# CATEGORICAL #################################################################

@ks.saving.register_keras_serializable(package='metrics', name='categorical_group_accuracy')
def categorical_group_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor, group: int=4, dtype: tf.dtypes.DType=None) -> tf.Tensor:
    __dtype = dtype or y_true.dtype
    # category indexes
    __yt = tf.argmax(y_true, axis=-1)
    __yp = tf.argmax(y_pred, axis=-1)
    # matching
    __match = tf.equal(__yt, __yp)
    # group all the predictions for a given token
    if group and group > 1:
        # repeat values so that the reduced tensor has the same shape as the original
        __match = mlable.masking.reduce_all(mask=__match, group=group, axis=-1, keepdims=True)
    # cast
    return tf.cast(__match, dtype=__dtype)

@ks.saving.register_keras_serializable(package='metrics')
class CategoricalGroupAccuracy(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, group: int=4, name: str='categorical_group_accuracy', dtype: tf.dtypes.DType=None, **kwargs):
        # serialization wrapper
        __wrap = ks.saving.register_keras_serializable(package='metrics', name='categorical_group_accuracy')
        # adapt the measure
        __fn = __wrap(functools.partial(categorical_group_accuracy, group=group, dtype=dtype))
        # init
        super(CategoricalGroupAccuracy, self).__init__(fn=__fn, name=name, dtype=dtype, **kwargs)
        # config
        self._config = {'group': group}
        # sould be maximized
        self._direction = 'up'

    def get_config(self) -> dict:
        __config = super(CategoricalGroupAccuracy, self).get_config()
        __config.update(self._config)
        return __config

# LOSS ########################################################################
