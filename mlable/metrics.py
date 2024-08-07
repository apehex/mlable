import functools

import keras as ks
import tensorflow as tf

import mlable.ops
import mlable.sampling
import mlable.utils

# CATEGORICAL #################################################################

@ks.saving.register_keras_serializable(package='metrics', name='categorical_group_accuracy')
def categorical_group_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor, group: int=4, dtype: tf.dtypes.DType=None) -> tf.Tensor:
    __dtype = dtype or y_true.dtype
    # category indexes
    __yt = mlable.sampling.categorical(prediction=y_true, random=False)
    __yp = mlable.sampling.categorical(prediction=y_pred, random=False)
    # matching
    __match = tf.equal(__yt, __yp)
    # group all the predictions for a given token
    if group and group > 1:
        # repeat values so that the reduced tensor has the same shape as the original
        __match = mlable.ops.reduce_all(data=__match, group=group, axis=-1, keepdims=True)
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

# BINARY ######################################################################

@ks.saving.register_keras_serializable(package='metrics', name='binary_group_accuracy')
def binary_group_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor, group: int=4, threshold: float=0.5, dtype: tf.dtypes.DType=None) -> tf.Tensor:
    __dtype = dtype or y_true.dtype
    # category indexes
    __yt = mlable.sampling.binary(prediction=y_true, threshold=threshold, random=False)
    __yp = mlable.sampling.binary(prediction=y_pred, threshold=threshold, random=False)
    # matching
    __match = tf.equal(__yt, __yp)
    # group all the predictions for a given token
    if group and group > 1:
        # repeat values so that the reduced tensor has the same shape as the original
        __match = mlable.ops.reduce_all(data=__match, group=group, axis=-1, keepdims=True)
    # mean over sequence axis
    return tf.cast(__match, dtype=__dtype)

@ks.saving.register_keras_serializable(package='metrics')
class BinaryGroupAccuracy(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, group: int=4, threshold: float=0.5, name: str='binary_group_accuracy', dtype: tf.dtypes.DType=None, **kwargs):
        # serialization wrapper
        __wrap = ks.saving.register_keras_serializable(package='metrics', name='binary_group_accuracy')
        # adapt the measure
        __fn = __wrap(functools.partial(binary_group_accuracy, group=group, threshold=threshold, dtype=dtype))
        # init
        super(BinaryGroupAccuracy, self).__init__(fn=__fn, name=name, dtype=dtype, **kwargs)
        # config
        self._config = {'group': group, 'threshold': threshold}
        # sould be maximized
        self._direction = 'up'

    def get_config(self) -> dict:
        __config = super(BinaryGroupAccuracy, self).get_config()
        __config.update(self._config)
        return __config
