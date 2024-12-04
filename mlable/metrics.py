import functools
import math

import tensorflow as tf

import mlable.ops
import mlable.sampling
import mlable.utils

# CATEGORICAL ##################################################################

@tf.keras.utils.register_keras_serializable(package='metrics', name='categorical_group_accuracy')
def categorical_group_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor, depth: int=-1, groups: iter=[4], axes: iter=[-1], dtype: tf.dtypes.DType=None) -> tf.Tensor:
    __dtype = dtype or y_true.dtype
    # category indexes
    __yt = mlable.sampling.categorical(prediction=y_true, depth=depth, random=False)
    __yp = mlable.sampling.categorical(prediction=y_pred, depth=depth, random=False)
    # matching
    __match = tf.equal(__yt, __yp)
    # group all the predictions for a given token
    for __g, __a in zip(groups, axes):
        # repeat values so that the reduced tensor has the same shape as the original
        __match = mlable.ops.reduce_all(data=__match, group=__g, axis=__a, keepdims=True)
    # cast
    return tf.cast(__match, dtype=__dtype)

@tf.keras.utils.register_keras_serializable(package='metrics')
class CategoricalGroupAccuracy(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, depth: int=-1, group: int=4, axis: int=-1, name: str='categorical_group_accuracy', dtype: tf.dtypes.DType=None, **kwargs):
        # serialization wrapper
        __wrap = tf.keras.utils.register_keras_serializable(package='metrics', name='categorical_group_accuracy')
        # allow to specify several groups / axes
        __axes = [axis] if isinstance(axis, int) else list(axis)
        __groups = [group] if isinstance(group, int) else list(group)
        # adapt the measure
        __fn = __wrap(functools.partial(categorical_group_accuracy, depth=depth, groups=__groups, axes=__axes, dtype=dtype))
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

# BINARY #######################################################################

@tf.keras.utils.register_keras_serializable(package='metrics', name='binary_group_accuracy')
def binary_group_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor, depth: int=-1, groups: iter=[4], axes: iter=[-1], threshold: float=0.5, dtype: tf.dtypes.DType=None) -> tf.Tensor:
    __dtype = dtype or y_true.dtype
    # category indexes
    __yt = mlable.sampling.binary(prediction=y_true, depth=depth, threshold=threshold, random=False)
    __yp = mlable.sampling.binary(prediction=y_pred, depth=depth, threshold=threshold, random=False)
    # matching
    __match = tf.equal(__yt, __yp)
    # group all the predictions for a given token
    for __g, __a in zip(groups, axes):
        # repeat values so that the reduced tensor has the same shape as the original
        __match = mlable.ops.reduce_all(data=__match, group=__g, axis=__a, keepdims=True)
    # mean over sequence axis
    return tf.cast(__match, dtype=__dtype)

@tf.keras.utils.register_keras_serializable(package='metrics')
class BinaryGroupAccuracy(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, depth: int=-1, group: int=4, axis: int=-1, threshold: float=0.5, name: str='binary_group_accuracy', dtype: tf.dtypes.DType=None, **kwargs):
        # serialization wrapper
        __wrap = tf.keras.utils.register_keras_serializable(package='metrics', name='binary_group_accuracy')
        # allow to specify several groups / axes
        __axes = [axis] if isinstance(axis, int) else list(axis)
        __groups = [group] if isinstance(group, int) else list(group)
        # adapt the measure
        __fn = __wrap(functools.partial(binary_group_accuracy, depth=depth, groups=__groups, axes=__axes, threshold=threshold, dtype=dtype))
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

# BINARY #######################################################################

@tf.keras.utils.register_keras_serializable(package='metrics', name='raw_group_accuracy')
def raw_group_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor, factor: float=256.0, groups: iter=[1], axes: iter=[-1], dtype: tf.dtypes.DType=None) -> tf.Tensor:
    __dtype = dtype or y_true.dtype
    # category indexes
    __yt = mlable.sampling.raw(prediction=y_true, factor=factor, random=False)
    __yp = mlable.sampling.raw(prediction=y_pred, factor=factor, random=False)
    # matching
    __match = tf.equal(__yt, __yp)
    # group all the predictions for a given token
    for __g, __a in zip(groups, axes):
        # repeat values so that the reduced tensor has the same shape as the original
        __match = mlable.ops.reduce_all(data=__match, group=__g, axis=__a, keepdims=True)
    # mean over sequence axis
    return tf.cast(__match, dtype=__dtype)

@tf.keras.utils.register_keras_serializable(package='metrics')
class RawGroupAccuracy(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, factor: float=256.0, group: int=1, axis: int=-1, name: str='raw_group_accuracy', dtype: tf.dtypes.DType=None, **kwargs):
        # serialization wrapper
        __wrap = tf.keras.utils.register_keras_serializable(package='metrics', name='raw_group_accuracy')
        # allow to specify several groups / axes
        __axes = [axis] if isinstance(axis, int) else list(axis)
        __groups = [group] if isinstance(group, int) else list(group)
        # adapt the measure
        __fn = __wrap(functools.partial(raw_group_accuracy, factor=factor, groups=__groups, axes=__axes, dtype=dtype))
        # init
        super(RawGroupAccuracy, self).__init__(fn=__fn, name=name, dtype=dtype, **kwargs)
        # config
        self._config = {'group': group, 'factor': factor}
        # sould be maximized
        self._direction = 'up'

    def get_config(self) -> dict:
        __config = super(RawGroupAccuracy, self).get_config()
        __config.update(self._config)
        return __config

# INCEPTION ####################################################################

@tf.keras.utils.register_keras_serializable(package='metrics')
class KernelInceptionDistance(tf.keras.metrics.Metric):
    def __init__(self, name: str='kernel_inception_distance', **kwargs):
        super(KernelInceptionDistance, self).__init__(name=name, **kwargs)
        # average across batches
        self._metric = tf.keras.metrics.Mean(name="mean_metric")
        # pretrained inception layer
        self._encoder = None

    def _build(self, input_shape: tuple=(64, 64, 3)) -> None:
        self._encoder = tf.keras.Sequential([
                tf.keras.Input(shape=input_shape),
                tf.keras.layers.Rescaling(255.0),
                tf.keras.layers.Resizing(height=75, width=75),
                tf.keras.layers.Lambda(tf.keras.applications.inception_v3.preprocess_input),
                tf.keras.applications.InceptionV3(include_top=False, input_shape=(75, 75, 3), weights="imagenet"),
                tf.keras.layers.GlobalAveragePooling2D(),],
            name="inception_encoder")

    def _kernel(self, left: tf.Tensor, right: tf.Tensor) -> tf.Tensor:
        __norm = 1. / float(list(left.shape)[-1])
        return (1.0 + __norm * left @ tf.transpose(right)) ** 3.0

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor=None) -> tf.Tensor:
        if self._encoder is None:
            self._build(input_shape=tuple(y_true.shape)[1:])
        # batch size
        __n = tuple(y_true.shape)[0]
        # compute inception features
        __f_t = self._encoder(y_true, training=False)
        __f_p = self._encoder(y_pred, training=False)
        # compute polynomial kernels
        __k_tt = self._kernel(__f_t, __f_t)
        __k_pp = self._kernel(__f_p, __f_p)
        __k_tp = self._kernel(__f_t, __f_p)
        # compute mmd
        __k_tt = tf.reduce_sum(__k_tt * (1.0 - tf.eye(__n))) / (__n * (__n - 1))
        __k_pp = tf.reduce_sum(__k_pp * (1.0 - tf.eye(__n))) / (__n * (__n - 1))
        __k_tp = tf.reduce_mean(__k_tp)
        # compute the final KID
        self._metric.update_state(__k_tt + __k_pp - 2.0 * __k_tp)

    def result(self) -> tf.Tensor:
        return self._metric.result()

    def reset_state(self) -> None:
        self._metric.reset_state()
