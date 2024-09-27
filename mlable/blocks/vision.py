import functools
import math

import keras
import tensorflow as tf

# CONSTANTS ####################################################################

DROPOUT = 0.1
EPSILON = 1e-6
PADDING = 'same'

# CONVOLUTION ##################################################################

@keras.saving.register_keras_serializable(package='blocks')
class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, channel_dim: int, kernel_dim: int, stride_dim: int, dropout_rate: float=DROPOUT, epsilon: float=EPSILON, padding: str=PADDING, **kwargs) -> None:
        super(ConvolutionBlock, self).__init__(**kwargs)
        # save config
        self._config = {'channel_dim': channel_dim, 'kernel_dim': kernel_dim, 'stride_dim': stride_dim, 'dropout_rate': dropout_rate, 'epsilon': epsilon, 'padding': padding,}
        # layers
        self._layers = [
            tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(rate=dropout_rate),
            tf.keras.layers.Conv2D(filters=channel_dim, kernel_size=kernel_dim, padding=padding, strides=stride_dim, activation=None),]

    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        return functools.reduce(lambda __t, __l: __l(__t, training=training, **kwargs), self._layers, inputs)

    def get_config(self) -> dict:
        __config = super(ConvolutionBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# TRANSPOSE ####################################################################

@keras.saving.register_keras_serializable(package='blocks')
class TransposeConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, channel_dim: int, kernel_dim: int, stride_dim: int, dropout_rate: float=DROPOUT, epsilon: float=EPSILON, padding: str=PADDING, **kwargs) -> None:
        super(TransposeConvolutionBlock, self).__init__(**kwargs)
        # save config
        self._config = {'channel_dim': channel_dim, 'kernel_dim': kernel_dim, 'stride_dim': stride_dim, 'dropout_rate': dropout_rate, 'epsilon': epsilon, 'padding': padding,}
        # layers
        self._layers = [
            tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(rate=dropout_rate),
            tf.keras.layers.Conv2DTranspose(filters=channel_dim, kernel_size=kernel_dim, padding=padding, strides=stride_dim, activation=None),]

    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        return functools.reduce(lambda __t, __l: __l(__t, training=training, **kwargs), self._layers, inputs)

    def get_config(self) -> dict:
        __config = super(TransposeConvolutionBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)