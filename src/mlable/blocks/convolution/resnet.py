import functools
import math

import tensorflow as tf

import mlable.blocks.normalization
import mlable.utils

# CONSTANTS ####################################################################

DROPOUT = 0.0
EPSILON = 1e-6

# RESNET #######################################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class ResnetBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        channel_dim: int=None,
        group_dim: int=None,
        dropout_rate: float=DROPOUT,
        epsilon_rate: float=EPSILON,
        **kwargs
    ) -> None:
        super(ResnetBlock, self).__init__(**kwargs)
        # save the config to allow serialization
        self._config = {
            'channel_dim': channel_dim,
            'group_dim': group_dim,
            'dropout_rate': dropout_rate,
            'epsilon_rate': epsilon_rate,}
        # layers
        self._norm1 = None
        self._norm2 = None
        self._conv0 = None
        self._conv1 = None
        self._conv2 = None
        self._drop = None
        self._silu = None

    def build(self, inputs_shape: tuple, contexts_shape: tuple=None) -> None:
        __shape = tuple(inputs_shape)
        # fill the config with default values
        self._update_config(inputs_shape)
        # init the layers
        self._norm1 = mlable.blocks.normalization.AdaptiveGroupNormalization(**self.get_normalization_config())
        self._norm2 = mlable.blocks.normalization.AdaptiveGroupNormalization(**self.get_normalization_config())
        self._conv0 = tf.keras.layers.Conv2D(**self.get_convolution_config(kernel_size=1))
        self._conv1 = tf.keras.layers.Conv2D(**self.get_convolution_config(kernel_size=3))
        self._conv2 = tf.keras.layers.Conv2D(**self.get_convolution_config(kernel_size=3))
        self._drop = tf.keras.layers.Dropout(self._config['dropout_rate'])
        self._silu = tf.keras.activations.silu
        # build the layers
        self._norm1.build(__shape, contexts_shape=contexts_shape)
        __shape = self._norm1.compute_output_shape(__shape, contexts_shape=contexts_shape)
        self._conv1.build(__shape)
        __shape = self._conv1.compute_output_shape(__shape)
        self._norm2.build(__shape, contexts_shape=contexts_shape)
        __shape = self._norm2.compute_output_shape(__shape, contexts_shape=contexts_shape)
        self._drop.build(__shape)
        __shape = self._drop.compute_output_shape(__shape)
        self._conv2.build(__shape)
        __shape = self._conv2.compute_output_shape(__shape)
        self._conv0.build(inputs_shape)
        __shape = self._conv0.compute_output_shape(__shape)
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, contexts: tf.Tensor=None, training: bool=False, **kwargs) -> tf.Tensor:
        # first branch
        __outputs = self._norm1(inputs, contexts=contexts)
        __outputs = self._silu(__outputs)
        __outputs = self._conv1(__outputs)
        # second branch
        __outputs = self._norm2(__outputs, context=contexts)
        __outputs = self._silu(__outputs)
        __outputs = self._drop(__outputs, training=training)
        __outputs = self._conv2(__outputs)
        # add the residuals
        return __outputs + self._conv0(inputs)

    def compute_output_shape(self, inputs_shape: tuple, contexts_shape: tuple=None) -> tuple:
        return tuple(inputs_shape)[:-1] + (self._config['channel_dim'],)

    def get_config(self) -> dict:
        __config = super(ResnetBlock, self).get_config()
        __config.update(self._config)
        return __config

    def get_convolution_config(self, kernel_size: int=3) -> dict:
        return {
            'filters': self._config['channel_dim'],
            'kernel_size': kernel_size,
            'use_bias': True,
            'activation': None,
            'padding': 'same',
            'data_format': 'channels_last'}

    def get_normalization_config(self) -> dict:
        return {
            'groups': self._config['group_dim'],
            'epsilon': self._config['epsilon_rate'],
            'axis': -1,
            'center': True,
            'scale': True,}

    def _update_config(self, inputs_shape: tuple) -> None:
        __input_dim = int(inputs_shape[-1])
        self._config['channel_dim'] = self._config['channel_dim'] or __input_dim
        self._config['group_dim'] = self._config['group_dim'] or mlable.utils.exproot2(min(__input_dim, self._config['channel_dim']))

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)
