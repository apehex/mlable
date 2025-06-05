import functools
import math

import tensorflow as tf

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

    def build(self, input_shape: tuple) -> None:
        __shape = tuple(input_shape)
        # parse
        self._config['channel_dim'] = self._config['channel_dim'] or int(input_shape[-1])
        self._config['group_dim'] = self._config['group_dim'] or (2 ** int(0.5 * math.log2(int(input_shape[-1]))))
        # factor
        __norm_args = {'groups': self._config['group_dim'], 'epsilon': self._config['epsilon_rate'], 'center': True, 'scale': True,}
        __conv_args = {'filters': self._config['channel_dim'], 'use_bias': True, 'activation': None, 'padding': 'same', 'data_format': 'channels_last'}
        # init
        self._norm1 = tf.keras.layers.GroupNormalization(**__norm_args)
        self._norm2 = tf.keras.layers.GroupNormalization(**__norm_args)
        self._conv0 = tf.keras.layers.Conv2D(kernel_size=1, **__conv_args)
        self._conv1 = tf.keras.layers.Conv2D(kernel_size=3, **__conv_args)
        self._conv2 = tf.keras.layers.Conv2D(kernel_size=3, **__conv_args)
        self._drop = tf.keras.layers.Dropout(self._config['dropout_rate'])
        self._silu = tf.keras.activations.silu
        # build
        self._norm1.build(__shape)
        __shape = self._norm1.compute_output_shape(__shape)
        self._conv1.build(__shape)
        __shape = self._conv1.compute_output_shape(__shape)
        self._norm2.build(__shape)
        __shape = self._norm2.compute_output_shape(__shape)
        self._drop.build(__shape)
        __shape = self._drop.compute_output_shape(__shape)
        self._conv2.build(__shape)
        __shape = self._conv2.compute_output_shape(__shape)
        self._conv0.build(input_shape)
        __shape = self._conv0.compute_output_shape(__shape)
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        # first branch
        __outputs = self._norm1(inputs)
        __outputs = self._silu(__outputs)
        __outputs = self._conv1(__outputs)
        # second branch
        __outputs = self._norm2(__outputs)
        __outputs = self._silu(__outputs)
        __outputs = self._drop(__outputs, training=training)
        __outputs = self._conv2(__outputs)
        # add the residuals
        return __outputs + self._conv0(inputs)

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return tuple(input_shape)[:-1] + (self._config['channel_dim'],)

    def get_config(self) -> dict:
        __config = super(ResnetBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)
