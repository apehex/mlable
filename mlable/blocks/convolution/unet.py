import functools
import math

import tensorflow as tf

import mlable.blocks.convolution.resnet
import mlable.blocks.normalization
import mlable.layers.shaping
import mlable.utils

# CONSTANTS ####################################################################

DROPOUT = 0.0
EPSILON = 1e-6

# 2D SELF ATTENTION ############################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class SelfAttentionBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        group_dim: int=None,
        head_dim: int=None,
        head_num: int=None,
        epsilon_rate: float=EPSILON,
        dropout_rate: float=DROPOUT,
        **kwargs
    ) -> None:
        # init
        super(SelfAttentionBlock, self).__init__(**kwargs)
        # config
        self._config = {
            'group_dim': group_dim,
            'head_dim': head_dim,
            'head_num': head_num,
            'epsilon_rate': epsilon_rate,
            'dropout_rate': dropout_rate,}
        # layers
        self._norm_channel = None
        self._merge_space = None
        self._split_space = None
        self._attend_space = None

    def build(self, inputs_shape: tuple, contexts_shape: tuple=None) -> None:
        __shape = tuple(inputs_shape)
        # fill the config with default values
        self._update_config(__shape)
        # init layers
        self._norm_channel = mlable.blocks.normalization.AdaptiveGroupNormalization(**self.get_normalization_config())
        self._merge_space = mlable.layers.shaping.Merge(**self.get_merge_config())
        self._split_space = mlable.layers.shaping.Divide(**self.get_divide_config(__shape))
        self._attend_space = tf.keras.layers.MultiHeadAttention(**self.get_attention_config())
        # build layers
        self._norm_channel.build(__shape, contexts_shape=contexts_shape)
        __shape = self._norm_channel.compute_output_shape(__shape, contexts_shape=contexts_shape)
        self._merge_space.build(__shape)
        __shape = self._merge_space.compute_output_shape(__shape)
        self._attend_space.build(query_shape=__shape, key_shape=__shape, value_shape=__shape)
        __shape = self._attend_space.compute_output_shape(query_shape=__shape, key_shape=__shape, value_shape=__shape)
        self._split_space.build(__shape)
        __shape = self._split_space.compute_output_shape(__shape)
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, contexts: tf.Tensor=None, training: bool=False, **kwargs) -> tf.Tensor:
        # normalize the channels
        __outputs = self._norm_channel(inputs, contexts=contexts, training=training)
        # merge the space axes
        __outputs = self._merge_space(__outputs)
        # attend to the space sequence
        __outputs = self._attend_space(query=__outputs, key=__outputs, value=__outputs, training=training, use_causal_mask=False, **kwargs)
        # split the space axes back
        return self._split_space(__outputs) + inputs

    def compute_output_shape(self, inputs_shape: tuple, contexts_shape: tuple=None) -> tuple:
        return tuple(inputs_shape)

    def get_config(self) -> dict:
        __config = super(SelfAttentionBlock, self).get_config()
        __config.update(self._config)
        return __config

    def get_merge_config(self) -> dict:
        return {'axis': 1, 'right': True,}

    def get_divide_config(self, inputs_shape: tuple) -> dict:
        return {'axis': 1, 'factor': inputs_shape[2], 'right': True, 'insert': True,}

    def get_attention_config(self) -> dict:
        return {
            'num_heads': self._config['head_num'],
            'key_dim': self._config['head_dim'],
            'value_dim': self._config['head_dim'],
            'dropout': self._config['dropout_rate'],
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'attention_axes': [1],
            'use_bias': True,}

    def get_normalization_config(self) -> dict:
        return {
            'groups': self._config['group_dim'],
            'epsilon': self._config['epsilon_rate'],
            'axis': -1,
            'center': True,
            'scale': True,}

    def _update_config(self, inputs_shape: tuple) -> None:
        # parse the input shape
        __shape = tuple(inputs_shape)
        __dim = int(__shape[-1])
        # fill with default values
        self._config['group_dim'] = self._config['group_dim'] or mlable.utils.exproot2(__dim)
        self._config['head_dim'] = self._config['head_dim'] or __dim
        self._config['head_num'] = self._config['head_num'] or max(1, __dim // self._config['head_dim'])

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# UNET #########################################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class UnetBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        channel_dim: int=None,
        group_dim: int=None,
        head_dim: int=None,
        head_num: int=None,
        layer_num: int=None,
        add_attention: bool=False,
        add_downsampling: bool=False,
        add_upsampling: bool=False,
        dropout_rate: float=DROPOUT,
        epsilon_rate: float=EPSILON,
        **kwargs
    ) -> None:
        super(UnetBlock, self).__init__(**kwargs)
        # save the config to allow serialization
        self._config = {
            'channel_dim': channel_dim,
            'group_dim': group_dim,
            'head_dim': head_dim,
            'head_num': head_num,
            'layer_num': layer_num,
            'add_attention': add_attention,
            'add_downsampling': add_downsampling,
            'add_upsampling': add_upsampling,
            'dropout_rate': max(0.0, dropout_rate),
            'epsilon_rate': max(1e-8, epsilon_rate),}
        # blocks
        self._resnet_blocks = []
        self._sampling_blocks = []

    def build(self, inputs_shape: tuple, contexts_shape: tuple=None) -> None:
        __shape = tuple(inputs_shape)
        # fill the config with default values
        self._update_config(__shape)
        # init the layers
        for _ in range(self._config['layer_num']):
            # always start with a resnet
            self._resnet_blocks.append(mlable.blocks.convolution.resnet.ResnetBlock(**self.get_resnet_config()))
            # interleave resnet and attention blocks
            if self._config['add_attention']:
                self._resnet_blocks.append(SelfAttentionBlock(**self.get_attention_config()))
        # postprocess the attention outputs with an extra resnet block
        if self._config['add_attention']:
            self._resnet_blocks.append(mlable.blocks.convolution.resnet.ResnetBlock(**self.get_resnet_config()))
        # add an optional downsampling block
        if self._config['add_downsampling']:
            self._sampling_blocks.append(tf.keras.layers.Conv2D(**self.get_convolution_config(strides=2)))
        # add an optional upsampling block
        if self._config['add_upsampling']:
            self._sampling_blocks.append(tf.keras.layers.UpSampling2D(**self.get_upsampling_config()))
            self._sampling_blocks.append(tf.keras.layers.Conv2D(**self.get_convolution_config(strides=1)))
        # build
        for __block in self._resnet_blocks:
            __block.build(__shape, contexts_shape=contexts_shape)
            __shape = __block.compute_output_shape(__shape, contexts_shape=contexts_shape)
        for __block in self._sampling_blocks:
            __block.build(__shape)
            __shape = __block.compute_output_shape(__shape)
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, contexts: tf.Tensor=None, training: bool=False, **kwargs) -> tf.Tensor:
        # succession of resnet and attention blocks
        __outputs = functools.reduce(lambda __x, __b: __b(__x, contexts=contexts, training=training, **kwargs), self._resnet_blocks, inputs)
        # either upsampling, downsampling or nothing
        return functools.reduce(lambda __x, __b: __b(__x), self._sampling_blocks, __outputs)

    def compute_output_shape(self, inputs_shape: tuple, contexts_shape: tuple=None) -> tuple:
        # succession of resnet and attention blocks
        __shape = functools.reduce(lambda __s, __b: __b.compute_output_shape(__s, contexts_shape=contexts_shape), self._resnet_blocks, inputs_shape)
        # either upsampling, downsampling or nothing
        return functools.reduce(lambda __s, __b: __b.compute_output_shape(__s), self._sampling_blocks, __shape)

    def get_config(self) -> dict:
        __config = super(UnetBlock, self).get_config()
        __config.update(self._config)
        return __config

    def get_resnet_config(self) -> dict:
        __keys = ['channel_dim', 'group_dim', 'dropout_rate', 'epsilon_rate']
        return {__k: __v for __k, __v in self._config.items() if __k in __keys}

    def get_attention_config(self) -> dict:
        __keys = ['group_dim', 'head_dim', 'head_num', 'dropout_rate', 'epsilon_rate']
        return {__k: __v for __k, __v in self._config.items() if __k in __keys}

    def get_convolution_config(self, strides: int=1) -> dict:
        return {
            'filters': self._config['channel_dim'],
            'strides': strides,
            'kernel_size': 3,
            'use_bias': True,
            'activation': None,
            'padding': 'same',
            'data_format': 'channels_last',}

    def get_upsampling_config(self) -> dict:
        return {
            'size': (2, 2),
            'interpolation': 'bilinear',
            'data_format': 'channels_last',}

    def _update_config(self, inputs_shape: tuple) -> None:
        # parse the input shape
        __shape = tuple(inputs_shape)
        __dim = int(__shape[-1])
        # fill with default values
        self._config['channel_dim'] = self._config['channel_dim'] or __dim
        self._config['group_dim'] = self._config['group_dim'] or mlable.utils.exproot2(min(self._config['channel_dim'], __dim))
        self._config['head_dim'] = self._config['head_dim'] or mlable.utils.exproot2(self._config['channel_dim'])
        self._config['head_num'] = self._config['head_num'] or max(1, self._config['channel_dim'] // self._config['head_dim'])
        self._config['layer_num'] = self._config['layer_num'] or 2

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)
