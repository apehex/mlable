import functools
import math

import tensorflow as tf

import mlable.blocks.convolution.resnet
import mlable.layers.shaping

# CONSTANTS ####################################################################

DROPOUT = 0.0
EPSILON = 1e-6

# 2D SELF ATTENTION ############################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class AttentionBlock(tf.keras.layers.Layer):
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
        super(AttentionBlock, self).__init__(**kwargs)
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

    def build(self, input_shape: tuple) -> None:
        __shape = tuple(input_shape)
        # fill the config with default values
        self._update_config(__shape)
        # factor args
        __norm_args = {'groups': self._config['group_dim'], 'epsilon': self._config['epsilon_rate'], 'axis': -1, 'center': True, 'scale': True,}
        # init layers
        self._norm_channel = tf.keras.layers.GroupNormalization(**__norm_args)
        self._merge_space = mlable.layers.shaping.Merge(axis=1, right=True)
        self._split_space = mlable.layers.shaping.Divide(axis=1, factor=__shape[2], right=True, insert=True)
        self._attend_space = tf.keras.layers.MultiHeadAttention(
            num_heads=self._config['head_num'],
            key_dim=self._config['head_dim'],
            value_dim=self._config['head_dim'],
            attention_axes=[1],
            use_bias=True,
            dropout=self._config['dropout_rate'],
            kernel_initializer='glorot_uniform')
        # build layers
        self._norm_channel.build(__shape)
        __shape = self._norm_channel.compute_output_shape(__shape)
        self._merge_space.build(__shape)
        __shape = self._merge_space.compute_output_shape(__shape)
        self._attend_space.build(query_shape=__shape, key_shape=__shape, value_shape=__shape)
        __shape = self._attend_space.compute_output_shape(query_shape=__shape, key_shape=__shape, value_shape=__shape)
        self._split_space.build(__shape)
        __shape = self._split_space.compute_output_shape(__shape)
        # register
        self.built = True

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return tuple(input_shape)

    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        # normalize the channels
        __outputs = self._norm_channel(inputs, training=training)
        # merge the space axes
        __outputs = self._merge_space(__outputs)
        # attend to the space sequence
        __outputs = self._attend_space(query=__outputs, key=__outputs, value=__outputs, training=training, use_causal_mask=False, **kwargs)
        # split the space axes back
        return self._split_space(__outputs)

    def get_config(self) -> dict:
        __config = super(AttentionBlock, self).get_config()
        __config.update(self._config)
        return __config

    def _update_config(self, input_shape: tuple) -> None:
        # parse the input shape
        __shape = tuple(input_shape)
        __dim = int(__shape[-1])
        # fill with default values
        self._config['group_dim'] = self._config['group_dim'] or (2 ** int(0.5 * math.log2(__dim)))
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
        # layers
        self._blocks = []

    def build(self, input_shape):
        __shape = tuple(input_shape)
        # fill the config with default values
        self._update_config(__shape)
        # init the layers
        for _ in range(self._config['layer_num']):
            # always start with a resnet
            self._blocks.append(mlable.blocks.convolution.resnet.ResnetBlock(
                channel_dim=self._config['channel_dim'],
                group_dim=self._config['group_dim'],
                dropout_rate=self._config['dropout_rate'],
                epsilon_rate=self._config['epsilon_rate']))
            # interleave resnet and attention blocks
            if self._config['add_attention']:
                self._blocks.append(AttentionBlock(
                    group_dim=self._config['group_dim'],
                    head_dim=self._config['head_dim'],
                    head_num=self._config['head_num'],
                    epsilon_rate=self._config['epsilon_rate'],
                    dropout_rate=self._config['dropout_rate']))
        # add an optional downsampling block
        if self._config['add_downsampling']:
            self._blocks.append(tf.keras.layers.Conv2D(
                filters=self._config['channel_dim'],
                kernel_size=3,
                strides=2,
                use_bias=True,
                activation=None,
                padding='same',
                data_format='channels_last'))
        # add an optional upsampling block
        if self._config['add_upsampling']:
            self._blocks.append(tf.keras.layers.UpSampling2D(
                size=(2, 2),
                interpolation='nearest',
                data_format='channels_last'))
            self._blocks.append(tf.keras.layers.Conv2D(
                filters=self._config['channel_dim'],
                kernel_size=3,
                strides=1,
                use_bias=True,
                activation=None,
                padding='same',
                data_format='channels_last'))
        # build
        for __block in self._blocks:
            __block.build(__shape)
            __shape = __block.compute_output_shape(__shape)
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        return functools.reduce(lambda __x, __b: __b(__x, training=training, **kwargs), self._blocks, inputs)

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return functools.reduce(lambda __s, __b: __b.compute_output_shape(__s), self._blocks, input_shape)

    def get_config(self) -> dict:
        __config = super(UnetBlock, self).get_config()
        __config.update(self._config)
        return __config

    def _update_config(self, input_shape: tuple) -> None:
        # parse the input shape
        __shape = tuple(input_shape)
        __dim = int(__shape[-1])
        # fill with default values
        self._config['channel_dim'] = self._config['channel_dim'] or __dim
        self._config['group_dim'] = self._config['group_dim'] or max(1, (2 ** int(0.5 * math.log2(__dim))))
        self._config['head_dim'] = self._config['head_dim'] or self._config['channel_dim']
        self._config['head_num'] = self._config['head_num'] or max(1, self._config['channel_dim'] // self._config['head_dim'])
        self._config['layer_num'] = self._config['layer_num'] or 2

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)
