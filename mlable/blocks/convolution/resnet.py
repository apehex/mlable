import functools
import math

import tensorflow as tf

import mlable.layers.shaping
import mlable.blocks.attention.generic

# CONSTANTS ####################################################################

DROPOUT = 0.0
EPSILON = 1e-6
PADDING = 'same'

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

# ENCODER ######################################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class EncoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        channel_dim: int,
        group_dim: int=32,
        layer_num: int=1,
        downsample_on: bool=True,
        dropout_rate: float=DROPOUT,
        epsilon_rate: float=EPSILON,
        **kwargs
    ) -> None:
        super(EncoderBlock, self).__init__(**kwargs)
        # save config
        self._config = {
            'channel_dim': channel_dim,
            'group_dim': group_dim,
            'layer_num': layer_num,
            'downsample_on': downsample_on,
            'dropout_rate': dropout_rate,
            'epsilon_rate': epsilon_rate,}
        # layers
        self._blocks = []

    def build(self, input_shape: tuple) -> None:
        __shape = tuple(input_shape)
        # init
        self._blocks = [
            ResnetBlock(
                channel_dim=self._config['channel_dim'],
                group_dim=self._config['group_dim'],
                dropout_rate=self._config['dropout_rate'],
                epsilon_rate=self._config['epsilon_rate'],)
            for _ in range(self._config['layer_num'])]
        if self._config['downsample_on']:
            self._blocks.append(tf.keras.layers.Conv2D(
                filters=self._config['channel_dim'],
                kernel_size=3,
                strides=2,
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
        return functools.reduce(lambda __x, __b: __b(__x, training=training), self._blocks, inputs)

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return functools.reduce(lambda __s, __b: __b.compute_output_shape(__s), self._blocks, input_shape)

    def get_config(self) -> dict:
        __config = super(EncoderBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# TRANSFORMER ##################################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        channel_dim: int,
        head_dim: int=1,
        group_dim: int=32,
        layer_num: int=1,
        dropout_rate: float=DROPOUT,
        epsilon_rate: float=EPSILON,
        use_causal_mask: bool=False,
        **kwargs
    ) -> None:
        super(TransformerBlock, self).__init__(**kwargs)
        # save the config to allow serialization
        self._config = {
            'channel_dim': max(1, channel_dim),
            'head_dim': max(1, head_dim),
            'group_dim': max(1, group_dim),
            'layer_num': max(1, layer_num),
            'dropout_rate': max(0.0, dropout_rate),
            'epsilon_rate': max(1e-8, epsilon_rate),
            'use_causal_mask': use_causal_mask,}
        # layers
        self._merge_space = None
        self._split_space = None
        self._resnet_blocks = []
        self._attention_blocks = []

    def build(self, input_shape):
        __shape = tuple(input_shape)
        # merge the space axes for the attention blocks
        self._merge_space = mlable.layers.shaping.Merge(axis=1, right=True)
        self._split_space = mlable.layers.shaping.Divide(axis=1, factor=__shape[2], right=True, insert=True)
        # prepend one additional resnet block before the first attention to even the shapes and smooth
        self._resnet_blocks = [
            ResnetBlock(
                channel_dim=self._config['channel_dim'],
                group_dim=self._config['group_dim'],
                dropout_rate=self._config['dropout_rate'],
                epsilon_rate=self._config['epsilon_rate'])
            for _ in range(self._config['layer_num'] + 1)]
        # interleave attention and resnet blocks
        self._attention_blocks = [
            mlable.blocks.attention.generic.AttentionBlock(
                head_num=max(1, self._config['channel_dim'] // self._config['head_dim']),
                key_dim=self._config['head_dim'],
                value_dim=self._config['head_dim'],
                attention_axes=[1],
                use_bias=True,
                center=False,
                scale=False,
                epsilon=self._config['epsilon_rate'],
                dropout_rate=self._config['dropout_rate'])
            for _ in range(self._config['layer_num'])]
        # build
        self._resnet_blocks[0].build(__shape)
        __shape = self._resnet_blocks[0].compute_output_shape(__shape)
        for __b_att, __b_res in zip(self._attention_blocks, self._resnet_blocks[1:]):
            self._merge_space.build(__shape)
            __shape = self._merge_space.compute_output_shape(__shape)
            __b_att.build(query_shape=__shape, key_shape=__shape, value_shape=__shape)
            __shape = __b_att.compute_output_shape(query_shape=__shape, key_shape=__shape, value_shape=__shape)
            self._split_space.build(__shape)
            __shape = self._split_space.compute_output_shape(__shape)
            __b_res.build(__shape)
            __shape = __b_res.compute_output_shape(__shape)
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        # smooth (B, H, W, E) => (B, H, W, C)
        __outputs = self._resnet_blocks[0](inputs, training=training)
        for __b_att, __b_res in zip(self._attention_blocks, self._resnet_blocks[1:]):
            # merge the space axes (B, H, W, C) => (B, HW, C)
            __outputs = self._merge_space(__outputs)
            # apply attention (B, HW, C) => (B, HW, C)
            __outputs = __b_att(query=__outputs, key=__outputs, value=__outputs, training=training, use_causal_mask=self._config['use_causal_mask'], **kwargs)
            # split the space axes back (B, HW, C) => (B, H, W, C)
            __outputs = self._split_space(__outputs)
            # improve the features (B, H, W, C) => (B, H, W, C)
            __outputs = __b_res(__outputs, training=training)
        # (B, H, W, C)
        return __outputs

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return input_shape[:-1] + (self._config['channel_dim'],)

    def get_config(self) -> dict:
        __config = super(TransformerBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# DECODER ######################################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class DecoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        channel_dim: int,
        group_dim: int=32,
        layer_num: int=1,
        upsample_on: bool=True,
        dropout_rate: float=DROPOUT,
        epsilon_rate: float=EPSILON,
        **kwargs
    ) -> None:
        super(DecoderBlock, self).__init__(**kwargs)
        # save config
        self._config = {
            'channel_dim': channel_dim,
            'group_dim': group_dim,
            'layer_num': layer_num,
            'upsample_on': upsample_on,
            'dropout_rate': dropout_rate,
            'epsilon_rate': epsilon_rate,}
        # layers
        self._blocks = []

    def build(self, input_shape: tuple) -> None:
        __shape = tuple(input_shape)
        # init
        self._blocks = [
            ResnetBlock(
                channel_dim=self._config['channel_dim'],
                group_dim=self._config['group_dim'],
                dropout_rate=self._config['dropout_rate'],
                epsilon_rate=self._config['epsilon_rate'],)
            for _ in range(self._config['layer_num'])]
        if self._config['upsample_on']:
            self._blocks.extend([
                tf.keras.layers.UpSampling2D(
                    size=(2, 2),
                    interpolation='nearest',
                    data_format='channels_last'),
                tf.keras.layers.Conv2D(
                    filters=self._config['channel_dim'],
                    kernel_size=3,
                    strides=1,
                    use_bias=True,
                    activation=None,
                    padding='same',
                    data_format='channels_last'),])
        # build
        for __block in self._blocks:
            __block.build(__shape)
            __shape = __block.compute_output_shape(__shape)
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        return functools.reduce(lambda __x, __b: __b(__x, training=training), self._blocks, inputs)

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return functools.reduce(lambda __s, __b: __b.compute_output_shape(__s), self._blocks, input_shape)

    def get_config(self) -> dict:
        __config = super(DecoderBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)
