import functools
import math

import tensorflow as tf

import mlable.layers.shaping
import mlable.blocks.transformer

# CONSTANTS ####################################################################

DROPOUT = 0.0
EPSILON = 1e-6
PADDING = 'same'

# CONVOLUTION ##################################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, channel_dim: int, kernel_dim: int, stride_dim: int, dropout_rate: float=DROPOUT, epsilon: float=EPSILON, padding: str=PADDING, **kwargs) -> None:
        super(ConvolutionBlock, self).__init__(**kwargs)
        # save config
        self._config = {'channel_dim': channel_dim, 'kernel_dim': kernel_dim, 'stride_dim': stride_dim, 'dropout_rate': dropout_rate, 'epsilon': epsilon, 'padding': padding,}
        # layers
        self._layers = [
            tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, center=True, scale=True),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(rate=dropout_rate),
            tf.keras.layers.Conv2D(filters=channel_dim, kernel_size=kernel_dim, padding=padding, strides=stride_dim, activation=None, use_bias=True, data_format='channels_last'),]

    def build(self, input_shape: tuple) -> None:
        # the input shape is progated / unchanged
        for __l in self._layers:
            __l.build(input_shape)
        # register
        self.built = True

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return functools.reduce(lambda __s, __l: __l.compute_output_shape(__s), self._layers, tuple(input_shape))

    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        return functools.reduce(lambda __t, __l: __l(__t, training=training, **kwargs), self._layers, inputs)

    def get_config(self) -> dict:
        __config = super(ConvolutionBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# RESIDUAL #####################################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class ResidualBlock(ConvolutionBlock):
    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        return inputs + functools.reduce(lambda __t, __l: __l(__t, training=training, **kwargs), self._layers, inputs)

# TRANSPOSE ####################################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class TransposeConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, channel_dim: int, kernel_dim: int, stride_dim: int, dropout_rate: float=DROPOUT, epsilon: float=EPSILON, padding: str=PADDING, **kwargs) -> None:
        super(TransposeConvolutionBlock, self).__init__(**kwargs)
        # save config
        self._config = {'channel_dim': channel_dim, 'kernel_dim': kernel_dim, 'stride_dim': stride_dim, 'dropout_rate': dropout_rate, 'epsilon': epsilon, 'padding': padding,}
        # layers
        self._layers = [
            tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, center=True, scale=True),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(rate=dropout_rate),
            tf.keras.layers.Conv2DTranspose(filters=channel_dim, kernel_size=kernel_dim, padding=padding, strides=stride_dim, activation=None, use_bias=True, data_format='channels_last'),]

    def build(self, input_shape: tuple) -> None:
        # the input shape is progated / unchanged
        for __l in self._layers:
            __l.build(input_shape)
        # register
        self.built = True

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return functools.reduce(lambda __s, __l: __l.compute_output_shape(__s), self._layers, tuple(input_shape))

    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        return functools.reduce(lambda __t, __l: __l(__t, training=training, **kwargs), self._layers, inputs)

    def get_config(self) -> dict:
        __config = super(TransposeConvolutionBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# RESNET #######################################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class ResnetBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        channel_dim: int=None,
        group_dim: int=None,
        dropout_rate: float=0.0,
        epsilon_rate: float=1e-6,
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
        dropout_rate: float=0.0,
        epsilon_rate: float=1e-6,
        downsample_on: bool=True,
        **kwargs
    ) -> None:
        super(EncoderBlock, self).__init__(**kwargs)
        # save config
        self._config = {
            'channel_dim': channel_dim,
            'group_dim': group_dim,
            'layer_num': layer_num,
            'dropout_rate': dropout_rate,
            'epsilon_rate': epsilon_rate,
            'downsample_on': downsample_on,
        }
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
        dropout_rate: float=0.0,
        epsilon_rate: float=1e-6,
        **kwargs
    ) -> None:
        super(TransformerBlock, self).__init__(**kwargs)
        # save the config to allow serialization
        self._config = {
            'channel_dim': channel_dim,
            'head_dim': head_dim,
            'group_dim': group_dim,
            'layer_num': layer_num,
            'dropout_rate': dropout_rate,
            'epsilon_rate': epsilon_rate,
        }
        # layers
        self._merge_space = None
        self._split_space = None
        self._resnet_blocks = []
        self._attention_blocks = []

    def build(self, input_shape):
        __shape = tuple(input_shape)
        # even the shapes to add the residuals with the intermediate outputs
        self._resnet_blocks.append(ResnetBlock(
            channel_dim=self._config['channel_dim'],
            group_dim=self._config['group_dim'],
            dropout_rate=self._config['dropout_rate'],
            epsilon_rate=self._config['epsilon_rate']))
        # interleave attention and resnet blocks
        for _ in range(self._config['layer_num']):
            self._attention_blocks.append(mlable.blocks.transformer.AttentionBlock(
                head_num=max(1, self._config['channel_dim'] // self._config['head_dim']),
                key_dim=self._config['head_dim'],
                value_dim=self._config['head_dim'],
                attention_axes=[1],
                use_position=False,
                use_bias=True,
                center=False,
                scale=False,
                epsilon=epsilon_rate,
                dropout_rate=dropout_rate))
            self._resnet_blocks.append(ResnetBlock(
                channel_dim=self._config['channel_dim'],
                group_dim=self._config['group_dim'],
                dropout_rate=self._config['dropout_rate'],
                epsilon_rate=self._config['epsilon_rate']))

        self.built = True

    def call(self, inputs, training=False, **kwargs):
        hidden_states = self._resnet_blocks[0](inputs, training=training)

        for attn, resnet in zip(self._attention_blocks, self._resnet_blocks[1:]):
            # Attention expects [batch, height*width, channels]
            batch_size, h, w, c = tf.shape(hidden_states)[0], hidden_states.shape[1], hidden_states.shape[2], hidden_states.shape[3]
            x_flat = tf.reshape(hidden_states, (batch_size, h * w, c))

            attn_out = attn(x_flat, x_flat, training=training)
            attn_out = tf.reshape(attn_out, (batch_size, h, w, c))

            hidden_states = hidden_states + attn_out
            hidden_states = resnet(hidden_states, training=training)

        return hidden_states

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
        dropout_rate: float=0.0,
        epsilon_rate: float=1e-6,
        upsample_on: bool=True,
        **kwargs
    ) -> None:
        super(DecoderBlock, self).__init__(**kwargs)
        # save config
        self._config = {
            'channel_dim': channel_dim,
            'group_dim': group_dim,
            'layer_num': layer_num,
            'dropout_rate': dropout_rate,
            'epsilon_rate': epsilon_rate,
            'upsample_on': upsample_on,
        }
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
