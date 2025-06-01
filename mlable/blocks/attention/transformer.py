import tensorflow as tf

import mlable.blocks.attention.generic
import mlable.layers.transformer

# CONSTANTS ####################################################################

EPSILON = 1e-6

# FEED FORWARD #################################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class FeedForwardBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_dim: int,
        dropout_rate: float=0.0,
        use_bias: bool=True,
        center: bool=False,
        scale: bool=False,
        epsilon: float=EPSILON,
        activation: str='gelu',
        **kwargs
    ) -> None:
        # init
        super(FeedForwardBlock, self).__init__(**kwargs)
        # config
        self._config = {
            'hidden_dim': hidden_dim,
            'dropout_rate': dropout_rate,
            'use_bias': use_bias,
            'center': center,
            'scale': scale,
            'epsilon': epsilon,
            'activation': activation,}
        # layers
        self._norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, center=center, scale=scale) # rms_scaling=True
        self._ffn = mlable.layers.transformer.FeedForwardNetwork(hidden_dim=hidden_dim, use_bias=use_bias, activation=activation, dropout_rate=dropout_rate)

    def build(self, input_shape: tuple) -> None:
        # the input shape is progated / unchanged
        self._norm.build(input_shape)
        self._ffn.build(input_shape)
        # register
        self.built = True

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return tuple(input_shape)

    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        # normalize
        __outputs = self._norm(inputs, training=training)
        # augment
        return self._ffn(__outputs, training=training)

    def get_config(self) -> dict:
        __config = super(FeedForwardBlock, self).get_config()
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
        head_num: int,
        key_dim: int,
        hidden_dim: int,
        value_dim: int=None,
        attention_axes: list=[1],
        epsilon: float=EPSILON,
        dropout_rate: float=0.0,
        use_bias: bool=True,
        center: bool=True,
        scale: bool=True,
        **kwargs
    ) -> None:
        # init
        super(DecoderBlock, self).__init__(**kwargs)
        # config
        self._config = {
            'head_num': head_num,
            'key_dim': key_dim,
            'value_dim': value_dim,
            'hidden_dim': hidden_dim,
            'attention_axes': attention_axes,
            'epsilon': epsilon,
            'dropout_rate': dropout_rate,
            'use_bias': use_bias,
            'center': center,
            'scale': scale,}
        # layers
        self._attention = mlable.blocks.attention.generic.AttentionBlock(head_num=head_num, key_dim=key_dim, value_dim=value_dim, attention_axes=attention_axes, dropout_rate=dropout_rate, epsilon=epsilon, use_bias=use_bias, center=center, scale=scale)
        self._ffn = FeedForwardBlock(hidden_dim=hidden_dim, dropout_rate=dropout_rate, epsilon=epsilon, center=center, scale=scale)

    def _build(self, query_shape: tuple, key_shape: tuple, value_shape: tuple) -> None:
        if not self.built:
            # the input shape is propagated / unchanged
            self._attention._build(query_shape=query_shape, key_shape=key_shape, value_shape=value_shape)
            self._ffn.build(query_shape)
            # register
            self.built = True

    def build(self, query_shape: tuple, key_shape: tuple=None, value_shape: tuple=None) -> None:
        if (key_shape is not None) and (value_shape is not None):
            self._build(query_shape=query_shape, key_shape=key_shape, value_shape=value_shape)

    def compute_output_shape(self, query_shape: tuple, key_shape: tuple=None, value_shape: tuple=None) -> tuple:
        return tuple(query_shape)

    def call(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        # build
        self._build(query_shape=tuple(query.shape), key_shape=tuple(key.shape), value_shape=tuple(value.shape))
        # position aware attention
        __outputs = self._attention(query=query, key=key, value=value, training=training, **kwargs)
        # augment
        return self._ffn(__outputs, training=training)

    def get_config(self) -> dict:
        __config = super(DecoderBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package='blocks')
class ResidualDecoderBlock(DecoderBlock):
    def call(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        # build
        self._build(query_shape=tuple(query.shape), key_shape=tuple(key.shape), value_shape=tuple(value.shape))
        # residual + cross attention
        __x = query + self._attention(query=query, key=key, value=value, training=training, **kwargs)
        # residual + augmentation
        return __x + self._ffn(__x, training=training)
