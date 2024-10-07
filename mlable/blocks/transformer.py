import math

import tensorflow as tf

import mlable.layers.embedding
import mlable.layers.transformer

# CONSTANTS ####################################################################

EPSILON = 1e-6

# FEED FORWARD #################################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class FeedForwardBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        center: bool=False,
        scale: bool=False,
        epsilon: float=EPSILON,
        **kwargs
    ) -> None:
        # init
        super(FeedForwardBlock, self).__init__(**kwargs)
        # config
        self._config = {
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'center': center,
            'scale': scale,
            'epsilon': epsilon,}
        # layers
        self._norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, center=center, scale=scale) # rms_scaling=True
        self._ffn = mlable.layers.transformer.FeedForwardGate(input_dim=embed_dim, hidden_dim=hidden_dim)

    def build(self, input_shape: tf.TensorShape) -> None:
        # the input shape is progated / unchanged
        self._norm.build(input_shape)
        self._ffn.build(input_shape)
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return self._ffn(self._norm(inputs))

    def get_config(self) -> dict:
        __config = super(FeedForwardBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# SELF ATTENTION ###############################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class BaseAttentionBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        head_num: int,
        head_dim: int,
        sequence_axis: int=1,
        bias: bool=True,
        center: bool=False,
        scale: bool=False,
        epsilon: float=EPSILON,
        **kwargs
    ) -> None:
        # init
        super(BaseAttentionBlock, self).__init__(**kwargs)
        # config
        self._config = {
            'head_num': head_num,
            'head_dim': head_dim,
            'sequence_axis': sequence_axis,
            'bias': bias,
            'center': center,
            'scale': scale,
            'epsilon': epsilon,}
        # layers
        self._norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, center=center, scale=scale) # rms_scaling=True
        self._position = mlable.layers.embedding.RotaryPositionalEmbedding(sequence_axis=sequence_axis, feature_axis=-1)
        self._attention = tf.keras.layers.MultiHeadAttention(num_heads=head_num, key_dim=head_dim, value_dim=head_dim, attention_axes=[sequence_axis], use_bias=bias, kernel_initializer='glorot_uniform')

    def build(self, input_shape: tf.TensorShape) -> None:
        # the input shape is progated / unchanged
        self._norm.build(input_shape)
        self._position.build(input_shape)
        # attention API depends on the version
        if hasattr(self._attention, '_build_from_signature'):
            self._attention._build_from_signature(query=input_shape, value=input_shape, key=input_shape)
        else:
            self._attention.build(query_shape=input_shape, value_shape=input_shape, key_shape=input_shape)
        # register
        self.built = True

    def get_config(self) -> dict:
        __config = super(BaseAttentionBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package='blocks')
class SelfAttentionBlock(BaseAttentionBlock):
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # normalize
        __y = self._norm(inputs)
        # position embedding
        __yp = self._position(inputs=__y, offset=0)
        # attention
        return self._attention(key=__yp, query=__yp, value=__y, **kwargs)

# CROSS ATTENTION ##############################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class BaseCrossAttentionBlock(BaseAttentionBlock):
    def __init__(
        self,
        head_num: int,
        head_dim: int,
        sequence_axis: int=1,
        bias: bool=True,
        center: bool=False,
        scale: bool=False,
        epsilon: float=EPSILON,
        **kwargs
    ) -> None:
        # init
        super(BaseCrossAttentionBlock, self).__init__(head_num=head_num, head_dim=head_dim, sequence_axis=sequence_axis, bias=bias, center=center, scale=scale, epsilon=epsilon, **kwargs)
        # layers
        self._key_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, center=center, scale=scale) # rms_scaling=True
        self._value_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, center=center, scale=scale) # rms_scaling=True

    def build(self, query_shape: tf.TensorShape, key_shape: tf.TensorShape, value_shape: tf.TensorShape) -> None:
        # the input shape is progated / unchanged
        self._norm.build(query_shape)
        self._key_norm.build(key_shape)
        self._value_norm.build(value_shape)
        self._position.build(query_shape)
        # attention API depends on the version
        if hasattr(self._attention, '_build_from_signature'):
            self._attention._build_from_signature(query=query_shape, key=key_shape, value=value_shape)
        else:
            self._attention.build(query_shape=query_shape, key_shape=key_shape, value_shape=value_shape)
        # register
        self.built = True

@tf.keras.utils.register_keras_serializable(package='blocks')
class CrossAttentionBlock(BaseCrossAttentionBlock):
    def call(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, **kwargs) -> tf.Tensor:
        # normalize
        __q = self._norm(query)
        __k = self._key_norm(key)
        __v = self._value_norm(value)
        # position embedding
        __qp = self._position(inputs=__q, offset=0)
        __kp = self._position(inputs=__k, offset=0)
        # attention
        return self._attention(query=__qp, key=__kp, value=__v, **kwargs)

# ATTENTION WITH CACHE #########################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class CachedBaseAttentionBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        head_num: int,
        head_dim: int,
        sequence_axis: int=1,
        bias: bool=True,
        center: bool=False,
        scale: bool=False,
        epsilon: float=EPSILON,
        **kwargs
    ) -> None:
        # init
        super(CachedBaseAttentionBlock, self).__init__(**kwargs)
        # config
        self._config = {
            'head_num': head_num,
            'head_dim': head_dim,
            'sequence_axis': sequence_axis,
            'bias': bias,
            'center': center,
            'scale': scale,
            'epsilon': epsilon,}
        # layers
        self._norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, center=center, scale=scale) # rms_scaling=True
        self._position = mlable.layers.embedding.RotaryPositionalEmbedding(sequence_axis=sequence_axis, feature_axis=-1)
        self._attention = mlable.layers.transformer.CachedMultiHeadAttention(num_heads=head_num, key_dim=head_dim, value_dim=head_dim, attention_axes=[sequence_axis], use_bias=bias, kernel_initializer='glorot_uniform')

    def build(self, input_shape: tf.TensorShape) -> None:
        # the input shape is progated / unchanged
        self._norm.build(input_shape)
        self._position.build(input_shape)
        # attention API depends on the version
        if hasattr(self._attention, '_build_from_signature'):
            self._attention._build_from_signature(query=input_shape, key=input_shape, value=input_shape)
        else:
            self._attention.build(query_shape=input_shape, key_shape=input_shape, value_shape=input_shape)
        # register
        self.built = True

    def get_config(self) -> dict:
        __config = super(CachedBaseAttentionBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package='blocks')
class CachedSelfAttentionBlock(CachedBaseAttentionBlock):
    def call(self, inputs: tf.Tensor, cache: tf.Tensor=None, position: int=None, **kwargs) -> tf.Tensor:
        # normalize
        __y = self._norm(inputs)
        # position embedding
        __yp = self._position(inputs=__y, offset=0)
        # attention
        return self._attention(key=__yp, query=__yp, value=__y, cache=cache, step=position, **kwargs)

# SELF DECODER #################################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class SelfDecoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        head_num: int,
        head_dim: int,
        hidden_dim: int,
        sequence_axis: int=1,
        epsilon: float=EPSILON,
        bias: bool=True,
        center: bool=True,
        scale: bool=True,
        **kwargs
    ) -> None:
        # init
        super(SelfDecoderBlock, self).__init__(**kwargs)
        # config
        self._config = {
            'embed_dim': embed_dim,
            'head_num': head_num,
            'head_dim': head_dim,
            'hidden_dim': hidden_dim,
            'sequence_axis': sequence_axis,
            'epsilon': epsilon,
            'bias': bias,
            'center': center,
            'scale': scale,}
        # layers
        self._attention = SelfAttentionBlock(head_num=head_num, head_dim=head_dim, sequence_axis=sequence_axis, epsilon=epsilon, bias=bias, center=center, scale=scale)
        self._ffn = FeedForwardBlock(embed_dim=embed_dim, hidden_dim=hidden_dim, epsilon=epsilon, center=center, scale=scale)

    def build(self, input_shape: tf.TensorShape) -> None:
        # the input shape is propagated / unchanged
        self._attention.build(input_shape)
        self._ffn.build(input_shape)
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return self._ffn(self._attention(inputs=inputs, **kwargs))

    def get_config(self) -> dict:
        __config = super(SelfDecoderBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package='blocks')
class ResidualSelfDecoderBlock(SelfDecoderBlock):
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # residual + self attention
        __x = inputs + self._attention(inputs=inputs, **kwargs)
        # residual + augmentation
        return __x + self._ffn(__x)

# CROSS DECODER ################################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class CrossDecoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        head_num: int,
        head_dim: int,
        hidden_dim: int,
        sequence_axis: int=1,
        epsilon: float=EPSILON,
        bias: bool=True,
        center: bool=True,
        scale: bool=True,
        **kwargs
    ) -> None:
        # init
        super(CrossDecoderBlock, self).__init__(**kwargs)
        # config
        self._config = {
            'embed_dim': embed_dim,
            'head_num': head_num,
            'head_dim': head_dim,
            'hidden_dim': hidden_dim,
            'sequence_axis': sequence_axis,
            'epsilon': epsilon,
            'bias': bias,
            'center': center,
            'scale': scale,}
        # layers
        self._attention = CrossAttentionBlock(head_num=head_num, head_dim=head_dim, sequence_axis=sequence_axis, epsilon=epsilon, bias=bias, center=center, scale=scale)
        self._ffn = FeedForwardBlock(embed_dim=embed_dim, hidden_dim=hidden_dim, epsilon=epsilon, center=center, scale=scale)

    def build(self, query_shape: tf.TensorShape, key_shape: tf.TensorShape, value_shape: tf.TensorShape) -> None:
        # the input shape is propagated / unchanged
        self._attention.build(query_shape=query_shape, key_shape=key_shape, value_shape=value_shape)
        self._ffn.build(query_shape)
        # register
        self.built = True

    def call(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, **kwargs) -> tf.Tensor:
        return self._ffn(self._attention(query=query, key=key, value=value, **kwargs))

    def get_config(self) -> dict:
        __config = super(CrossDecoderBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package='blocks')
class ResidualCrossDecoderBlock(CrossDecoderBlock):
    def call(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, **kwargs) -> tf.Tensor:
        # residual + cross attention
        __x = query + self._attention(query=query, key=key, value=value, **kwargs)
        # residual + augmentation
        return __x + self._ffn(__x)
