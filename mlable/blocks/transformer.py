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
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'use_bias': use_bias,
            'center': center,
            'scale': scale,
            'epsilon': epsilon,
            'activation': activation,}
        # layers
        self._norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, center=center, scale=scale) # rms_scaling=True
        self._ffn = mlable.layers.transformer.FeedForwardNetwork(input_dim=embed_dim, hidden_dim=hidden_dim, use_bias=use_bias, activation=activation)

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
class AttentionBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        head_num: int,
        key_dim: int,
        value_dim: int=None,
        attention_axes: list=[1],
        use_bias: bool=True,
        center: bool=False,
        scale: bool=False,
        epsilon: float=EPSILON,
        **kwargs
    ) -> None:
        # init
        super(AttentionBlock, self).__init__(**kwargs)
        # normalize
        __axes = [attention_axes] if isinstance(attention_axes, int) else list(attention_axes)
        # config
        self._config = {
            'head_num': head_num,
            'key_dim': key_dim,
            'value_dim': value_dim,
            'attention_axes': __axes,
            'use_bias': use_bias,
            'center': center,
            'scale': scale,
            'epsilon': epsilon,}
        # normalization layers
        self._query_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, center=center, scale=scale) # rms_scaling=True
        self._key_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, center=center, scale=scale) # rms_scaling=True
        self._value_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, center=center, scale=scale) # rms_scaling=True
        # position layers
        self._position = {__a: mlable.layers.embedding.RotaryPositionalEmbedding(sequence_axis=__a, feature_axis=-1) for __a in __axes}
        # attention layer
        self._attention = tf.keras.layers.MultiHeadAttention(num_heads=head_num, key_dim=key_dim, value_dim=value_dim, attention_axes=__axes, use_bias=use_bias, kernel_initializer='glorot_uniform')
        # specific building mechanism != built-in
        self._built = False

    def _build(self, query_shape: tf.TensorShape, key_shape: tf.TensorShape, value_shape: tf.TensorShape) -> None:
        if not self._built:
            # the input shape is progated / unchanged
            self._query_norm.build(query_shape)
            self._key_norm.build(key_shape)
            self._value_norm.build(value_shape)
            for __p in self._position.values(): __p.build(query_shape)
            # attention API depends on the version
            if hasattr(self._attention, '_build_from_signature'):
                self._attention._build_from_signature(query=query_shape, key=key_shape, value=value_shape)
            else:
                self._attention.build(query_shape=query_shape, key_shape=key_shape, value_shape=value_shape)
            # register
            self.built, self._built = True, True

    def build(self, query_shape: tf.TensorShape, key_shape: tf.TensorShape=None, value_shape: tf.TensorShape=None) -> None:
        if (key_shape is not None) and (value_shape is not None):
            self._build(query_shape=query_shape, key_shape=key_shape, value_shape=value_shape)

    def call(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, **kwargs) -> tf.Tensor:
        # build
        self._build(query_shape=tuple(query.shape), key_shape=tuple(key.shape), value_shape=tuple(value.shape))
        # normalize
        __q = self._query_norm(query)
        __k = self._key_norm(key)
        __v = self._value_norm(value)
        # position embedding, along each axis
        __qp, __kp = __q, __k
        for __position in self._position.values():
            __qp = __position(inputs=__qp, offset=0)
            __kp = __position(inputs=__kp, offset=0)
        # attention
        return self._attention(query=__qp, key=__kp, value=__v, **kwargs)

    def get_config(self) -> dict:
        __config = super(AttentionBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# ATTENTION WITH CACHE #########################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class CachedAttentionBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        head_num: int,
        key_dim: int,
        value_dim: int=None,
        attention_axes: list=[1],
        use_bias: bool=True,
        center: bool=False,
        scale: bool=False,
        epsilon: float=EPSILON,
        **kwargs
    ) -> None:
        # init
        super(CachedAttentionBlock, self).__init__(**kwargs)
        # normalize
        __axes = [attention_axes] if isinstance(attention_axes, int) else list(attention_axes)
        # config
        self._config = {
            'head_num': head_num,
            'key_dim': key_dim,
            'value_dim': value_dim,
            'attention_axes': __axes,
            'use_bias': use_bias,
            'center': center,
            'scale': scale,
            'epsilon': epsilon,}
        # normalization layers
        self._query_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, center=center, scale=scale) # rms_scaling=True
        self._key_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, center=center, scale=scale) # rms_scaling=True
        self._value_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, center=center, scale=scale) # rms_scaling=True
        # position layers
        self._position = {__a: mlable.layers.embedding.RotaryPositionalEmbedding(sequence_axis=__a, feature_axis=-1) for __a in __axes}
        # attention layer
        self._attention = mlable.layers.transformer.CachedMultiHeadAttention(num_heads=head_num, key_dim=key_dim, value_dim=value_dim, attention_axes=__axes, use_bias=use_bias, kernel_initializer='glorot_uniform')
        # specific building mechanism != built-in
        self._built = False

    def _build(self, query_shape: tf.TensorShape, key_shape: tf.TensorShape, value_shape: tf.TensorShape) -> None:
        if not self._built:
            # the input shape is progated / unchanged
            self._query_norm.build(query_shape)
            self._key_norm.build(key_shape)
            self._value_norm.build(value_shape)
            for __p in self._position.values(): __p.build(query_shape)
            # attention API depends on the version
            if hasattr(self._attention, '_build_from_signature'):
                self._attention._build_from_signature(query=query_shape, key=key_shape, value=value_shape)
            else:
                self._attention.build(query_shape=query_shape, key_shape=key_shape, value_shape=value_shape)
            # register
            self.built, self._built = True, True

    def build(self, query_shape: tf.TensorShape, key_shape: tf.TensorShape=None, value_shape: tf.TensorShape=None) -> None:
        if (key_shape is not None) and (value_shape is not None):
            self._build(query_shape=query_shape, key_shape=key_shape, value_shape=value_shape)

    def call(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, cache: tf.Tensor=None, position: int=None, **kwargs) -> tf.Tensor:
        # build
        self._build(query_shape=tuple(query.shape), key_shape=tuple(key.shape), value_shape=tuple(value.shape))
        # normalize
        __q = self._query_norm(query)
        __k = self._key_norm(key)
        __v = self._value_norm(value)
        # position embedding, along each axis
        __qp, __kp = __q, __k
        for __position in self._position.values():
            __qp = __position(inputs=__qp, offset=0)
            __kp = __position(inputs=__kp, offset=0)
        # attention
        return self._attention(query=__qp, key=__kp, value=__v, cache=cache, step=position, **kwargs)

    def get_config(self) -> dict:
        __config = super(CachedAttentionBlock, self).get_config()
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
        embed_dim: int,
        key_dim: int,
        hidden_dim: int,
        value_dim: int=None,
        attention_axes: list=[1],
        epsilon: float=EPSILON,
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
            'embed_dim': embed_dim,
            'key_dim': key_dim,
            'value_dim': value_dim,
            'hidden_dim': hidden_dim,
            'attention_axes': attention_axes,
            'epsilon': epsilon,
            'use_bias': use_bias,
            'center': center,
            'scale': scale,}
        # layers
        self._attention = AttentionBlock(head_num=head_num, key_dim=key_dim, value_dim=value_dim, attention_axes=attention_axes, epsilon=epsilon, use_bias=use_bias, center=center, scale=scale)
        self._ffn = mlable.blocks.transformer.FeedForwardBlock(embed_dim=embed_dim, hidden_dim=hidden_dim, epsilon=epsilon, center=center, scale=scale)
        # specific building mechanism != built-in
        self._built = False

    def _build(self, query_shape: tf.TensorShape, key_shape: tf.TensorShape, value_shape: tf.TensorShape) -> None:
        if not self._built:
            # the input shape is propagated / unchanged
            self._attention._build(query_shape=query_shape, key_shape=key_shape, value_shape=value_shape)
            self._ffn.build(query_shape)
            # register
            self.built, self._built = True, True

    def build(self, query_shape: tf.TensorShape, key_shape: tf.TensorShape=None, value_shape: tf.TensorShape=None) -> None:
        if (key_shape is not None) and (value_shape is not None):
            self._build(query_shape=query_shape, key_shape=key_shape, value_shape=value_shape)

    def call(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, **kwargs) -> tf.Tensor:
        # build
        self._build(query_shape=tuple(query.shape), key_shape=tuple(key.shape), value_shape=tuple(value.shape))
        # forward
        return self._ffn(self._attention(query=query, key=key, value=value, **kwargs))

    def get_config(self) -> dict:
        __config = super(DecoderBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package='blocks')
class ResidualDecoderBlock(DecoderBlock):
    def call(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, **kwargs) -> tf.Tensor:
        # build
        self._build(query_shape=tuple(query.shape), key_shape=tuple(key.shape), value_shape=tuple(value.shape))
        # residual + cross attention
        __x = query + self._attention(query=query, key=key, value=value, **kwargs)
        # residual + augmentation
        return __x + self._ffn(__x)
