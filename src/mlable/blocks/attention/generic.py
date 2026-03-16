import tensorflow as tf

import mlable.layers.embedding
import mlable.layers.transformer

# CONSTANTS ####################################################################

EPSILON = 1e-6

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
        dropout_rate: float=0.0,
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
            'epsilon': epsilon,
            'dropout_rate': dropout_rate,}
        # normalization layers
        self._query_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, center=center, scale=scale) # rms_scaling=True
        self._key_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, center=center, scale=scale) # rms_scaling=True
        self._value_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, center=center, scale=scale) # rms_scaling=True
        # attention layer
        self._attention = tf.keras.layers.MultiHeadAttention(num_heads=head_num, key_dim=key_dim, value_dim=value_dim, attention_axes=__axes, use_bias=use_bias, dropout=dropout_rate, kernel_initializer='glorot_uniform')

    def _build(self, query_shape: tuple, key_shape: tuple, value_shape: tuple) -> None:
        if not self.built:
            # the input shape is progated / unchanged
            self._query_norm.build(query_shape)
            self._key_norm.build(key_shape)
            self._value_norm.build(value_shape)
            # attention API depends on the version
            if hasattr(self._attention, '_build_from_signature'):
                self._attention._build_from_signature(query=query_shape, key=key_shape, value=value_shape)
            else:
                self._attention.build(query_shape=query_shape, key_shape=key_shape, value_shape=value_shape)
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
        # normalize
        __q = self._query_norm(query, training=training)
        __k = self._key_norm(key, training=training)
        __v = self._value_norm(value, training=training)
        # attention
        return self._attention(query=__q, key=__k, value=__v, training=training, **kwargs)

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
        dropout_rate: float=0.0,
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
            'epsilon': epsilon,
            'dropout_rate': dropout_rate,}
        # normalization layers
        self._query_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, center=center, scale=scale) # rms_scaling=True
        self._key_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, center=center, scale=scale) # rms_scaling=True
        self._value_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, center=center, scale=scale) # rms_scaling=True
        # attention layer
        self._attention = mlable.layers.transformer.CachedMultiHeadAttention(num_heads=head_num, key_dim=key_dim, value_dim=value_dim, attention_axes=__axes, use_bias=use_bias, dropout=dropout_rate, kernel_initializer='glorot_uniform')

    def _build(self, query_shape: tuple, key_shape: tuple, value_shape: tuple) -> None:
        if not self.built:
            # the input shape is progated / unchanged
            self._query_norm.build(query_shape)
            self._key_norm.build(key_shape)
            self._value_norm.build(value_shape)
            # attention API depends on the version
            if hasattr(self._attention, '_build_from_signature'):
                self._attention._build_from_signature(query=query_shape, key=key_shape, value=value_shape)
            else:
                self._attention.build(query_shape=query_shape, key_shape=key_shape, value_shape=value_shape)
            # register
            self.built = True

    def build(self, query_shape: tuple, key_shape: tuple=None, value_shape: tuple=None) -> None:
        if (key_shape is not None) and (value_shape is not None):
            self._build(query_shape=query_shape, key_shape=key_shape, value_shape=value_shape)

    def compute_output_shape(self, query_shape: tuple, key_shape: tuple=None, value_shape: tuple=None) -> tuple:
        return tuple(query_shape)

    def call(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, cache: tf.Tensor=None, position: int=None, training: bool=False, **kwargs) -> tf.Tensor:
        # build
        self._build(query_shape=tuple(query.shape), key_shape=tuple(key.shape), value_shape=tuple(value.shape))
        # normalize
        __q = self._query_norm(query, training=training)
        __k = self._key_norm(key, training=training)
        __v = self._value_norm(value, training=training)
        # attention
        return self._attention(query=__q, key=__k, value=__v, cache=cache, step=position, training=training, **kwargs)

    def get_config(self) -> dict:
        __config = super(CachedAttentionBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)
