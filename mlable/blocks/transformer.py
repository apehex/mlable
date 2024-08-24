import math

import keras
import tensorflow as tf

import mlable.layers.embedding
import mlable.layers.transformer

# CONSTANTS ###################################################################

EPSILON = 1e-6

# FEED FORWARD ################################################################

@keras.saving.register_keras_serializable(package='blocks')
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

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._ffn(self._norm(inputs))

    def get_config(self) -> dict:
        __config = super(FeedForwardBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# ATTENTION ###################################################################

@keras.saving.register_keras_serializable(package='blocks')
class BaseAttentionBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        sequence_axis: int=1,
        center: bool=False,
        scale: bool=False,
        epsilon: float=EPSILON,
        **kwargs
    ) -> None:
        # init
        super(BaseAttentionBlock, self).__init__(**kwargs)
        # config
        self._config = {
            'num_heads': num_heads,
            'head_dim': head_dim,
            'sequence_axis': sequence_axis,
            'center': center,
            'scale': scale,
            'epsilon': epsilon,}
        # layers
        self._input_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, center=center, scale=scale) # rms_scaling=True
        self._context_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, center=center, scale=scale) # rms_scaling=True
        self._position = mlable.layers.embedding.RotaryPositionalEmbedding(sequence_axis=sequence_axis, feature_axis=-1)
        self._attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_dim, value_dim=head_dim, attention_axes=[sequence_axis], use_bias=False, kernel_initializer='glorot_uniform')

    def get_config(self) -> dict:
        __config = super(BaseAttentionBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

@keras.saving.register_keras_serializable(package='blocks')
class SelfAttentionBlock(BaseAttentionBlock):
    def call(
        self,
        inputs: tf.Tensor,
        attention_mask: tf.Tensor=None,
        use_causal_mask: bool=True,
        training: bool=False,
    ) -> tf.Tensor:
        # normalize
        __y = self._input_norm(inputs)
        # position embedding
        __yp = self._position(inputs=__y, offset=0)
        # attention
        return self._attention(key=__yp, query=__yp, value=__y, training=training, attention_mask=attention_mask, use_causal_mask=use_causal_mask, return_attention_scores=False)

@keras.saving.register_keras_serializable(package='blocks')
class CrossAttentionBlock(BaseAttentionBlock):
    def call(
        self,
        inputs: tf.Tensor,
        contexts: tf.Tensor,
        attention_mask: tf.Tensor=None,
        use_causal_mask: bool=False, # use ALL the context
        training: bool=False,
    ) -> tf.Tensor:
        # normalize
        __x = self._input_norm(inputs)
        __y = self._context_norm(contexts) # may need a dedicated norm layer
        # position embedding
        __xp = self._position(inputs=__x, offset=0)
        __yp = self._position(inputs=__y, offset=0)
        # attention
        return self._attention(key=__yp, query=__xp, value=__y, training=training, attention_mask=attention_mask, use_causal_mask=use_causal_mask, return_attention_scores=False)
