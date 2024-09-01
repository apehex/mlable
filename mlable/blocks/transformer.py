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

    def build(self, input_shape: tf.TensorShape) -> None:
        # the input shape is progated / unchanged
        self._norm.build(input_shape)
        self._ffn.build(input_shape)
        # register
        self.built = True

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._ffn(self._norm(inputs))

    def get_config(self) -> dict:
        __config = super(FeedForwardBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# SELF ATTENTION ##############################################################

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
        self._position = mlable.layers.embedding.RotaryPositionalEmbedding(sequence_axis=sequence_axis, feature_axis=-1)
        self._attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_dim, value_dim=head_dim, attention_axes=[sequence_axis], use_bias=False, kernel_initializer='glorot_uniform')

    def build(self, input_shape: tf.TensorShape) -> None:
        # the input shape is progated / unchanged
        self._input_norm.build(input_shape)
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

# CROSS ATTENTION #############################################################

@keras.saving.register_keras_serializable(package='blocks')
class BaseCrossAttentionBlock(BaseAttentionBlock):
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
        super(BaseCrossAttentionBlock, self).__init__(num_heads=num_heads, head_dim=head_dim, sequence_axis=sequence_axis, center=center, scale=scale, epsilon=epsilon, **kwargs)
        # layers
        self._context_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=epsilon, center=center, scale=scale) # rms_scaling=True

    def build(self, inputs_shape: tf.TensorShape, contexts_shape: tf.TensorShape) -> None:
        # the input shape is progated / unchanged
        self._input_norm.build(inputs_shape)
        self._context_norm.build(contexts_shape)
        self._position.build(inputs_shape)
        # attention API depends on the version
        if hasattr(self._attention, '_build_from_signature'):
            self._attention._build_from_signature(query=inputs_shape, value=contexts_shape, key=contexts_shape)
        else:
            self._attention.build(query_shape=inputs_shape, value_shape=contexts_shape, key_shape=contexts_shape)
        # register
        self.built = True

@keras.saving.register_keras_serializable(package='blocks')
class CrossAttentionBlock(BaseCrossAttentionBlock):
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

# ATTENTION WITH CACHE ########################################################

@keras.saving.register_keras_serializable(package='blocks')
class CachedBaseAttentionBlock(tf.keras.layers.Layer):
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
        super(CachedBaseAttentionBlock, self).__init__(**kwargs)
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
        self._position = mlable.layers.embedding.RotaryPositionalEmbedding(sequence_axis=sequence_axis, feature_axis=-1)
        self._attention = mlable.layers.transformer.CachedMultiHeadAttention(num_heads=num_heads, key_dim=head_dim, value_dim=head_dim, attention_axes=[sequence_axis], use_bias=False, kernel_initializer='glorot_uniform')

    def build(self, input_shape: tf.TensorShape) -> None:
        # the input shape is progated / unchanged
        self._input_norm.build(input_shape)
        self._position.build(input_shape)
        # attention API depends on the version
        if hasattr(self._attention, '_build_from_signature'):
            self._attention._build_from_signature(query=input_shape, value=input_shape, key=input_shape)
        else:
            self._attention.build(query_shape=input_shape, value_shape=input_shape, key_shape=input_shape)
        # register
        self.built = True

    def get_config(self) -> dict:
        __config = super(CachedBaseAttentionBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

@keras.saving.register_keras_serializable(package='blocks')
class CachedSelfAttentionBlock(CachedBaseAttentionBlock):
    def call(
        self,
        inputs: tf.Tensor,
        cache: tf.Tensor=None,
        position: int=None,
        attention_mask: tf.Tensor=None,
        use_causal_mask: bool=True,
        training: bool=False,
    ) -> tf.Tensor:
        # normalize
        __y = self._input_norm(inputs)
        # position embedding
        __yp = self._position(inputs=__y, offset=0)
        # attention
        return self._attention(key=__yp, query=__yp, value=__y, cache=cache, step=position, training=training, attention_mask=attention_mask, use_causal_mask=use_causal_mask, return_attention_scores=False)
