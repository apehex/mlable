import math

import keras
import tensorflow as tf

import mlable.utils

# FEED FORWARD ################################################################

@keras.saving.register_keras_serializable(package='layers')
class FeedForwardGate(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        **kwargs
    ) -> None:
        super(FeedForwardGate, self).__init__(**kwargs)
        # config
        self._config = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,}
        # layers
        self._gelu = tf.keras.layers.Dense(units=self._config['hidden_dim'], activation='gelu', use_bias=False, kernel_initializer='glorot_uniform', name='gate')
        self._linear = tf.keras.layers.Dense(units=self._config['hidden_dim'], activation='linear', use_bias=False, kernel_initializer='glorot_uniform', name='linear')
        self._output = tf.keras.layers.Dense(units=self._config['input_dim'], activation='linear', use_bias=False, kernel_initializer='glorot_uniform', name='output')

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # gating mechanism
        return self._output(self._gelu(inputs) * self._linear(inputs))

    def get_config(self) -> dict:
        __config = super(FeedForwardGate, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# ATTENTION ###################################################################

@tf.keras.utils.register_keras_serializable(package="Text")
class CachedMultiHeadAttention(tf.keras.layers.MultiHeadAttention):
    """
    Arguments are the same as `tf.keras.layers.MultiHeadAttention` layer.
    
    Scalar dimensions referenced here:
        B = batch_dim (number of sequences)
        F = seq_dim `from_tensor`
        T = seq_dim `to_tensor`
        N = num_heads
        H = head_dim
    """
    def call(
        self,
        query: tf.Tensor,
        value: tf.Tensor,
        key: tf.Tensor=None,
        cache: tf.Tensor=None,
        step: int=None,
        training: bool=False,
        attention_mask: tf.Tensor=None,
        return_attention_scores: bool=False,
        use_causal_mask: bool=True,
    ) -> tf.Tensor:
        if (hasattr(self, "_build_from_signature") and hasattr(self, "_built_from_signature") and not self._built_from_signature):
            self._build_from_signature(query=query, value=value, key=key)
        # attention mask
        __mask = self._compute_attention_mask(query=query, value=value, attention_mask=attention_mask, use_causal_mask=use_causal_mask) # TODO here or after the cache update??
        # init
        __cache = None
        __key = value if key is None else key
        # [B, F, N ,H]
        __query = self._query_dense(query)
        # [B, T, N, H]
        __key = self._key_dense(__key)
        # [B, T, N, H]
        __value = self._value_dense(value)
        # update the key + value caches
        if not training and cache is not None:
            __key = mlable.utils.update_cache(tensor=__key, cache=cache[0], step=step, axis=self._attention_axes[0]) # custom seq axis?
            __value = mlable.utils.update_cache(tensor=__value, cache=cache[1], step=step, axis=self._attention_axes[0]) # custom seq axis?
            __cache = tf.stack(values=(__key, __value), axis=0)
        # use the parent functionalities
        __output, __scores = self._compute_attention(query=__query, key=__key, value=__value, attention_mask=__mask, training=training)
        # projection
        __output = self._output_dense(__output)
        # output
        if return_attention_scores:
            return __output, __scores, __cache
        return __output, __cache
