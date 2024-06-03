import math

import keras
import tensorflow as tf

import mlable.utils

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
        value_cache: tf.Tensor=None,
        key_cache: tf.Tensor=None,
        attention_mask: tf.Tensor=None,
        decode_loop_step: int=None,
        return_attention_scores: bool=False,
        use_causal_mask: bool=True,
    ) -> tf.Tensor:
        if (hasattr(self, "_build_from_signature") and hasattr(self, "_built_from_signature") and not self._built_from_signature):
            self._build_from_signature(query=query, value=value, key=key)
        # attention mask
        __mask = self._compute_attention_mask(query=query, value=value, attention_mask=attention_mask, use_causal_mask=use_causal_mask) # TODO here or after the cache update??
        # init
        __key = value if key is None else key
        # [B, F, N ,H]
        __query = self._query_dense(query)
        # [B, T, N, H]
        __key = self._key_dense(__key)
        # [B, T, N, H]
        __value = self._value_dense(value)
        # update the key cache
        if key_cache is not None:
            __key = mlable.utils.update_cache(tensor=__key, cache=key_cache, step=decode_loop_step, axis=self._attention_axes[0]) # custom seq axis?
        # update the key cache
        if value_cache is not None:
            __value = mlable.utils.update_cache(tensor=__value, cache=value_cache, step=decode_loop_step, axis=self._attention_axes[0]) # custom seq axis?
        # rescale
        __query = tf.multiply(__query, 1.0 / math.sqrt(float(self._key_dim)))
        # attention scores
        __scores = tf.einsum(self._dot_product_equation, __key, __query)
        # [B, N, F, T]
        __scores = self._masked_softmax(__scores, __mask)
        # dropout
        __scores = self._dropout_layer(__scores)
        # [B, F, N, H]
        __output = tf.einsum(self._combine_equation, __scores, __value)
        # projection
        __output = self._output_dense(__output)
        # output
        if return_attention_scores:
            return __output, __scores, __key, __value
        return __output, __key, __value
