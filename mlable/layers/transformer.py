import math

import tensorflow as tf

import mlable.utils

# CONSTANTS ####################################################################

EPSILON = 1e-6

# FEED FORWARD #################################################################

@tf.keras.utils.register_keras_serializable(package='layers')
class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        use_bias: bool=True,
        activation: str='gelu',
        **kwargs
    ) -> None:
        super(FeedForwardNetwork, self).__init__(**kwargs)
        # config
        self._config = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'use_bias': use_bias,
            'activation': activation,}
        # layers
        self._hidden = tf.keras.layers.Dense(units=self._config['hidden_dim'], activation=activation, use_bias=use_bias, kernel_initializer='glorot_uniform', bias_initializer='zeros')
        self._output = tf.keras.layers.Dense(units=self._config['input_dim'], activation='linear', use_bias=use_bias, kernel_initializer='glorot_uniform', bias_initializer='zeros')

    def build(self, input_shape: tf.TensorShape) -> None:
        __hidden_shape = list(input_shape)
        __hidden_shape[-1] = self._config['hidden_dim']
        # directly handle the input
        self._hidden.build(input_shape)
        # from hidden to input dim
        self._output.build(__hidden_shape)
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # gating mechanism
        return self._output(self._hidden(inputs))

    def get_config(self) -> dict:
        __config = super(FeedForwardNetwork, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# GATE #########################################################################

@tf.keras.utils.register_keras_serializable(package='layers')
class GatedLinearUnit(tf.keras.layers.Layer):
    def __init__(
        self,
        output_dim: int,
        use_bias: bool=True,
        activation: str='gelu',
        **kwargs
    ) -> None:
        super(GatedLinearUnit, self).__init__(**kwargs)
        # config
        self._config = {
            'output_dim': output_dim,
            'use_bias': use_bias,
            'activation': activation,}
        # layers
        self._gate = tf.keras.layers.Dense(units=output_dim, activation=activation, use_bias=use_bias, kernel_initializer='glorot_uniform', bias_initializer='zeros')
        self._linear = tf.keras.layers.Dense(units=output_dim, activation='linear', use_bias=use_bias, kernel_initializer='glorot_uniform', bias_initializer='zeros')

    def build(self, input_shape: tf.TensorShape) -> None:
        self._gate.build(input_shape)
        self._linear.build(input_shape)
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return self._gate(inputs) * self._linear(inputs)

    def get_config(self) -> dict:
        __config = super(GatedLinearUnit, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# ATTENTION ####################################################################

@tf.keras.utils.register_keras_serializable(package="layers")
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
        **kwargs
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
