import math

import tensorflow as tf

import mlable.caching

# CONSTANTS ####################################################################

EPSILON = 1e-6

# FEED FORWARD #################################################################

@tf.keras.utils.register_keras_serializable(package='layers')
class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_dim: int,
        use_bias: bool=True,
        dropout_rate: float=0.0,
        activation: str='gelu',
        **kwargs
    ) -> None:
        super(FeedForwardNetwork, self).__init__(**kwargs)
        # config
        self._config = {
            'hidden_dim': hidden_dim,
            'use_bias': use_bias,
            'dropout_rate': dropout_rate,
            'activation': activation,}
        # layers
        self._hidden = None
        self._dropout = None
        self._output = None

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return tuple(input_shape)

    def build(self, input_shape: tuple) -> None:
        __input_shape = tuple(input_shape)
        __hidden_shape = __input_shape[:-1] + (self._config['hidden_dim'],)
        # common args
        __args = {
            'use_bias': self._config['use_bias'],
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',}
        # init
        self._hidden = tf.keras.layers.Dense(units=__hidden_shape[-1], activation=self._config['activation'], **__args)
        self._dropout = tf.keras.layers.Dropout(rate=self._config['dropout_rate'])
        self._output = tf.keras.layers.Dense(units=__input_shape[-1], activation='linear', **__args)
        # build
        self._hidden.build(__input_shape)
        self._dropout.build(__hidden_shape)
        self._output.build(__hidden_shape)
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        # expand
        __outputs = self._hidden(inputs)
        # drop random values
        __outputs = self._dropout(__outputs, training=training)
        # shrink
        return self._output(__outputs)

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
        self._gate = None
        self._linear = None

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return tuple(input_shape)[:-1] + (self._config['output_dim'],)

    def build(self, input_shape: tuple) -> None:
        # common args
        __args = {
            'units': self._config['output_dim'],
            'use_bias': self._config['use_bias'],
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',}
        # init
        self._gate = tf.keras.layers.Dense(activation=self._config['activation'], **__args)
        self._linear = tf.keras.layers.Dense(activation='linear', **__args)
        # build
        self._gate.build(input_shape)
        self._linear.build(input_shape)
        # register
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
        __kwargs = {}
        # in older versions, the parent methods use a property rather than an arg...
        if hasattr(self, '_return_attention_scores'):
            self._return_attention_scores = return_attention_scores
        else:
            __kwargs = {'return_attention_scores': return_attention_scores}
        # older versions
        if (hasattr(self, '_build_from_signature') and hasattr(self, '_built_from_signature') and not self._built_from_signature):
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
            __key = mlable.caching.update(tensor=__key, cache=cache[0], step=step, axis=self._attention_axes[0]) # custom seq axis?
            __value = mlable.caching.update(tensor=__value, cache=cache[1], step=step, axis=self._attention_axes[0]) # custom seq axis?
            __cache = tf.stack(values=(__key, __value), axis=0)
        # use the parent functionalities
        __outputs, __scores = self._compute_attention(query=__query, key=__key, value=__value, attention_mask=__mask, training=training, **__kwargs)
        # projection
        __outputs = self._output_dense(__outputs)
        # output
        if return_attention_scores:
            return __outputs, __scores, __cache
        return __outputs, __cache
