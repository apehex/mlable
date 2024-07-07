import keras
import numpy as np
import tensorflow as tf

# ACTIVATIONS #################################################################

@keras.saving.register_keras_serializable(package='layers')
class Activation(tf.keras.layers.Layer):
    def __init__(
        self,
        function: callable,
        **kwargs
    ):
        super(Activation, self).__init__(**kwargs)
        self._function = function

    def call(self, inputs: tf.Tensor, **kwargs):
        return self._function(inputs)

    def get_config(self) -> dict:
        __config = super(Activation, self).get_config()
        __config.update({'function': keras.saving.serialize_keras_object(self._function),})
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        __fn_config = config.pop('function')
        __fn = keras.saving.deserialize_keras_object(__fn_config)
        return cls(function=__fn, **config)

@keras.saving.register_keras_serializable(package='layers')
class Softmax(tf.keras.layers.Layer):
    def __init__(
        self,
        axis: int=-1,
        **kwargs
    ):
        super(Softmax, self).__init__(**kwargs)
        self._config = {'axis': axis}

    def call(self, inputs: tf.Tensor, **kwargs):
        return tf.nn.softmax(inputs, axis=self._config['axis'])

    def get_config(self) -> dict:
        __parent_config = super(Softmax, self).get_config()
        return {**__parent_config, **self._config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# NORMALIZATION ###############################################################

@keras.saving.register_keras_serializable(package='layers')
class BatchNormalization(tf.keras.layers.Layer):
    def __init__(
        self,
        axis: int=0,
        momentum: float=0.99,
        epsilon: float=0.001,
        **kwargs
    ):
        super(BatchNormalization, self).__init__(**kwargs)
        self._config = {
            'axis': axis,
            'momentum': momentum,
            'epsilon': epsilon,}
        self._mean = None
        self._stddev = None
        self._gain = None
        self._bias = None

    def build(self, input_shape: tuple):
        # shape
        __axis = self._config['axis'] % len(input_shape) # positive index even when the axis is specified negatively, like -2
        __shape = [1 if __i == __axis else __d for __i, __d in enumerate(input_shape)]
        # values
        __mean_init = tf.keras.initializers.GlorotNormal()
        __stddev_init = tf.keras.initializers.GlorotNormal()
        __gain_init = tf.keras.initializers.GlorotNormal()
        __bias_init = tf.keras.initializers.Zeros()
        # tensors
        self._mean = self.add_weight("mean", shape=__shape, initializer=__mean_init)
        self._stddev = self.add_weight("stddev", shape=__shape, initializer=__stddev_init)
        self._gain = self.add_weight("gain", shape=__shape, initializer=__gain_init)
        self._bias = self.add_weight("bias", shape=__shape, initializer=__bias_init)
        # notify the model
        self.built = True

    def call(self, inputs: tf.Tensor, training: bool=True, **kwargs):
        if training:
            # current values
            __batch_mean = tf.math.reduce_mean(inputs, axis=self._config['axis'], keepdims=True)
            __batch_stddev = tf.math.reduce_std(inputs, axis=self._config['axis'], keepdims=True)
            # update parameters
            self._mean = tf.stop_gradient(self._config['momentum'] * self._mean + (1. - self._config['momentum']) * __batch_mean)
            self._stddev = tf.stop_gradient(self._config['momentum'] * self._stddev + (1. - self._config['momentum']) * __batch_stddev)
        # normalize
        __normalized = tf.math.divide(inputs - self._mean, self._stddev + self._config['epsilon'])
        # scale
        return tf.math.multiply(self._gain, __normalized) + self._bias

    def get_config(self) -> dict:
        __parent_config = super(BatchNormalization, self).get_config()
        return {**__parent_config, **self._config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

@keras.saving.register_keras_serializable(package='layers')
class LayerNormalization(tf.keras.layers.Layer):
    def __init__(
        self,
        axis: int=-1,
        epsilon: float=0.001,
        **kwargs
    ):
        super(LayerNormalization, self).__init__(**kwargs)
        self._config= {
            'axis': axis,
            'epsilon': epsilon,}
        self._gain = None
        self._bias = None

    def build(self, input_shape: tuple):
        # shape
        __shape = [1] + input_shape[1:]
        # values
        __gain_init = tf.keras.initializers.GlorotNormal()
        __bias_init = tf.keras.initializers.Zeros()
        # tensors
        self._gain = self.add_weight("gain", shape=__shape, initializer=__gain_init)
        self._bias = self.add_weight("bias", shape=__shape, initializer=__bias_init)
        # notify the model
        self.built = True

    def call(self, inputs: tf.Tensor, training: bool=True, **kwargs):
        # current values
        __layer_mean = tf.math.reduce_mean(inputs, axis=self._config['axis'], keepdims=True)
        __layer_stddev = tf.math.reduce_std(inputs, axis=self._config['axis'], keepdims=True)
        # normalize
        __normalized = tf.math.divide(inputs - __layer_mean, __layer_stddev + self._config['epsilon'])
        # scale
        return tf.math.add(tf.math.multiply(self._gain, __normalized), self._bias)

    def get_config(self) -> dict:
        __parent_config = super(LayerNormalization, self).get_config()
        return {**__parent_config, **self._config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# LINEAR ######################################################################

@keras.saving.register_keras_serializable(package='layers')
class Dense(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int,
        use_bias: bool=True,
        **kwargs
    ):
        super(Dense, self).__init__(**kwargs)
        self._config = {
            'units': units,
            'use_bias': use_bias,}
        self._kernel = None
        self._bias = None

    def build(self, input_shape: tuple):
        # kernel
        __kernel_init = tf.keras.initializers.GlorotNormal()
        self._kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), self._config['units']], initializer=__kernel_init)
        # bias
        if self._config['use_bias']:
            __bias_init = tf.keras.initializers.Zeros()
            self._bias = self.add_weight("bias", shape=[self._config['units']], initializer=__bias_init)
        # notify the model
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs):
        return tf.matmul(inputs, self._kernel) + self._bias if (self._config['use_bias'] and self._bias is not None) else tf.matmul(inputs, self._kernel)

    def get_config(self) -> dict:
        __parent_config = super(Dense, self).get_config()
        return {**__parent_config, **self._config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# EMBEDDING ###################################################################

@keras.saving.register_keras_serializable(package='layers')
class Embedding(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        **kwargs
    ):
        super(Embedding, self).__init__(**kwargs)
        self._config = {
            'input_dim': input_dim,
            'output_dim': output_dim,}
        self._kernel = None

    def build(self, input_shape: tuple):
        __kernel_init = tf.keras.initializers.GlorotNormal()
        # register the weights
        self._kernel = self.add_weight(name="kernel", shape=[self._config['input_dim'], self._config['output_dim']], initializer=__kernel_init)
        # notify the model
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs):
        # content embedding
        __x = tf.one_hot(indices=inputs, depth=self._config['input_dim'], dtype=tf.dtypes.float32)
        return tf.matmul(__x, self._kernel)

    def get_config(self) -> dict:
        __parent_config = super(Embedding, self).get_config()
        return {**__parent_config, **self._config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# POSITION EMBEDDING ##########################################################

@keras.saving.register_keras_serializable(package='layers')
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        input_axis: int=1, # axis of the sequence
        output_axis: int=-1, # axis of the embedding
        **kwargs
    ) -> None:
        super(PositionalEmbedding, self).__init__(**kwargs)
        self._config = {
            'input_axis': input_axis,
            'output_axis': output_axis,}
        self._kernel = None

    def build(self, input_shape: tuple) -> None:
        # shape
        __axes = [self._config['input_axis'] % len(input_shape), self._config['output_axis'] % len(input_shape)]
        __shape = [(__d if __i in __axes else 1) for __i, __d in enumerate(list(input_shape))]
        # init values
        __kernel_init = tf.keras.initializers.GlorotNormal()
        # register the weights
        self._kernel = self.add_weight(name="kernel", shape=__shape, initializer=__kernel_init)
        # notify the model
        self.built = True

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs + self._kernel # each index in the sequence axis has a dedicated bias (different from dense bias)

    def get_config(self) -> dict:
        __config = super(PositionalEmbedding, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# QUADRATIC ###################################################################

@keras.saving.register_keras_serializable(package='layers')
class Attention(tf.keras.layers.Layer):
    def __init__(
        self,
        head_dim: int,
        head_count: int=1,
        **kwargs
    ):
        super(Attention, self).__init__(**kwargs)
        self._time_dim = None
        self._head_dim = head_dim
        self._head_count = head_count
        self._key = None
        self._query = None
        self._value = None

    def build(self, input_shape: tuple) -> None:
        self._time_dim = list(input_shape)[-2]
        # init
        __key_init = tf.keras.initializers.GlorotNormal()
        __query_init = tf.keras.initializers.GlorotNormal()
        __value_init = tf.keras.initializers.GlorotNormal()
        # kernels
        self._key = self.add_weight("key", shape=[int(input_shape[-1]), self._head_dim], initializer=__key_init)
        self._query = self.add_weight("query", shape=[int(input_shape[-1]), self._head_dim], initializer=__query_init)
        self._value = self.add_weight("value", shape=[self._time_dim, self._head_dim], initializer=__value_init)
        # notify the model
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # transpose the last two axes
        __perm = list(range(len(list(inputs.shape))))
        __perm[-1] = len(__perm) - 2
        __perm[-2] = len(__perm) - 1
        # key
        __k = tf.matmul(inputs, self._key) # (B, T, E) * (E, H) = (B, T, H)
        # query
        __q = tf.matmul(inputs, self._query) # (B, T, E) * (E, H) = (B, T, H)
        # weight
        __w = tf.matmul(__k, tf.transpose(__q, perm=__perm)) / tf.math.sqrt(float(self._head_dim)) # (B, T, H) * (B, H, T) = (B, T, T)
        # mask
        __m = tf.linalg.band_part(tf.ones((self._time_dim, self._time_dim)), num_lower=0, num_upper=-1) - tf.linalg.diag(self._time_dim * [1.]) # (T, T)
        __u = tf.where(__m == 1., -np.inf, 0.) # (T, T)
        __l = tf.linalg.band_part(__w, num_lower=-1, num_upper=0) # (B, T, T) may fail because of the first dimension => diag of tensor with 3 axes
        # probabilities
        __w = tf.nn.softmax(__u + __l, axis=-1) # (T, T) + (B, T, T) = (B, T, T)
        # value
        return tf.matmul(__w, self._value) # (B, T, T) * (T, H) = (B, T, H)

    def get_config(self) -> dict:
        __parent_config = super(Attention, self).get_config()
        __child_config = {
            'head_dim': self._head_dim,
            'head_count': self._head_count}
        return {**__parent_config, **__child_config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# FEED FORWARD BLOCK ##########################################################

@keras.saving.register_keras_serializable(package='layers')
class ResidualFeedForwardBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        normalization_epsilon: float=0.001,
        **kwargs
    ) -> None:
        # init
        super(ResidualFeedForwardBlock, self).__init__(**kwargs)
        # config
        self._config = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'normalization_epsilon': normalization_epsilon}
        # layers
        self._normalization = tf.keras.layers.LayerNormalization(axis=-1, epsilon=normalization_epsilon, center=True, scale=True, beta_initializer='zeros', gamma_initializer='glorot_normal')
        self._hidden = tf.keras.layers.Dense(units=hidden_dim, activation='relu', use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros')
        self._projection = tf.keras.layers.Dense(units=input_dim, activation='linear', use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros')

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        __dx = inputs # (B, T, C)
        # normalize the features
        __dx = self._normalization(__dx) # (B, T, C)
        # expand inside the hidden layer
        __dx = self._hidden(__dx) # (B, T, C) * (C, H) = (B, T, H)
        # projection: match the input shape
        __dx = self._projection(__dx) # (B, T, H) * (H, C) = (B, T, C)
        # residual
        return inputs + __dx # (B, T, C)

    def get_config(self) -> dict:
        __parent_config = super(ResidualFeedForwardBlock, self).get_config()
        return {**__parent_config, **self._config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# ATTENTION BLOCK #############################################################

@keras.saving.register_keras_serializable(package='layers')
class ResidualSelfAttentionBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        attention_head_dim: int,
        attention_head_count: int=1,
        normalization_epsilon: float=0.001,
        dropout: float=0.0,
        **kwargs
    ) -> None:
        # init
        super(ResidualSelfAttentionBlock, self).__init__(**kwargs)
        # config
        self._config = {
            'attention_head_dim': attention_head_dim,
            'attention_head_count': attention_head_count,
            'normalization_epsilon': normalization_epsilon,
            'dropout': dropout}
        # layers
        self._normalization = tf.keras.layers.LayerNormalization(axis=-1, epsilon=normalization_epsilon, center=True, scale=True, beta_initializer='zeros', gamma_initializer='glorot_normal')
        self._attention = tf.keras.layers.MultiHeadAttention(num_heads=attention_head_count, key_dim=attention_head_dim, value_dim=attention_head_dim, dropout=dropout, use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros')

    def call(self, inputs: tf.Tensor)  -> tf.Tensor:
        __dx = inputs # (B, T, C)
        # normalize the features
        __dx = self._normalization(__dx) # (B, T, C)
        # self-attention
        __dx = self._attention(key=__dx, query=__dx, value=__dx, return_attention_scores=False, use_causal_mask=True) # (B, T, H_d * H_c) = (B, T, C) use_causal_mask=True
        # residual
        return inputs + __dx # (B, T, C)

    def get_config(self) -> dict:
        __parent_config = super(ResidualSelfAttentionBlock, self).get_config()
        return {**__parent_config, **self._config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# META BLOCK ##################################################################

@keras.saving.register_keras_serializable(package='layers')
class ResidualSelfAttentionDecoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        attention_head_dim: int,
        attention_head_count: int=1,
        normalization_epsilon: float=0.001,
        dropout: float=0.0,
        **kwargs
    ) -> None:
        # init
        super(ResidualSelfAttentionDecoderBlock, self).__init__(**kwargs)
        # config
        self._config = {
            'hidden_dim': hidden_dim,
            'attention_head_dim': attention_head_dim,
            'attention_head_count': attention_head_count,
            'normalization_epsilon': normalization_epsilon,
            'dropout': dropout}
        # layers
        self._feedforward = ResidualFeedForwardBlock(input_dim=input_dim,hidden_dim=hidden_dim, normalization_epsilon=normalization_epsilon)
        self._attention = ResidualSelfAttentionBlock(attention_head_dim=attention_head_dim, attention_head_count=attention_head_count, normalization_epsilon=normalization_epsilon, dropout=dropout)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        __dx = inputs # (B, T, C)
        # residual self-attention
        __dx = self._attention(__dx) # (B, T, C)
        # residual FF
        __dx = self._feedforward(__dx) # (B, T, C)
        # residual
        return __dx # (B, T, C)

    def get_config(self) -> dict:
        __parent_config = super(ResidualSelfAttentionDecoderBlock, self).get_config()
        return {**__parent_config, **self._config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
