import keras
import tensorflow as tf

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
