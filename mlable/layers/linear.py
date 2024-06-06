import keras
import tensorflow as tf

# EINSUM ######################################################################

@keras.saving.register_keras_serializable(package='layers')
class Einsum(tf.keras.layers.Layer):
    def __init__(
        self,
        equation: str,
        shape: tuple,
        **kwargs
    ) -> None:
        super(Einsum, self).__init__(**kwargs)
        self._config = {'equation': equation, 'shape': shape}
        self._w = None

    def build(self, input_shape):
        self._w = self.add_weight(name='w', shape=self._config['shape'], initializer='glorot_normal', trainable=True)
        self.built = True

    def call(self, inputs):
        return tf.einsum(self._config['equation'], inputs, self._w)

    def get_config(self) -> dict:
        __config = super(Einsum, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

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
        self._gelu = tf.keras.layers.Dense(units=self._config['hidden_dim'], activation='gelu', use_bias=False, kernel_initializer='zeros', name='gate')
        self._linear = tf.keras.layers.Dense(units=self._config['hidden_dim'], activation='linear', use_bias=False, kernel_initializer='zeros', name='linear')
        self._output = tf.keras.layers.Dense(units=self._config['input_dim'], activation='linear', use_bias=False, kernel_initializer='zeros', name='output')

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
