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
