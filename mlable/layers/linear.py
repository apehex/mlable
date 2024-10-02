import tensorflow as tf

# EINSUM ######################################################################

@tf.keras.utils.register_keras_serializable(package='layers')
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

    def build(self, input_shape: tf.TensorShape) -> None:
        self._w = self.add_weight(name='w', shape=self._config['shape'], initializer='glorot_normal', trainable=True)
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return tf.einsum(self._config['equation'], inputs, self._w)

    def get_config(self) -> dict:
        __config = super(Einsum, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)
