import keras
import tensorflow as tf

# GENERIC #####################################################################

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
