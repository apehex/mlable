import keras
import tensorflow as tf

# WITH SIMPLE BIAS ############################################################

@keras.saving.register_keras_serializable(package='layers')
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        input_axis: int=1, # axis of the sequence
        output_axis: int=-1, # axis of the embedding
        **kwargs
    ):
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
        __parent_config = super(PositionalEmbedding, self).get_config()
        return {**__parent_config, **self._config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# ROPE ########################################################################
