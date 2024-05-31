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
        __parent_config = super(PositionalEmbedding, self).get_config()
        return {**__parent_config, **self._config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# ROPE ########################################################################

MAX_WAVELENGTH = 10_000

def filter_shape(shape: list, axes: list) -> list:
    return [__d if __i in axes else 1 for __i, __d in enumerate(shape)]

@keras.saving.register_keras_serializable(package='layers')
class RotaryPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        head_dim: int,
        sequence_axis: int=1,
        max_wavelength: int=MAX_WAVELENGTH,
        **kwargs
    ) -> None:
        super(RotaryPositionalEmbedding, self).__init__(**kwargs)
        self._config = {
            'head_dim': head_dim,
            'sequence_axis': sequence_axis,
            'max_wavelength': max_wavelength,}

    def call(self, inputs: tf.Tensor, positions: tf.Tensor) -> tf.Tensor:
        # # position shape
        # __shape = filter_shape(shape=list(inputs.shape), axes=[self._config['sequence_axis'] % len(inputs.shape)])
        # # position indices
        # __positions = tf.reshape(tensor=tf.range(max(__shape), dtype=tf.float32), shape=__shape)
        # timescale
        __fraction = 2 * tf.range(self._config['head_dim'] // 2, dtype=tf.float32) / self._config['head_dim']
        __timescale = self._config['max_wavelength'] ** __fraction
        # angle inputs
        __angles = tf.expand_dims(positions, axis=-1) / tf.reshape(__timescale, [1, 1, -1])
        # actual sinusoids
        __sin = tf.sin(__angles)
        __cos = tf.cos(__angles)
        # Split inputs into two halves
        __left, __right = tf.split(inputs, 2, axis=-1)
        # apply the rotation
        __left_rot = __left * __cos - __right * __sin
        __right_rot = __left * __sin + __right * __cos
        # concatenate the rotated parts
        __output = tf.concat([__left_rot, __right_rot], axis=-1)
        # make sure the type is correct
        return tf.cast(__output, inputs.dtype)

    def get_config(self) -> dict:
        __parent_config = super(PositionalEmbedding, self).get_config()
        return {**__parent_config, **self._config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
