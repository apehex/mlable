import math

import tensorflow as tf

import mlable.shaping

# CONSTANTS ####################################################################

EPSILON = 1e-6

# IMAGE PATCH EXTRACTION #######################################################

class Patching(tf.keras.layers.Layer):
    def __init__(self, width: int, height: int, **kwargs):
        # init
        super(Patching, self).__init__(**kwargs)
        # save for import / export
        self._config = {'width': width, 'height': height,}

    def build(self, input_shape: tf.TensorShape=None) -> None:
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return tf.image.extract_patches(
            images=inputs,
            sizes=[1, self._config['height'], self._config['width'], 1],
            strides=[1, self._config['height'], self._config['width'], 1],
            rates=[1, 1, 1, 1],
            padding='VALID')

    def get_config(self) -> dict:
        __config = super(RotaryPositionalEmbedding, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# RECOMPOSE THE IMAGE #########################################################

class Unpatching(tf.keras.layers.Layer):
    def __init__(self, width: int, height: int, **kwargs):
        # init
        super(Unpatching, self).__init__(**kwargs)
        # save for import / export
        self._config = {'width': width, 'height': height,}

    def build(self, input_shape: tf.TensorShape=None) -> None:
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # parse the inputs shape
        __batch_dim, __height_num, __width_num, __channel_dim = tuple(inputs.shape)
        # patch channels => pixel channels
        __channel_dim = __channel_dim // (self._config['width'] * self._config['height'])
        # split the patch channels into individual pixels
        __patches = tf.reshape(inputs, shape=(__batch_dim, __height_num, __width_num, self._config['height'], self._config['width'], __channel_dim))
        # move the patch axes next to the corresponding image axes
        __patches = tf.transpose(__patches, perm=(0, 1, 3, 2, 4, 5), conjugate=False)
        # merge the patch and image axes
        return tf.reshape(__patches, shape=(__batch_dim, __height_num * self._config['height'], __width_num * self._config['width'], __channel_dim))

    def get_config(self) -> dict:
        __config = super(RotaryPositionalEmbedding, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)
