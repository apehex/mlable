import math

import tensorflow as tf

import mlable.shaping

# CONSTANTS ####################################################################

EPSILON = 1e-6

# IMAGE PATCH EXTRACTION #######################################################

class Patching(tf.keras.layers.Layer):
    def __init__(self, width: int, height: int, merge: bool=True, **kwargs):
        # init
        super(Patching, self).__init__(**kwargs)
        # save for import / export
        self._config = {'width': width, 'height': height, 'merge': merge,}

    def build(self, input_shape: tf.TensorShape=None) -> None:
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # split the 2D inputs
        __patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self._config['height'], self._config['width'], 1],
            strides=[1, self._config['height'], self._config['width'], 1],
            rates=[1, 1, 1, 1],
            padding='VALID')
        # optionally merge the width and height axes
        return mlable.shaping.merge(__patches, left_axis=1, right_axis=2, left=True) if self._config['merge'] else __patches

    def get_config(self) -> dict:
        __config = super(RotaryPositionalEmbedding, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)
