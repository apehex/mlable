import math

import tensorflow as tf

import mlable.layers.reshaping
import mlable.shaping

# CONSTANTS ####################################################################

EPSILON = 1e-6

# IMAGE PATCH EXTRACTION #######################################################

class Patching(tf.keras.layers.Layer):
    def __init__(
        self,
        height_dim: int,
        width_dim: int,
        height_axis: int=1,
        width_axis: int=2,
        merge_patch_axes: bool=True,
        merge_space_axes: bool=True,
        **kwargs
    ) -> None:
        # init
        super(Patching, self).__init__(**kwargs)
        # always interpret the smallest axis as height
        __height_axis = min(height_axis, width_axis)
        __width_axis = max(height_axis, width_axis)
        __height_dim = height_dim if height_axis < width_axis else width_dim
        __width_dim = width_dim if height_axis < width_axis else height_dim
        # save for import / export
        self._config = {
            'height_axis': __height_axis,
            'height_dim': __height_dim,
            'width_axis': __width_axis,
            'width_dim': __width_dim,
            'merge_patch_axes': merge_patch_axes,
            'merge_space_axes': merge_space_axes,}
        # reshaping layers
        self._split_height = mlable.layers.reshaping.Divide(input_axis=__height_axis, output_axis=__height_axis + 1, factor=__height_dim, insert=True)
        self._split_width = mlable.layers.reshaping.Divide(input_axis=__width_axis, output_axis=__width_axis + 1, factor=__width_dim, insert=True)
        self._merge_space = mlable.layers.reshaping.Merge(left_axis=__height_axis, right_axis=__height_axis + 1, left=True)
        self._merge_patch = mlable.layers.reshaping.Merge(left_axis=__width_axis + 1, right_axis=__width_axis + 2, left=True) # moved by splitting axes

    def build(self, input_shape: tf.TensorShape=None) -> None:
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # parse the input shape
        __axis_h, __axis_w = self._config['height_axis'], self._config['width_axis']
        # split the last axis first, because it increases the position of the following axes
        __patched = self._split_height(self._split_width(inputs))
        # the width axis has been pushed right by the insertion of the patch height axis
        __perm = mlable.shaping.swap_axes(rank=len(list(__patched.shape)), left=__axis_h + 1, right=__axis_w + 1)
        # group the space axes and the patch axes
        __patched = tf.transpose(__patched, perm=__perm, conjugate=False)
        # merge the last axes first because it moves other axes
        if self._config['merge_patch_axes']:
            __patched = self._merge_patch(__patched)
        if self._config['merge_space_axes']:
            __patched = self._merge_space(__patched)
        # donzo
        return __patched

    def get_config(self) -> dict:
        __config = super(RotaryPositionalEmbedding, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# RECOMPOSE THE IMAGE #########################################################

class Unpatching(tf.keras.layers.Layer):
    def __init__(
        self,
        width_dim: int,
        height_dim: int,
        space_axes: iter=[1, 2],
        patch_axes: iter=[3, 4],
        **kwargs
    ) -> None:
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
        __patched = tf.reshape(inputs, shape=(__batch_dim, __height_num, __width_num, self._config['height'], self._config['width'], __channel_dim))
        # move the patch axes next to the corresponding image axes
        __patched = tf.transpose(__patched, perm=(0, 1, 3, 2, 4, 5), conjugate=False)
        # merge the patch and image axes
        return tf.reshape(__patched, shape=(__batch_dim, __height_num * self._config['height'], __width_num * self._config['width'], __channel_dim))

    def get_config(self) -> dict:
        __config = super(RotaryPositionalEmbedding, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)
