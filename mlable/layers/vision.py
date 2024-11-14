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
        patch_dim: iter,
        height_axis: int=1,
        width_axis: int=2,
        **kwargs
    ) -> None:
        # init
        super(Patching, self).__init__(**kwargs)
        # the patch dim should always be an iterable
        __patch_dim = [patch_dim] if isinstance(patch_dim, int) else list(patch_dim)
        # match the ordering of the axes
        __patch_dim = __patch_dim[::-1] if (width_axis < height_axis) else __patch_dim
        # always interpret the smallest axis as height
        __height_axis = min(height_axis, width_axis)
        __width_axis = max(height_axis, width_axis)
        # save for import / export
        self._config = {
            'height_axis': __height_axis,
            'width_axis': __width_axis,
            'patch_dim': __patch_dim,}
        # reshaping layers
        self._split_width = mlable.layers.reshaping.Divide(input_axis=__width_axis, output_axis=__width_axis + 1, factor=__patch_dim[-1], insert=True)
        self._split_height = mlable.layers.reshaping.Divide(input_axis=__height_axis, output_axis=__height_axis + 1, factor=__patch_dim[0], insert=True)

    def build(self, input_shape: tuple=None) -> None:
        # no weights
        self._split_height.build()
        self._split_width.build()
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # split the last axis first, because it increases the position of the following axes
        __patched = self._split_height(self._split_width(inputs))
        # the width axis has been pushed right by the insertion of the patch height axis
        __perm = mlable.shaping.swap_axes(rank=len(list(__patched.shape)), left=self._config['height_axis'] + 1, right=self._config['width_axis'] + 1)
        # group the space axes and the patch axes
        return tf.transpose(__patched, perm=__perm, conjugate=False)

    def get_config(self) -> dict:
        __config = super(Patching, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# RECOMPOSE THE IMAGE #########################################################

class Unpatching(tf.keras.layers.Layer):
    def __init__(
        self,
        space_height_axis: int=1,
        space_width_axis: int=2,
        patch_height_axis: int=3,
        patch_width_axis: int=4,
        **kwargs
    ) -> None:
        # init
        super(Unpatching, self).__init__(**kwargs)
        # normalize
        __space_axes = sorted([space_height_axis, space_width_axis])
        __patch_axes = sorted([patch_height_axis, patch_width_axis])
        # save for import / export
        self._config = {
            'space_axes': __space_axes,
            'patch_axes': __patch_axes,}
        # reshaping layers
        self._merge_width = mlable.layers.reshaping.Merge(left_axis=min(__patch_axes), right_axis=max(__patch_axes), left=True)
        self._merge_height = mlable.layers.reshaping.Merge(left_axis=min(__space_axes), right_axis=max(__space_axes), left=True)

    def build(self, input_shape: tuple=None) -> None:
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        __space_axes = list(self._config['space_axes'])
        __patch_axes = list(self._config['patch_axes'])
        # swap image width with patch height
        __perm = mlable.shaping.swap_axes(rank=len(list(inputs.shape)), left=max(__space_axes), right=min(__patch_axes))
        # match the height axis from the patch with the height axis from the image
        __outputs = tf.transpose(inputs, perm=__perm, conjugate=False)
        # after transposing, the patch axes are now the width axes
        __outputs = self._merge_width(__outputs)
        # and the space axes are the height axes
        return self._merge_height(__outputs)

    def get_config(self) -> dict:
        __config = super(Unpatching, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# PIXEL SHUFFLING ##############################################################

class PixelShuffle(Unpatching):
    def __init__(
        self,
        space_dim: iter,
        patch_dim: iter,
        space_axes: iter=[1, 2],
        **kwargs
    ) -> None:
        # normalize
        __space_dim = [space_dim] if isinstance(space_dim, int) else list(space_dim)
        __patch_dim = [patch_dim] if isinstance(patch_dim, int) else list(patch_dim)
        # init
        super(PixelShuffle, self).__init__(space_dim=(__space_dim[0] * __patch_dim[0], __space_dim[-1] * __patch_dim[-1]), patch_dim=__patch_dim, space_axes=space_axes, patch_axes=[-3], **kwargs)
        # reset the config after the unpatch init
        self._config = {
            'space_dim': __space_dim,
            'patch_dim': __patch_dim,
            'space_axes': __space_axes,}
        # reshaping layers
        self._split_feature = mlable.layers.reshaping.Divide(input_axis=-1, output_axis=-2, factor=self._config['patch_dim'][0] * self._config['patch_dim'][-1], insert=True)

    def build(self, input_shape: tuple=None) -> None:
        self._split_feature.build(input_shape)
        # unpatching happens after the feature axis is split
        super(PixelShuffle, self).build(mlable.shaping.divide_shape(input_shape, input_axis=-1, output_axis=-2, factor=self._config['patch_dim'][0] * self._config['patch_dim'][-1], insert=True))
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # split the feature axis by chunks of patch size
        __outputs = self._split_feature(inputs)
        # merge the patches with the global space
        return super(PixelShuffle, self).call(inputs=__outputs, **kwargs)

    def get_config(self) -> dict:
        __config = super(PixelShuffle, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)
