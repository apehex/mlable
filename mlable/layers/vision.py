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
        merge_patch_axes: bool=True,
        merge_space_axes: bool=True,
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
            'patch_dim': __patch_dim,
            'merge_patch_axes': merge_patch_axes,
            'merge_space_axes': merge_space_axes,}
        # reshaping layers
        self._split_width = mlable.layers.reshaping.Divide(input_axis=__width_axis, output_axis=__width_axis + 1, factor=__patch_dim[-1], insert=True)
        self._split_height = mlable.layers.reshaping.Divide(input_axis=__height_axis, output_axis=__height_axis + 1, factor=__patch_dim[0], insert=True)
        self._merge_patch = mlable.layers.reshaping.Merge(left_axis=__width_axis + 1, right_axis=__width_axis + 2, left=True) # moved by splitting axes
        self._merge_space = mlable.layers.reshaping.Merge(left_axis=__height_axis, right_axis=__height_axis + 1, left=True)

    def build(self, input_shape: tuple=None) -> None:
        # no weights
        self._split_height.build()
        self._split_width.build()
        self._merge_space.build()
        self._merge_patch.build()
        # register
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
        # merge the last (patch) axes first because it doesn't affect axes before (space)
        if self._config['merge_patch_axes']:
            __patched = self._merge_patch(__patched)
        if self._config['merge_space_axes']:
            __patched = self._merge_space(__patched)
        # donzo
        return __patched

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
        space_dim: iter,
        patch_dim: iter,
        space_axes: iter=[1, 2],
        patch_axes: iter=[3, 4],
        **kwargs
    ) -> None:
        # init
        super(Unpatching, self).__init__(**kwargs)
        # normalize & check
        __space_axes = [space_axes] if isinstance(space_axes, int) else list(space_axes)
        __patch_axes = [patch_axes] if isinstance(patch_axes, int) else list(patch_axes)
        __space_dim = [space_dim] if isinstance(space_dim, int) else list(space_dim)
        __patch_dim = [patch_dim] if isinstance(patch_dim, int) else list(patch_dim)
        # match the ordering of the axes (see below)
        __space_dim = __space_dim[::-1] if (__space_axes[-1] < __space_axes[0]) else __space_dim
        __patch_dim = __patch_dim[::-1] if (__patch_axes[-1] < __patch_axes[0]) else __patch_dim
        # sort the axes
        __space_axes = sorted(__space_axes)
        __patch_axes = sorted(__patch_axes)
        # save for import / export
        self._config = {
            'space_dim': __space_dim,
            'patch_dim': __patch_dim,
            'space_axes': __space_axes,
            'patch_axes': __patch_axes,}
        # reshaping layers
        self._split_patch = mlable.layers.reshaping.Divide(input_axis=min(__patch_axes), output_axis=min(__patch_axes) + 1, factor=__patch_dim[-1], insert=True)
        self._split_space = mlable.layers.reshaping.Divide(input_axis=min(__space_axes), output_axis=min(__space_axes) + 1, factor=__space_dim[-1] // __patch_dim[-1], insert=True)

    def build(self, input_shape: tuple=None) -> None:
        # no weights
        self._split_patch.build()
        self._split_space.build()
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        __space_axes = list(self._config['space_axes'])
        __patch_axes = list(self._config['patch_axes'])
        __outputs = inputs
        # split the patch axes first
        if len(__patch_axes) < 2:
            __outputs = self._split_patch(__outputs)
            __patch_axes.append(min(__patch_axes) + 1)
        # the axis insertion moves the patch axes to the right
        if len(__space_axes) < 2:
            __outputs = self._split_space(__outputs)
            __space_axes.append(min(__space_axes) + 1)
            __patch_axes = [__a + 1 for __a in __patch_axes]
        # swap image width with patch height
        __perm = mlable.shaping.swap_axes(rank=len(list(__outputs.shape)), left=max(__space_axes), right=min(__patch_axes))
        # match the height axis from the patch with the height axis from the image
        __outputs = tf.transpose(__outputs, perm=__perm, conjugate=False)
        # after transposing, the space axes are the height axes from both space and the patch axes are now the width axes
        __outputs = mlable.shaping.merge(__outputs, left_axis=min(__patch_axes), right_axis=max(__patch_axes), left=True)
        return mlable.shaping.merge(__outputs, left_axis=min(__space_axes), right_axis=max(__space_axes), left=True)

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
