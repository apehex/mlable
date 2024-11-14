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
        self._split_width = None
        self._split_height = None
        self._swap_groups = None

    def build(self, input_shape: tuple) -> None:
        __split_width = {
            'input_axis': self._config['width_axis'],
            'output_axis': self._config['width_axis'] + 1,
            'factor': self._config['patch_dim'][-1],
            'insert': True}
        __split_height = {
            'input_axis': self._config['height_axis'],
            'output_axis': self._config['height_axis'] + 1,
            'factor': self._config['patch_dim'][0],
            'insert': True}
        # shape after splitting both height and width axes
        __shape = mlable.shaping.divide_shape(input_shape, **__split_width)
        __shape = mlable.shaping.divide_shape(__shape, **__split_height)
        # init
        self._split_width = mlable.layers.reshaping.Divide(**__split_width)
        self._split_height = mlable.layers.reshaping.Divide(**__split_height)
        # the width axis has been pushed right by the insertion of the patch height axis
        self._swap_groups = mlable.layers.reshaping.Swap(left_axis=self._config['height_axis'] + 1, right_axis=self._config['width_axis'] + 1)
        # no weights
        self._split_height.build()
        self._split_width.build()
        self._swap_groups.build(__shape)
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # split the last axis first, because it increases the position of the following axes
        __outputs = self._split_height(self._split_width(inputs))
        # group by space and patch instead of height and width
        return self._swap_groups(__outputs)

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
        self._space_axes = sorted([space_height_axis, space_width_axis])
        self._patch_axes = sorted([patch_height_axis, patch_width_axis])
        # save for import / export
        self._config = {
            'space_height_axis': min(self._space_axes),
            'space_width_axis': max(self._space_axes),
            'patch_height_axis': min(self._patch_axes),
            'patch_width_axis': max(self._patch_axes),}
        # reshaping layers
        self._swap_groups = None
        self._merge_width = None
        self._merge_height = None

    def build(self, input_shape: tuple) -> None:
        # init
        self._swap_groups = mlable.layers.reshaping.Swap(left_axis=max(self._space_axes), right_axis=min(self._patch_axes))
        self._merge_width = mlable.layers.reshaping.Merge(left_axis=min(self._patch_axes), right_axis=max(self._patch_axes), left=True)
        self._merge_height = mlable.layers.reshaping.Merge(left_axis=min(self._space_axes), right_axis=max(self._space_axes), left=True)
        # build
        self._swap_groups.build(input_shape)
        self._merge_width.build()
        self._merge_height.build()
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # swap image width with patch height
        __outputs = self._swap_groups(inputs)
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

class PixelShuffle(tf.keras.layers.Layer):
    def __init__(
        self,
        patch_dim: iter,
        height_axis: int=1,
        width_axis: int=2,
        **kwargs
    ) -> None:
        # init
        super(PixelShuffle, self).__init__(**kwargs)
        # normalize
        __patch_dim = [patch_dim] if isinstance(patch_dim, int) else list(patch_dim)
        # save config
        self._config = {
            'patch_dim': __patch_dim,
            'height_axis': height_axis,
            'width_axis': width_axis,}
        # reshaping layers
        self._split_height = None
        self._split_width = None
        self._unpatch_space = None

    def build(self, input_shape: tuple=None) -> None:
        # common args
        __args = {'input_axis': -1, 'output_axis': -2, 'insert': True,}
        # shape after splitting the feature axis
        __shape = mlable.shaping.divide_shape(input_shape, factor=self._config['patch_dim'][0], **__args)
        __shape = mlable.shaping.divide_shape(__shape, factor=self._config['patch_dim'][-1], **__args)
        # init
        self._split_height = mlable.layers.reshaping.Divide(factor=self._config['patch_dim'][0], **__args)
        self._split_width = mlable.layers.reshaping.Divide(factor=self._config['patch_dim'][-1], **__args)
        self._unpatch_space = Unpatching(space_height_axis=self._config['height_axis'], space_width_axis=self._config['width_axis'], patch_height_axis=-3, patch_width_axis=-2)
        # no weights
        self._split_height.build()
        self._split_width.build()
        self._unpatch_space.build(__shape)
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # split the feature axis by chunks of patch size
        __outputs = self._split_width(self._split_height(inputs))
        # merge the patches with the global space
        return self._unpatch_space(__outputs)

    def get_config(self) -> dict:
        __config = super(PixelShuffle, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)
