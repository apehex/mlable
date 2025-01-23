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
        transpose: bool=False,
        **kwargs
    ) -> None:
        # init
        super(Patching, self).__init__(**kwargs)
        # save for import / export
        self._config = {
            'height_axis': height_axis,
            'width_axis': width_axis,
            'patch_dim': [patch_dim] if isinstance(patch_dim, int) else list(patch_dim),
            'transpose': transpose,}
        # reshaping layers
        self._split_width = None
        self._split_height = None
        self._swap_height = None
        self._swap_width = None
        self._swap_groups = None

    def _divide_args(self, axis: int, dim: int) -> dict:
        return {'input_axis': axis, 'output_axis': axis + 1, 'factor': dim, 'insert': True,}

    def _normalize_axes(self, input_shape: tuple) -> list:
        __rank = len(input_shape)
        return [self._config['height_axis'] % __rank, self._config['width_axis'] % __rank]

    def _normalize_dims(self, input_shape: tuple) -> list:
        __axes_s = self._normalize_axes(input_shape)
        return self._config['patch_dim'][::-1] if (__axes_s[-1] < __axes_s[0]) else self._config['patch_dim']

    def build(self, input_shape: tuple) -> None:
        # normalize negative indexes
        __axes_s = self._normalize_axes(input_shape)
        # match the ordering of the axes
        __dim_p = self._normalize_dims(input_shape)
        # common args
        __split_width = self._divide_args(axis=max(__axes_s), dim=__dim_p[-1])
        __split_height = self._divide_args(axis=min(__axes_s), dim=__dim_p[0])
        # init
        self._split_width = mlable.layers.reshaping.Divide(**__split_width)
        self._split_height = mlable.layers.reshaping.Divide(**__split_height)
        # the width axis has been pushed right by the insertion of the patch height axis
        self._swap_height = mlable.layers.reshaping.Swap(left_axis=min(__axes_s), right_axis=min(__axes_s) + 1)
        self._swap_width = mlable.layers.reshaping.Swap(left_axis=max(__axes_s) + 1, right_axis=max(__axes_s) + 2)
        self._swap_groups = mlable.layers.reshaping.Swap(left_axis=min(__axes_s) + 1, right_axis=max(__axes_s) + 1)
        # no weights
        self._split_height.build()
        self._split_width.build()
        # shape after splitting both height and width axes
        __shape = self._split_height.compute_output_shape(self._split_width.compute_output_shape(input_shape))
        # only the rank is used
        self._swap_height.build(__shape)
        self._swap_width.build(__shape)
        self._swap_groups.build(__shape)
        # register
        self.built = True

    def compute_output_shape(self, input_shape) -> tuple:
        # split width axis
        __shape = self._split_width.compute_output_shape(input_shape)
        # split height axis
        __shape = self._split_height.compute_output_shape(__shape)
        # move the patch axes before the space axes
        if self._config['transpose']:
            # swap the space and patch height axes
            __shape = self._swap_height.compute_output_shape(__shape)
            # swap the space and patch width axes
            __shape = self._swap_width.compute_output_shape(__shape)
        # swap the patch height and the space width axes
        return self._swap_groups.compute_output_shape(__shape)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # split the last axis first, because it increases the position of the following axes
        __outputs = self._split_height(self._split_width(inputs))
        # swap the patch with the space axes => local order rather than global
        if self._config['transpose']:
            __outputs = self._swap_width(self._swap_height(__outputs))
        # group by space and patch instead of height and width
        return self._swap_groups(__outputs)

    def get_config(self) -> dict:
        __config = super(Patching, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# RECOMPOSE THE IMAGE ##########################################################

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
        # save for import / export
        self._config = {
            'space_height_axis': space_height_axis,
            'space_width_axis': space_width_axis,
            'patch_height_axis': patch_height_axis,
            'patch_width_axis': patch_width_axis,}
        # reshaping layers
        self._swap_height = None
        self._swap_width = None
        self._swap_groups = None
        self._merge_width = None
        self._merge_height = None

    def _normalize_config(self, input_shape: tuple) -> dict:
        __rank = len(input_shape)
        return {__k: __v % __rank for __k, __v in self._config.items()}

    def _is_transposed(self, input_shape: tuple) -> bool:
        __config = self._normalize_config(input_shape)
        return max(__config['patch_height_axis'], __config['patch_width_axis']) < min(__config['space_height_axis'], __config['space_width_axis'])

    def build(self, input_shape: tuple) -> None:
        # normalize negative indexes, relative to the input rank
        __config = self._normalize_config(input_shape)
        # by convention, the space axes come first and then the patch axes
        __space_axes = sorted(__config.values())[:2]
        __patch_axes = sorted(__config.values())[-2:]
        # symmetric (space and patch can be swapped)
        self._swap_height = mlable.layers.reshaping.Swap(left_axis=min(__space_axes), right_axis=min(__patch_axes))
        self._swap_width = mlable.layers.reshaping.Swap(left_axis=max(__space_axes), right_axis=max(__patch_axes))
        # asymmetric (space and patch cannot be interverted)
        self._swap_groups = mlable.layers.reshaping.Swap(left_axis=max(__space_axes), right_axis=min(__patch_axes))
        self._merge_width = mlable.layers.reshaping.Merge(left_axis=min(__patch_axes), right_axis=max(__patch_axes), left=True)
        self._merge_height = mlable.layers.reshaping.Merge(left_axis=min(__space_axes), right_axis=max(__space_axes), left=True)
        # build
        self._swap_height.build(input_shape)
        self._swap_width.build(input_shape)
        self._swap_groups.build(input_shape)
        self._merge_width.build()
        self._merge_height.build()
        # register
        self.built = True

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        __shape = tuple(input_shape)
        if self._is_transposed(input_shape):
            # swap the space and patch axes
            __shape = self._swap_height.compute_output_shape(__shape)
            __shape = self._swap_width.compute_output_shape(__shape)
        # group by height and width instead of space and patch
        __shape = self._swap_groups.compute_output_shape(__shape)
        # after swapping, the patch axes are now the width axes
        __shape = self._merge_width.compute_output_shape(__shape)
        # and the space axes are the height axes
        return self._merge_height.compute_output_shape(__shape)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        __outputs = inputs
        # space and patch axes need to be swapped first
        if self._is_transposed(tuple(tf.shape(inputs))):
            __outputs = self._swap_width(self._swap_height(__outputs))
        # group by height and width instead of space and patch
        __outputs = self._swap_groups(__outputs)
        # after swapping, the patch axes are now the width axes
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

# PIXEL PACKING ################################################################

class PixelPacking(tf.keras.layers.Layer):
    def __init__(
        self,
        patch_dim: iter,
        height_axis: int=1,
        width_axis: int=2,
        **kwargs
    ) -> None:
        # init
        super(PixelPacking, self).__init__(**kwargs)
        # normalize
        __patch_dim = [patch_dim] if isinstance(patch_dim, int) else list(patch_dim)
        # save config
        self._config = {
            'patch_dim': __patch_dim,
            'height_axis': height_axis,
            'width_axis': width_axis,}
        # reshaping layers
        self._patch_space = None
        self._merge_patch = None

    def build(self, input_shape: tuple=None) -> None:
        # init
        self._patch_space = Patching(transpose=False, **self._config)
        self._merge_patch = mlable.layers.reshaping.Merge(left_axis=-2, right_axis=-1, left=True)
        # no weights
        self._patch_space.build(input_shape)
        self._merge_patch.build()
        # register
        self.built = True

    def compute_output_shape(self, input_shape) -> tuple:
        # common args
        __args = {'output_axis': -1, 'insert': False,}
        # merge the height axes
        __shape = mlable.shaping.divide_shape(input_shape, input_axis=self._config['height_axis'], factor=self._config['patch_dim'][0], **__args)
        # merge the width axes
        __shape = mlable.shaping.divide_shape(__shape, input_axis=self._config['width_axis'], factor=self._config['patch_dim'][-1], **__args)
        # interpret 0 dimensions as None in symbolic shapes
        return tuple(mlable.shaping.symbolic_shape(__shape))

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # split the space axes into patches
        __outputs = self._patch_space(inputs)
        # merge the feature axis with the patch axes
        return self._merge_patch(self._merge_patch(__outputs))

    def get_config(self) -> dict:
        __config = super(PixelPacking, self).get_config()
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

    def compute_output_shape(self, input_shape) -> tuple:
        # common args
        __args = {'input_axis': -1, 'insert': False,}
        # merge the height axes
        __shape = mlable.shaping.divide_shape(input_shape, output_axis=self._config['height_axis'], factor=self._config['patch_dim'][0], **__args)
        # merge the width axes
        __shape = mlable.shaping.divide_shape(__shape, output_axis=self._config['width_axis'], factor=self._config['patch_dim'][-1], **__args)
        # interpret 0 dimensions as None in symbolic shapes
        return tuple(mlable.shaping.symbolic_shape(__shape))

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
