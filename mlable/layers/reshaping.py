import keras as ks
import tensorflow as tf

import mlable.utils

# GENERIC #####################################################################

@ks.saving.register_keras_serializable(package='layers')
class Reshape(ks.layers.Layer):
    def __init__(
        self,
        target_shape: tuple,
        **kwargs
    ) -> None:
        super(Reshape, self).__init__(**kwargs)
        self._config = {'target_shape': target_shape}

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return ks.ops.reshape(x=inputs, newshape=self._config['target_shape'])

    def get_config(self) -> dict:
        __config = super(Reshape, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> ks.layers.Layer:
        return cls(**config)

# DIVIDE ######################################################################

@ks.saving.register_keras_serializable(package='layers')
class Divide(ks.layers.Layer):
    def __init__(
        self,
        input_axis: int, # relative to the NEW shape / rank
        output_axis: int, # same
        factor: int,
        insert: bool=False,
        **kwargs
    ) -> None:
        super(Divide, self).__init__(**kwargs)
        self._config = {
            'input_axis': input_axis,
            'output_axis': output_axis,
            'factor': factor,
            'insert': insert,}

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # infer the dimension of the symbolic axis
        __shape = mlable.utils.normalize_shape(list(inputs.shape))
        # move data from axis 0 to axis 1
        __shape = mlable.utils.divide_shape(shape=__shape, **self._config)
        # actually reshape
        return ks.ops.reshape(x=inputs, newshape=__shape)

    def get_config(self) -> dict:
        __config = super(Divide, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> ks.layers.Layer:
        return cls(**config)

# MERGE #######################################################################

@ks.saving.register_keras_serializable(package='layers')
class Merge(ks.layers.Layer):
    def __init__(
        self,
        left_axis: int=-2,
        right_axis: int=-1,
        left: bool=True,
        **kwargs
    ) -> None:
        super(Merge, self).__init__(**kwargs)
        self._config = {
            'left_axis': left_axis,
            'right_axis': right_axis,
            'left': left,}

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # infer the dimension of the symbolic axis
        __shape = mlable.utils.normalize_shape(list(inputs.shape))
        # new shape
        __shape = mlable.utils.merge_shape(shape=__shape, **self._config)
        # actually merge the two axes
        return ks.ops.reshape(x=inputs, newshape=__shape)

    def get_config(self) -> dict:
        __config = super(Merge, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> ks.layers.Layer:
        return cls(**config)
