import tensorflow as tf

import mlable.shaping

# GENERIC ######################################################################

@tf.keras.utils.register_keras_serializable(package='layers')
class Reshape(tf.keras.layers.Layer):
    def __init__(
        self,
        target_shape: tuple,
        **kwargs
    ) -> None:
        super(Reshape, self).__init__(**kwargs)
        # save for import / export
        self._config = {'target_shape': target_shape}

    def build(self, input_shape: tuple=None) -> None:
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return tf.reshape(inputs, self._config['target_shape'])

    def get_config(self) -> dict:
        __config = super(Reshape, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# DIVIDE #######################################################################

@tf.keras.utils.register_keras_serializable(package='layers')
class Divide(tf.keras.layers.Layer):
    def __init__(
        self,
        input_axis: int, # relative to the NEW shape / rank
        output_axis: int, # same
        factor: int,
        insert: bool=False,
        **kwargs
    ) -> None:
        super(Divide, self).__init__(**kwargs)
        # save for import / export
        self._config = {
            'input_axis': input_axis,
            'output_axis': output_axis,
            'factor': factor,
            'insert': insert,}

    def build(self, input_shape: tuple=None) -> None:
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # move data from axis 0 to axis 1
        return mlable.shaping.divide(data=inputs, **self._config)

    def get_config(self) -> dict:
        __config = super(Divide, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# MERGE ########################################################################

@tf.keras.utils.register_keras_serializable(package='layers')
class Merge(tf.keras.layers.Layer):
    def __init__(
        self,
        left_axis: int=-2,
        right_axis: int=-1,
        left: bool=True,
        **kwargs
    ) -> None:
        super(Merge, self).__init__(**kwargs)
        # save for import / export
        self._config = {
            'left_axis': left_axis,
            'right_axis': right_axis,
            'left': left,}

    def build(self, input_shape: tuple=None) -> None:
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # merge the two axes
        return mlable.shaping.merge(data=inputs, **self._config)

    def get_config(self) -> dict:
        __config = super(Merge, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# SWAP #########################################################################

@tf.keras.utils.register_keras_serializable(package='layers')
class Swap(tf.keras.layers.Layer):
    def __init__(
        self,
        left_axis: int,
        right_axis: int,
        **kwargs
    ) -> None:
        super(Swap, self).__init__(**kwargs)
        # save for import / export
        self._config = {'left_axis': left_axis, 'right_axis': right_axis,}
        # the actual permutation depends on the rank of the input
        self._perm = []

    def build(self, input_shape: tuple) -> None:
        self._perm = mlable.shaping.swap_axes(rank=len(input_shape), left=self._config['left_axis'], right=self._config['right_axis'], perm=[])
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return tf.transpose(inputs, perm=self._perm, conjugate=False)

    def get_config(self) -> dict:
        __config = super(Swap, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# MOVE #########################################################################

@tf.keras.utils.register_keras_serializable(package='layers')
class Move(tf.keras.layers.Layer):
    def __init__(
        self,
        from_axis: int,
        to_axis: int,
        **kwargs
    ) -> None:
        super(Move, self).__init__(**kwargs)
        # save for import / export
        self._config = {'from_axis': from_axis, 'to_axis': to_axis,}
        # the actual permutation depends on the rank of the input
        self._perm = []

    def build(self, input_shape: tuple) -> None:
        self._perm = mlable.shaping.move_axis(rank=len(input_shape), before=self._config['from_axis'], after=self._config['to_axis'], perm=[])
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return tf.transpose(inputs, perm=self._perm, conjugate=False)

    def get_config(self) -> dict:
        __config = super(Move, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)
