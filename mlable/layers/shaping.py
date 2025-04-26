import tensorflow as tf

import mlable.shapes
import mlable.shaping.axes

# GENERIC ######################################################################

@tf.keras.utils.register_keras_serializable(package='layers')
class Reshape(tf.keras.layers.Layer):
    def __init__(
        self,
        shape: tuple,
        **kwargs
    ) -> None:
        super(Reshape, self).__init__(**kwargs)
        # save for import / export
        self._config = {'shape': shape}

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return tuple(self._config['shape'])

    def build(self, input_shape: tuple=None) -> None:
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return tf.reshape(inputs, self._config['shape'])

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
        axis: int, # relative to the original shape
        factor: int,
        insert: bool=False,
        right: bool=True,
        **kwargs
    ) -> None:
        super(Divide, self).__init__(**kwargs)
        # save for import / export
        self._config = {
            'axis': axis,
            'factor': factor,
            'insert': insert,
            'right': right,}

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        # normalize all dims as ints and divide
        __shape = mlable.shapes.divide(input_shape, **self._config)
        # interpret 0 dimensions as None in symbolic shapes
        return tuple(mlable.shapes.symbolic(__shape))

    def build(self, input_shape: tuple=None) -> None:
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # move data from axis 0 to axis 1
        return mlable.shaping.axes.divide(data=inputs, **self._config)

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
        axis: int,
        right: bool=True,
        **kwargs
    ) -> None:
        super(Merge, self).__init__(**kwargs)
        # save for import / export
        self._config = {
            'axis': axis,
            'right': right,}

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        # normalize all dims as ints and divide
        __shape = mlable.shapes.merge(input_shape, **self._config)
        # interpret 0 dimensions as None in symbolic shapes
        return tuple(mlable.shapes.symbolic(__shape))

    def build(self, input_shape: tuple=None) -> None:
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # merge the two axes
        return mlable.shaping.axes.merge(data=inputs, **self._config)

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

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return tuple(mlable.shapes.swap(input_shape, left=self._config['left_axis'], right=self._config['right_axis']))

    def build(self, input_shape: tuple=None) -> None:
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return mlable.shaping.axes.swap(inputs, **self._config)

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

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return tuple(mlable.shapes.move(input_shape, before=self._config['from_axis'], after=self._config['to_axis']))

    def build(self, input_shape: tuple=None) -> None:
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return mlable.shaping.axes.move(inputs, **self._config)

    def get_config(self) -> dict:
        __config = super(Move, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)
