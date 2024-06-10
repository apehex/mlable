import keras
import tensorflow as tf

import mlable.utils

# WITH SIMPLE BIAS ############################################################

@keras.saving.register_keras_serializable(package='layers')
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        input_axis: int=1, # axis of the sequence
        output_axis: int=-1, # axis of the embedding
        **kwargs
    ) -> None:
        super(PositionalEmbedding, self).__init__(**kwargs)
        self._config = {
            'input_axis': input_axis,
            'output_axis': output_axis,}
        self._kernel = None

    def build(self, input_shape: tuple) -> None:
        # shape
        __axes = [self._config['input_axis'] % len(input_shape), self._config['output_axis'] % len(input_shape)]
        __shape = [(__d if __i in __axes else 1) for __i, __d in enumerate(list(input_shape))]
        # init values
        __kernel_init = tf.keras.initializers.GlorotNormal()
        # register the weights
        self._kernel = self.add_weight(name="kernel", shape=__shape, initializer=__kernel_init)
        # notify the model
        self.built = True

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs + self._kernel # each index in the sequence axis has a dedicated bias (different from dense bias)

    def get_config(self) -> dict:
        __config = super(PositionalEmbedding, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# ROPE ########################################################################

WAVELENGTH = 10_000

def compute_positions(dim: int, offset: int=0, factor: float=1.0) -> tf.Tensor:
    __range = tf.range(dim, dtype=tf.dtypes.float32)
    __offset = tf.cast(offset, dtype=tf.dtypes.float32)
    __factor = tf.cast(1. / factor, dtype=tf.dtypes.float32)
    return __factor * (__range + __offset)

def compute_inverse_freq(dim: int, wavelength: int=WAVELENGTH) -> tf.Tensor:
    __freq = tf.divide(tf.range(0, limit=dim, delta=2, dtype=tf.dtypes.float32), tf.cast(dim, dtype=tf.dtypes.float32))
    return 1.0 / (wavelength ** __freq)

def compute_cos_sin_embedding(sequence_dim: int, feature_dim: int, offset: int=0, factor: float=1.0, wavelength: float=WAVELENGTH) -> tuple:
    # inverse frequencies
    __freq = compute_inverse_freq(dim=feature_dim, wavelength=wavelength)
    # positions
    __pos = compute_positions(dim=sequence_dim, offset=offset, factor=factor)
    # (S, E / 2)
    __angles = tf.einsum("i,j->ij", __pos, __freq)
    # (S, E)
    __angles = tf.concat(values=(__angles, __angles), axis=-1)
    # trigonometric embeddings
    return tf.cos(__angles), tf.sin(__angles)

def compute_rotation_embedding(inputs: tf.Tensor, cos_emb: tf.Tensor, sin_emb: tf.Tensor) -> tf.Tensor:
    __x1, __x2 = tf.split(inputs, 2, axis=-1)
    __orthogonal = tf.concat(values=(-__x2, __x1), axis=-1)
    return (inputs * cos_emb) + (__orthogonal * sin_emb)

def reshape_embedding(embeddings: tf.Tensor, shape: list, sequence_axis: int=1, feature_axis: int=-1) -> tf.Tensor:
    __rank = len(shape)
    __axes = [sequence_axis % __rank, feature_axis % __rank]
    __shape = mlable.utils.filter_shape(shape=shape, axes=__axes)
    return tf.reshape(tensor=embeddings, shape=__shape)

def swap_axes(axes: tuple, left: int, right: int) -> tuple:
    __rank = len(axes)
    __perm = list(axes)
    __left, __right = left % __rank, right % __rank
    __perm[__left], __perm[__right] = __perm[__right], __perm[__left]
    return __perm

def swap_to_default(rank: int, sequence_axis: int, feature_axis: int) -> list:
    __swaps = []
    # current positions
    __sequence_axis = sequence_axis % rank
    __feature_axis = feature_axis % rank
    # set sequence_axis to 1
    if __sequence_axis != 1:
        __swaps.append((__sequence_axis, 1))
        # check whether the feature axis was moved
        if __feature_axis == 1:
            __feature_axis = __sequence_axis
    # now the feature axis cannot be 1 (unless feature = sequence = 1 which is wrong)
    if __feature_axis != rank - 1:
        __swaps.append((__feature_axis, -1))
    # conclude
    return __swaps

def transpose_axes(tensor: tf.Tensor, swaps: list) -> tf.Tensor:
    __rank = len(list(tensor.shape))
    __perm = list(range(__rank))
    for __s in swaps:
        __perm = swap_axes(axes=__perm, left=__s[0], right=__s[1])
    return tf.transpose(tensor, perm=__perm)

@keras.saving.register_keras_serializable(package='layers')
class RotaryPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        sequence_axis: int=1,
        feature_axis: int=-1,
        max_wavelength: int=WAVELENGTH,
        scaling_factor: float=1.0,
        **kwargs
    ) -> None:
        # init
        super(RotaryPositionalEmbedding, self).__init__(**kwargs)
        # save for import / export
        self._config = {
            'sequence_axis': sequence_axis,
            'feature_axis': feature_axis,
            'max_wavelength': max_wavelength,
            'scaling_factor': scaling_factor}
        # no weights
        self.built = True

    def call(self, inputs: tf.Tensor, offset: int=0) -> tf.Tensor:
        __rank = len(list(inputs.shape))
        # swap the seq and feat axes to their defaut positions
        __swaps = swap_to_default(rank=__rank, sequence_axis=self._config['sequence_axis'], feature_axis=self._config['feature_axis'])
        __inputs = transpose_axes(tensor=inputs, swaps=__swaps)
        # meta
        __shape = list(__inputs.shape)
        __sequence_dim = inputs.shape[1]
        __feature_dim = inputs.shape[-1]
        # compute the trigo embeddings
        __cos, __sin = compute_cos_sin_embedding(sequence_dim=__sequence_dim, feature_dim=__feature_dim, offset=offset, factor=self._config['scaling_factor'], wavelength=self._config['max_wavelength'])
        # add placeholder axes to match the input shape
        __cos = reshape_embedding(embeddings=__cos, shape=__shape, sequence_axis=1, feature_axis=-1)
        __sin = reshape_embedding(embeddings=__sin, shape=__shape, sequence_axis=1, feature_axis=-1)
        # actually rotate
        __outputs = compute_rotation_embedding(inputs=__inputs, cos_emb=__cos, sin_emb=__sin)
        # swap the axes back, in reverse order
        return transpose_axes(tensor=__outputs, swaps=__swaps[::-1])

    def get_config(self) -> dict:
        __config = super(RotaryPositionalEmbedding, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)