import math

import tensorflow as tf

import mlable.shapes
import mlable.shaping.axes

# CONSTANTS #####################################################################

WAVELENGTH = 10_000

# TRIGONOMETRY ##################################################################

def compute_positions(dim: int, offset: int=0, factor: float=1.0, dtype: tf.dtypes.DType=tf.dtypes.float32) -> tf.Tensor:
    __range = tf.cast(tf.range(dim, dtype=tf.dtypes.float32), dtype=dtype)
    __offset = tf.cast(offset, dtype=dtype)
    __factor = tf.cast(1. / factor, dtype=dtype)
    return __factor * (__range + __offset)

def compute_inverse_freq(dim: int, wavelength: int=WAVELENGTH, dtype: tf.dtypes.DType=tf.dtypes.float32) -> tf.Tensor:
    __exp = tf.divide(tf.range(dim, dtype=tf.float32), tf.cast(dim, dtype=tf.float32))
    return 1.0 / (wavelength ** tf.cast(__exp, dtype=dtype))

# ROPE ##########################################################################

def compute_cos_sin_embedding(sequence_dim: int, feature_dim: int, offset: int=0, factor: float=1.0, wavelength: float=WAVELENGTH, dtype: tf.dtypes.DType=tf.dtypes.float32) -> tuple:
    # inverse frequencies
    __freq = compute_inverse_freq(dim=feature_dim // 2, wavelength=wavelength, dtype=dtype)
    # positions
    __pos = compute_positions(dim=sequence_dim, offset=offset, factor=factor, dtype=dtype)
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
    __shape = mlable.shapes.filter(shape=shape, axes=__axes)
    return tf.reshape(tensor=embeddings, shape=__shape)

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
        __perm = mlable.shapes.swap(shape=__perm, left=__s[0], right=__s[1])
    return tf.transpose(tensor, perm=__perm)

@tf.keras.utils.register_keras_serializable(package='layers')
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

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return tuple(input_shape)

    def build(self, input_shape: tuple=None) -> None:
        self.built = True

    def call(self, inputs: tf.Tensor, offset: int=0, **kwargs) -> tf.Tensor:
        __dtype = inputs.dtype
        __rank = len(list(inputs.shape))
        # swap the seq and feat axes to their defaut positions
        __swaps = swap_to_default(rank=__rank, sequence_axis=self._config['sequence_axis'], feature_axis=self._config['feature_axis'])
        __inputs = transpose_axes(tensor=inputs, swaps=__swaps)
        # meta
        __shape = list(__inputs.shape)
        __sequence_dim = __inputs.shape[1]
        __feature_dim = __inputs.shape[-1]
        # compute the trigo embeddings
        __cos, __sin = compute_cos_sin_embedding(sequence_dim=__sequence_dim, feature_dim=__feature_dim, offset=offset, factor=self._config['scaling_factor'], wavelength=self._config['max_wavelength'], dtype=__dtype)
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
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# SINE #########################################################################

@tf.keras.utils.register_keras_serializable(package='layers')
class SinePositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_axis: int=1, feature_axis: int=-1, wavelength_dim: int=WAVELENGTH, **kwargs) -> None:
        super(SinePositionalEmbedding, self).__init__(**kwargs)
        # save the config to init later
        self._config = {'sequence_axis': sequence_axis, 'feature_axis': feature_axis, 'wavelength_dim': wavelength_dim,}

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return tuple(input_shape)

    def build(self, input_shape: tuple=None) -> None:
        self.built = True

    def call(self, inputs: tf.Tensor, offset: int=0, **kwargs) -> tf.Tensor:
        # dtype of the computations
        __dtype = self.compute_dtype
        # normalize: positive indexes for comparisons
        __rank = len(list(inputs.shape))
        __axis_s = self._config['sequence_axis'] % __rank
        __axis_f = self._config['feature_axis'] % __rank
        # keep only the sequence and feature dimensions, eg (1, S, 1, F)
        __shape = mlable.shapes.filter(shape=inputs.shape, axes=[__axis_s, __axis_f])
        # parse
        __dim_s = __shape[__axis_s]
        __dim_f = __shape[__axis_f]
        # flat range up to sequence dim
        __pos = compute_positions(dim=__dim_s, offset=offset, factor=1.0, dtype=__dtype)
        # inverse frequencies
        __freq = compute_inverse_freq(dim=__dim_f, wavelength=self._config['wavelength_dim'], dtype=__dtype)
        # the angles have shape (S, F)
        __angles = tf.einsum("i,j->ij", __pos, __freq)
        # even indices are sine, odd are cosine
        __mask_c = tf.cast(tf.range(__dim_f, dtype=tf.int32) % 2, __dtype)
        __mask_s = 1.0 - __mask_c
        # embedding shape is [__dim_s, __dim_f]
        __embed = tf.math.sin(__angles) * __mask_s + tf.math.cos(__angles) * __mask_c
        # broadcast to match the inputs shape
        return tf.reshape(__embed, __shape)

    def get_config(self) -> dict:
        __config = super(SinePositionalEmbedding, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# GENERIC ######################################################################

@tf.keras.utils.register_keras_serializable(package='layers')
class GenericEmbedding(tf.keras.layers.Embedding):
    def compute_internal_shape(self, input_shape: tuple) -> tuple:
        return tuple(input_shape)[:1] + (math.prod(tuple(input_shape)[1:]),)

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return tuple(input_shape) + (self.output_dim,)

    def build(self, input_shape: tuple) -> None:
        super(GenericEmbedding, self).build(self.compute_internal_shape(input_shape))

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # this class lifts the limitation on tensor rank (<= 3) in the base class tf.keras.layers.Embedding
        __shape = tuple(inputs.shape)
        # merge all the intermediate axes
        __outputs = tf.reshape(inputs, shape=GenericEmbedding.compute_internal_shape(self, __shape))
        # embed each element separately
        __outputs = tf.keras.layers.Embedding.call(self, __outputs)
        # split the intermediate axes back
        return tf.reshape(__outputs, shape=GenericEmbedding.compute_output_shape(self, __shape))

# POSITION #####################################################################

@tf.keras.utils.register_keras_serializable(package='layers')
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_axis: int=1, feature_axis: int=-1, **kwargs) -> None:
        super(PositionalEmbedding, self).__init__(**kwargs)
        # save the config to init later
        self._config = {'sequence_axis': sequence_axis, 'feature_axis': feature_axis,}
        # create when the input shape is knwon
        self._layer = None

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return tuple(input_shape)

    def build(self, input_shape: tuple) -> None:
        __shape = list(input_shape)
        __rank = len(__shape)
        __axis_s = self._config['sequence_axis'] % __rank
        __axis_f = self._config['feature_axis'] % __rank
        __dim_s = __shape[__axis_s]
        __dim_f = __shape[__axis_f]
        # actually init & build
        self._layer = tf.keras.layers.Embedding(input_dim=__dim_s, output_dim=__dim_f)
        self._layer.build((__dim_s,))
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # normalize: positive indexes for comparisons
        __rank = len(list(inputs.shape))
        __axis_s = self._config['sequence_axis'] % __rank
        __axis_f = self._config['feature_axis'] % __rank
        # keep only the sequence and feature dimensions, eg (1, S, 1, F)
        __shape = mlable.shapes.filter(shape=inputs.shape, axes=[__axis_s, __axis_f])
        # flat range up to sequence dim
        __embed = tf.range(__shape[__axis_s])
        # could just output the kernel, but it might not be built
        __embed = self._layer.call(__embed)
        # transpose the embeddings if the channels come first
        __embed = tf.transpose(__embed, perm=(1, 0), conjugate=False) if (__axis_f < __axis_s) else __embed
        # match the input shape
        return inputs + tf.reshape(__embed, shape=__shape)

    def get_config(self) -> dict:
        __config = super(PositionalEmbedding, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# TOKUN ########################################################################

@tf.keras.utils.register_keras_serializable(package='layers')
class TokunEmbedding(GenericEmbedding):
    def compute_output_shape(self, input_shape: tuple) -> tuple:
        # add the embedding axis
        __shape = super(TokunEmbedding, self).compute_output_shape(input_shape)
        # merge the patch and feature axes
        __shape = mlable.shapes.merge(__shape, axis=-1, right=False)
        # format 0 dimensions as None in symbolic shapes
        return tuple(mlable.shapes.symbolic(__shape))

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # embed each element separately
        __outputs = super(TokunEmbedding, self).call(inputs)
        # concatenate the embeddings
        return mlable.shaping.axes.merge(__outputs, axis=-1, right=False)
