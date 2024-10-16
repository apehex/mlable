import tensorflow as tf

import mlable.shaping

# CONSTANTS #####################################################################

WAVELENGTH = 10_000

# ROPE ##########################################################################

def compute_positions(dim: int, offset: int=0, factor: float=1.0, dtype: tf.dtypes.DType=tf.dtypes.float32) -> tf.Tensor:
    __range = tf.cast(tf.range(dim, dtype=tf.dtypes.float32), dtype=dtype)
    __offset = tf.cast(offset, dtype=dtype)
    __factor = tf.cast(1. / factor, dtype=dtype)
    return __factor * (__range + __offset)

def compute_inverse_freq(dim: int, wavelength: int=WAVELENGTH, dtype: tf.dtypes.DType=tf.dtypes.float32) -> tf.Tensor:
    __freq = tf.divide(tf.cast(tf.range(0, limit=dim, delta=2, dtype=tf.dtypes.float32), dtype=dtype), tf.cast(dim, dtype=dtype))
    return 1.0 / (wavelength ** __freq)

def compute_cos_sin_embedding(sequence_dim: int, feature_dim: int, offset: int=0, factor: float=1.0, wavelength: float=WAVELENGTH, dtype: tf.dtypes.DType=tf.dtypes.float32) -> tuple:
    # inverse frequencies
    __freq = compute_inverse_freq(dim=feature_dim, wavelength=wavelength, dtype=dtype)
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
    __shape = mlable.shaping.filter_shape(shape=shape, axes=__axes)
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
        __perm = mlable.shaping.swap_axes(rank=__rank, left=__s[0], right=__s[1], perm=__perm)
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
        # no weights
        self.build()

    def build(self, input_shape: tf.TensorShape=None) -> None:
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

# POSITION #####################################################################

@tf.keras.utils.register_keras_serializable(package='layers')
class PositionalEmbedding(tf.keras.layers.Embedding):
    def __init__(self, sequence_dim: int, feature_dim: int, sequence_axis: int=1, feature_axis: int=-1, **kwargs) -> None:
        # init
        super(PositionalEmbedding, self).__init__(input_dim=sequence_dim, output_dim=feature_dim, **kwargs)
        # config
        self._config = {'sequence_axis': sequence_axis, 'sequence_dim': sequence_dim, 'feature_axis': feature_axis, 'feature_dim': feature_dim,}

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # normalize: positive indexes for comparisons
        __rank = len(list(inputs.shape))
        __axis_s = self._config['sequence_axis'] % __rank
        __axis_f = self._config['feature_axis'] % __rank
        # keep only the sequence and feature dimensions, eg (1, S, 1, F)
        __shape = mlable.shaping.filter_shape(shape=inputs.shape, axes=[__axis_s, __axis_f])
        # flat range up to sequence dim
        __embed = tf.range(__shape[__axis_s])
        # could just output the kernel, but it might not be built
        __embed = super(PositionalEmbedding, self).call(__embed)
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
class TokunEmbedding(tf.keras.layers.Embedding):
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # embed each element separately
        __outputs = super(TokunEmbedding, self).call(inputs)
        # concatenate the embeddings
        return mlable.shaping.merge(__outputs, left_axis=-2, right_axis=-1, left=True)
