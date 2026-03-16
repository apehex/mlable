import tensorflow as tf

import mlable.shapes

# GROUP ########################################################################

@tf.keras.utils.register_keras_serializable(package='blocks')
class AdaptiveGroupNormalization(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AdaptiveGroupNormalization, self).__init__()
        # save for import / export serialization
        self._config = dict(kwargs)
        # layers
        self._silu = None
        self._norm = None
        self._proj = None

    def build(self, inputs_shape: tuple, contexts_shape: tuple=None) -> None:
        __inputs_shape = tuple(inputs_shape)
        __contexts_shape = tuple(contexts_shape) if (contexts_shape is not None) else __inputs_shape
        # init
        self._silu = tf.keras.activations.silu
        self._norm = tf.keras.layers.GroupNormalization(**self._config)
        self._proj = tf.keras.layers.Dense(
            units=2 * int(__inputs_shape[-1]),
            use_bias=True,
            activation=None,
            kernel_initializer='zeros',
            bias_initializer='zeros',)
        # build
        self._norm.build(__inputs_shape)
        self._proj.build(__contexts_shape)
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, contexts: tf.Tensor=None, training: bool=True, **kwargs) -> tf.Tensor:
        __dtype = inputs.dtype
        __shape = self.compute_context_shape(inputs.shape)
        # regular group norm
        __outputs = self._norm(inputs, training=training)
        # no influence when there is no context
        __scale, __shift = tf.zeros(__shape, dtype=__dtype), tf.zeros(__shape, dtype=__dtype)
        # adapt according to the context embeddings
        if contexts is not None:
            # only accept contexts with rank above 2, to differentiate the batch and feature axes
            assert len(contexts.shape) > 1, 'The context tensor must have at least a batch axis and a feature axis.'
            # match the batch dimension of the inputs
            __shape = self.expand_batch_axis(inputs_shape=inputs.shape, contexts_shape=contexts.shape)
            __contexts = tf.ones(__shape, dtype=__dtype) * contexts
            # double the embeddings dimension to represent both scale and shift
            __contexts = self._proj(self._silu(__contexts))
            # split in two
            __scale, __shift = tf.split(__contexts, 2, axis=-1)
            # match the inputs' rank
            __shape = self.compute_context_shape(inputs.shape)
            __scale = tf.reshape(__scale, shape=__shape)
            __shift = tf.reshape(__shift, shape=__shape)
        # Apply adaptive scale and shift, even when null
        return __shift + __outputs * (1.0 + __scale)

    def compute_context_shape(self, inputs_shape: tuple) -> tuple:
        return tuple(mlable.shapes.filter(inputs_shape, axes=[0, -1]))

    def compute_output_shape(self, inputs_shape: tuple, contexts_shape: tuple=None) -> tuple:
        return tuple(inputs_shape)

    def expand_batch_axis(self, inputs_shape: tuple, contexts_shape: tuple) -> tuple:
        return tuple(inputs_shape[:1]) + tuple(contexts_shape[1:])

    def get_config(self) -> dict:
        __config = super(Reshape, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)
