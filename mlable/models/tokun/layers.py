import tensorflow as tf

import mlable.tensorflow.layers as _mtl

# ENCODING BLOCKS #############################################################

class TokenizeBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        left_axis: int=-2,
        right_axis: int=-1,
        token_dim: int=4,
        latent_dim: int=256,
        **kwargs
    ) -> None:
        super(TokenizeBlock, self).__init__(**kwargs)
        # layers
        self._divide = _mtl.Divide(input_axis=0, output_axis=1, factor=token_dim, insert=True, name='group') # (B * G, E) => (B, G, E)
        self._embedding = _mtl.PositionalEmbedding(input_axis=left_axis, output_axis=right_axis, name='position-embeddings') # (B, G, E) + (1, G, E)
        self._merge = _mtl.Merge(left_axis=left_axis, right_axis=right_axis, left=True, name='merge-embeddings') # (B, G, E) => (B, G * E)
        self._dense = tf.keras.layers.Dense(units=latent_dim, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='compress-embeddings') # (B, G * E) => (B, L), typically L = E

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._dense(self._merge(self._embedding(self._divide(inputs))))

# DECODING BLOCKS #############################################################

class DetokenizeBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        token_dim: int=4,
        embedding_dim: int=256,
        **kwargs
    ) -> None:
        super(DetokenizeBlock, self).__init__(**kwargs)
        # layers
        self._dense = tf.keras.layers.Dense(units=token_dim * embedding_dim, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='decompress-embeddings') # (B, L) => (B, G * E), typically L = E
        self._divide = _mtl.Divide(input_axis=-2, output_axis=-1, insert=True, factor=embedding_dim, name='divide-embeddings') # (B, G * E) => (B, G, E)
        self._merge = _mtl.Merge(left_axis=0, right_axis=1, left=True) # (B, G, E) => (B * G, E)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._merge(self._divide(self._dense(inputs)))

# HEAD ########################################################################

class HeadBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        encoding_dim: int=256,
        **kwargs
    ) -> None:
        super(HeadBlock, self).__init__(**kwargs)
        # layers
        self._dense = tf.keras.layers.Dense(units=encoding_dim, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='project-head') # (..., G, E) => (..., G, U), typically U = E
        self._softmax = tf.keras.layers.Softmax(axis=-1, name='softmax') # (..., G, U)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._softmax(self._dense(inputs))