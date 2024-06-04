import tensorflow as tf

# INITIALIZER #################################################################

class SmallNormal(tf.keras.initializers.Initializer):
    def __init__(self, mean: float=0., stddev: float=0.1):
        self._config = {
            'mean': mean,
            'stddev': stddev,}

    def __call__(self, shape, dtype=None, **kwargs):
        return tf.random.normal(shape, mean=self._config['mean'], stddev=self._config['stddev'], dtype=dtype)

    def get_config(self) -> dict:
        __config = super(SmallNormal, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
