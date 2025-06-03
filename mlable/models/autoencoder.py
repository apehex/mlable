import functools

import tensorflow as tf

import mlable.schedules

# VAE ##########################################################################

@tf.keras.utils.register_keras_serializable(package='models')
class VaeModel(tf.keras.Model):
    def __init__(self, step_min: int=0, step_max: int=2 ** 12, beta_min: float=0.0, beta_max: float=1.0, **kwargs):
        super(VaeModel, self).__init__(**kwargs)
        # save the config
        self._config = {'step_min': step_min, 'step_max': step_max, 'beta_min': beta_min, 'beta_max': beta_max,}
        # track the training step
        self._step = tf.Variable(-1, trainable=False, dtype=tf.int32)
        # set the KL loss factor accordingly
        self._rate = functools.partial(mlable.schedules.linear_rate, start_step=step_min, end_step=step_max, start_rate=beta_min, end_rate=beta_max)

    def sample(self, mean: tf.Tensor, logvar: tf.Tensor, dtype: tf.DType=None) -> tf.Tensor:
        __dtype = self.compute_dtype if (dtype is None) else dtype
        __mean = tf.cast(mean, dtype=__dtype)
        __std = tf.cast(tf.exp(logvar * 0.5), dtype=__dtype)
        return tf.random.normal(shape=tf.shape(__mean), mean=__mean, stddev=__std, dtype=__dtype)

    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        # encode the input into the latent space
        __m, __v = self.encode(inputs, training=training, **kwargs)
        # sample from the latent space, according to the prior distribution
        __z = self.sample(__m, __v)
        # KL divergence between the current latent distribution and the normal
        if training:
            __cast = functools.partial(tf.cast, dtype=tf.float32)
            # track the training step
            self._step.assign_add(1)
            # compute the KL divergence estimate
            __kl = tf.reduce_mean(self.compute_kl(mean=__m, logvar=__v))
            # compute the matching schedule rate
            __rate = __cast(self._rate(self._step))
            # register the extra loss term
            self.add_loss(__cast(__rate * __kl))
        # reconstruct the input from the latent encoding
        return self.decode(__z, training=training, **kwargs)

    def compute_kl(self, mean: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
        __cast = functools.partial(tf.cast, dtype=tf.float32)
        __mean, __logvar = __cast(mean), __cast(logvar)
        return __cast(0.5) * tf.reduce_sum(tf.square(__mean) + tf.exp(__logvar) - __cast(1.0) - __logvar, axis=-1)

    # def train_step(self, data: tf.Tensor) -> dict:
    #     return super(VaeModel, self).train_step((data, data))

    # def test_step(self, data: tf.Tensor) -> dict:
    #     return super(VaeModel, self).test_step((data, data))

    def get_config(self) -> dict:
        __config = super(VaeModel, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)
