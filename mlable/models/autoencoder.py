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
        self._rate = functools.partial(mlable.schedules.linear_rate, step_min=step_min, step_max=step_max, rate_min=beta_min, rate_max=beta_max)

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
            # track the training step
            self._step.assign_add(1)
            # compute the KL divergence estimate
            __kl = tf.reduce_mean(self.compute_kl(mean=__m, logvar=__v))
            # compute the matching schedule rate
            __rate = tf.cast(self._rate(self._step), dtype=__kl.dtype)
            # register the extra loss term
            self.add_loss(tf.cast(__rate * __kl, dtype=tf.float32))
        # reconstruct the input from the latent encoding
        return self.decode(__z, training=training, **kwargs)

    def compute_kl(self, mean: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
        return -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=-1)

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
