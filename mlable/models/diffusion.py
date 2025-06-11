import functools

import tensorflow as tf

# import mlable.models
import mlable.shapes
import mlable.shaping.axes

import mlable.schedules

# CONSTANTS ####################################################################

START_RATE = 0.95 # signal rate at the start of the forward diffusion process
END_RATE = 0.02 # signal rate at the start of the forward diffusion process

# UTILITIES ####################################################################

def reduce_mean(current: tf.Tensor, sample: tf.Tensor) -> tf.Tensor:
    return current + tf.cast(tf.math.reduce_mean(
            sample,
            axis=tf.range(tf.rank(sample) - 1),
            keepdims=True),
        dtype=current.dtype)

def reduce_std(current: tf.Tensor, sample: tf.Tensor) -> tf.Tensor:
    return current + tf.cast(tf.math.reduce_std(
            sample,
            axis=tf.range(tf.rank(sample) - 1),
            keepdims=True),
        dtype=current.dtype)

# NORMALIZED DIFFUSION #########################################################

@tf.keras.utils.register_keras_serializable(package='models')
class BaseDiffusionModel(tf.keras.models.Model): # mlable.models.ContrastModel
    def __init__(
        self,
        start_rate: float=START_RATE, # signal rate at the start of the forward diffusion process
        end_rate: float=END_RATE, # signal rate at the start of the forward diffusion process
        **kwargs
    ) -> None:
        # init
        super(BaseDiffusionModel, self).__init__(**kwargs)
        # save config for IO
        self._config = {'start_rate': start_rate, 'end_rate': end_rate,}
        # diffusion schedule
        self._schedule = functools.partial(mlable.schedules.cosine_rates, start_rate=start_rate, end_rate=end_rate)
        # scale the data to a normal distribution and back
        self._mean = tf.cast(0.0, dtype=self.compute_dtype)
        self._std = tf.cast(1.0, dtype=self.compute_dtype)
        # save the data shape for generation
        self._shape = ()

    # WEIGHTS ##################################################################

    def build(self, inputs_shape: tuple) -> None:
        self._shape = tuple(inputs_shape)

    # SHAPES ###################################################################

    def compute_data_shape(self, inputs_shape: tuple=(), batch_dim: int=0) -> tuple:
        __shape = tuple(inputs_shape) or tuple(self._shape)
        __batch_dim = int(batch_dim or __shape[0])
        return (__batch_dim,) + __shape[1:]

    def compute_rate_shape(self, inputs_shape: tuple=(), batch_dim: int=0) -> tuple:
        __shape = BaseDiffusionModel.compute_data_shape(self, inputs_shape=inputs_shape, batch_dim=batch_dim)
        return tuple(mlable.shapes.filter(__shape, axes=[0]))

    # NORMALIZE ################################################################

    def _norm(self, data: tf.Tensor, dtype: tf.DType=None) -> tf.Tensor:
        __dtype = dtype or self.compute_dtype
        __cast = functools.partial(tf.cast, dtype=__dtype)
        return (__cast(data) - __cast(self._mean)) / __cast(self._std)

    def _denorm(self, data: tf.Tensor, dtype: tf.DType=None) -> tf.Tensor:
        __dtype = dtype or self.compute_dtype
        __cast = functools.partial(tf.cast, dtype=__dtype)
        return __cast(self._mean) + __cast(self._std) * __cast(data)

    def adapt(self, dataset: tf.data.Dataset, mean_fn: callable=reduce_mean, std_fn: callable=reduce_std, batch_num: int=2 ** 10, dtype: tf.DType=None) -> None:
        __dtype = dtype or self.compute_dtype
        __cast = functools.partial(tf.cast, dtype=__dtype)
        # process only a subset for speed
        __dataset = dataset.take(batch_num)
        # compute the dataset cardinality
        __scale = __dataset.reduce(0, lambda __c, _: __c + 1)
        __scale = __cast(1.0) / __cast(tf.maximum(1, __scale))
        # compute the mean
        self._mean = __scale * __dataset.reduce(__cast(0.0), mean_fn)
        # compute the standard deviation
        self._std = __scale * __dataset.reduce(__cast(0.0), std_fn)

    # END-TO-END PRE / POST PROCESSING #########################################

    def preprocess(self, data: tf.Tensor, dtype: tf.DType=None, **kwargs) -> tf.Tensor:
        # scale to N(0, I)
        return BaseDiffusionModel._norm(self, data, dtype=dtype)

    def postprocess(self, data: tf.Tensor, **kwargs) -> tf.Tensor:
        # scale back to the signal space
        __data = BaseDiffusionModel._denorm(self, data)
        # enforce types
        return tf.cast(__data, dtype=tf.int32)

    # NOISE ####################################################################

    def denoise(self, noisy_data: tf.Tensor, noise_rates: tf.Tensor, data_rates: tf.Tensor, **kwargs) -> tuple:
        # predict noise component
        __noises = self.call((noisy_data, noise_rates), training=False, **kwargs)
        # remove noise component from data
        __data = (noisy_data - noise_rates * __noises) / data_rates
        # return both
        return __noises, __data

    # DIFFUSION ################################################################

    def reverse_diffusion(self, initial_noises: tf.Tensor, step_num: int, dtype: tf.DType=None, **kwargs) -> tf.Tensor:
        __dtype = dtype or self.compute_dtype
        __cast = functools.partial(tf.cast, dtype=__dtype)
        # reverse diffusion = sampling
        __shape = BaseDiffusionModel.compute_rate_shape(self, inputs_shape=initial_noises.shape)
        __delta = __cast(1.0 / step_num)
        # the current predictions for the noise and the signal
        __noises = __cast(initial_noises)
        __data = __cast(initial_noises)
        for __i in range(step_num + 1):
            # even pure noise (step 0) is considered to contain some signal
            __angles = tf.ones(__shape, dtype=__dtype) - __cast(__i) * __delta
            __alpha, __beta = self._schedule(__angles, dtype=__dtype)
            # remix the components, with a noise level corresponding to the current iteration
            __data = (__beta * __data + __alpha * __noises)
            # predict the cumulated noise in the sample, and remove it from the sample
            __noises, __data = self.denoise(__data, __alpha, __beta, **kwargs)
        return __data

    # SAMPLING #################################################################

    def generate(self, sample_num: int, step_num: int, dtype: tf.DType=None, **kwargs) -> tf.Tensor:
        __dtype = dtype or self.compute_dtype
        # adapt the batch dimension
        __shape = BaseDiffusionModel.compute_data_shape(self, batch_dim=sample_num)
        # sample the initial noise
        __noises = tf.random.normal(shape=__shape, dtype=__dtype)
        # remove the noise
        __data = self.reverse_diffusion(__noises, step_num=step_num, **kwargs)
        # denormalize
        return self.postprocess(__data, training=False)

    # TRAINING #################################################################

    def train_step(self, data: tf.Tensor) -> dict:
        __dtype = self.compute_dtype
        # normalize data to have standard deviation of 1, like the noises
        __data = self.preprocess(data, training=True, dtype=__dtype)
        # compute the shapes in the latent space
        __shape_n = BaseDiffusionModel.compute_data_shape(self, inputs_shape=__data.shape)
        __shape_a = BaseDiffusionModel.compute_rate_shape(self, inputs_shape=__data.shape)
        # sample the noises = targets
        __noises = tf.random.normal(shape=__shape_n, dtype=__dtype)
        # sample the diffusion angles
        __angles = tf.random.uniform(shape=__shape_a, minval=0.0, maxval=1.0, dtype=__dtype)
        # compute the signal to noise ratio
        __noise_rates, __data_rates = self._schedule(__angles, dtype=__dtype)
        # mix the data with noises
        __data = __data_rates * __data + __noise_rates * __noises
        # train to predict the noise from scrambled data
        return super(BaseDiffusionModel, self).train_step(((__data, __noise_rates), __noises))

    def test_step(self, data: tf.Tensor) -> dict:
        __dtype = self.compute_dtype
        # normalize data to have standard deviation of 1, like the noises
        __data = self.preprocess(data, training=False, dtype=__dtype)
        # compute the shapes in the latent space
        __shape_n = BaseDiffusionModel.compute_data_shape(self, inputs_shape=__data.shape)
        __shape_a = BaseDiffusionModel.compute_rate_shape(self, inputs_shape=__data.shape)
        # sample the noises = targets
        __noises = tf.random.normal(shape=__shape_n, dtype=__dtype)
        # sample the diffusion angles
        __angles = tf.random.uniform(shape=__shape_a, minval=0.0, maxval=1.0, dtype=__dtype)
        # compute the signal to noise ratio
        __noise_rates, __data_rates = self._schedule(__angles, dtype=__dtype)
        # mix the data with noises
        __data = __data_rates * __data + __noise_rates * __noises
        # train to predict the noise from scrambled data
        return super(BaseDiffusionModel, self).test_step(((__data, __noise_rates), __noises))

    # CONFIG ###################################################################

    def get_config(self) -> dict:
        __config = super(BaseDiffusionModel, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# LATENT DIFFUSION #############################################################

@tf.keras.utils.register_keras_serializable(package='models')
class LatentDiffusionModel(BaseDiffusionModel): # mlable.models.ContrastModel
    def __init__(
        self,
        start_rate: float=START_RATE, # signal rate at the start of the forward diffusion process
        end_rate: float=END_RATE, # signal rate at the start of the forward diffusion process
        **kwargs
    ) -> None:
        super(LatentDiffusionModel, self).__init__(start_rate=start_rate, end_rate=end_rate, **kwargs)
        # encoding / decoding model
        self._vae = None

    # LATENT <=> SIGNAL SPACES #################################################

    def _encode(self, data: tf.Tensor, training: bool=False, dtype: tf.DType=None, **kwargs) -> tf.Tensor:
        __dtype = dtype or self.compute_dtype
        __cast = functools.partial(tf.cast, dtype=__dtype)
        __latents = self._vae.encode(data, training=training, **kwargs)
        __latents = __latents if isinstance(__latents, tf.Tensor) else self._vae.sample(*__latents)
        return __cast(__latents)

    def _decode(self, data: tf.Tensor, training: bool=False, dtype: tf.DType=None, **kwargs) -> tf.Tensor:
        __dtype = dtype or self.compute_dtype
        __cast = functools.partial(tf.cast, dtype=__dtype)
        return __cast(self._vae.decode(data, training=training, **kwargs))

    def get_vae(self) -> tf.keras.Model:
        return self._vae

    def set_vae(self, model: tf.keras.Model, trainable: bool=False) -> None:
        self._vae = model
        self._vae.trainable = trainable

    # PRE / POST PROCESSING ####################################################

    def preprocess(self, data: tf.Tensor, training: bool=False, dtype: tf.DType=None, **kwargs) -> tf.Tensor:
        # encode in the latent space
        __data = LatentDiffusionModel._encode(self, data, training=training, dtype=dtype, **kwargs)
        # scale to N(0, I)
        return BaseDiffusionModel._norm(self, __data, dtype=dtype)

    def postprocess(self, data: tf.Tensor, training: bool=False, dtype: tf.DType=None, **kwargs) -> tf.Tensor:
        # scale the pixel values back to the latent space
        __data = BaseDiffusionModel._denorm(self, data, dtype=dtype)
        # decode back to the signal space
        return LatentDiffusionModel._decode(self, __data, training=training, dtype=dtype, **kwargs)

    # CONFIG ###################################################################

    def get_config(self) -> dict:
        __config = super(LatentDiffusionModel, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)
