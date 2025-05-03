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

def reduce_mean(data: tf.Tensor) -> tf.Tensor:
    return tf.math.reduce_mean(
        data,
        axis=tf.range(tf.rank(data) - 1),
        keepdims=True)

def reduce_std(data: tf.Tensor) -> tf.Tensor:
    return tf.math.reduce_std(
        data,
        axis=tf.range(tf.rank(data) - 1),
        keepdims=True)

# NORMALIZED DIFFUSION #########################################################

@tf.keras.utils.register_keras_serializable(package='models')
class NormalizedDiffusionModel(tf.keras.models.Model): # mlable.models.ContrastModel
    def __init__(
        self,
        start_rate: float=START_RATE, # signal rate at the start of the forward diffusion process
        end_rate: float=END_RATE, # signal rate at the start of the forward diffusion process
        **kwargs
    ) -> None:
        # init
        super(NormalizedDiffusionModel, self).__init__(**kwargs)
        # save config for IO
        self._config = {'start_rate': start_rate, 'end_rate': end_rate,}
        # diffusion schedule
        self._schedule = functools.partial(mlable.schedules.cosine_rates, start_rate=start_rate, end_rate=end_rate)
        # scale the data to a normal distribution and back
        self._mean = tf.cast(0.0, dtype=self.compute_dtype)
        self._std = tf.cast(1.0, dtype=self.compute_dtype)
        # save the data shape for generation
        self._shape = ()

    def _norm(self, data: tf.Tensor, dtype: tf.DType=None) -> tf.Tensor:
        __dtype = dtype or self.compute_dtype
        __cast = functools.partial(tf.cast, dtype=__dtype)
        return (__cast(data) - __cast(self._mean)) / __cast(self._std)

    def _denorm(self, data: tf.Tensor, dtype: tf.DType=None) -> tf.Tensor:
        __dtype = dtype or self.compute_dtype
        __cast = functools.partial(tf.cast, dtype=__dtype)
        return __cast(self._mean) + __cast(self._std) * __cast(data)

    def adapt(self, dataset: tf.data.Dataset, dtype: tf.DType=None) -> None:
        __dtype = dtype or self.compute_dtype
        __cast = functools.partial(tf.cast, dtype=__dtype)
        # compute the dataset cardinality
        __scale = dataset.reduce(0, lambda __c, _: __c + 1)
        __scale = __cast(1.0) / __cast(tf.maximum(1, __scale))
        # compute the mean
        self._mean = __scale * dataset.reduce(__cast(0.0), lambda __m, __x: __m + __cast(reduce_mean(__x)))
        self._mean = __cast(self._mean)
        # compute the standard deviation
        self._std = __scale * dataset.reduce(__cast(0.0), lambda __m, __x: __m + __cast(reduce_std(__x)))
        self._std = __cast(self._std)

    def build(self, input_shape: tuple) -> None:
        self._shape = tuple(input_shape)

    def postprocess(self, data: tf.Tensor) -> tf.Tensor:
        # scale the pixel values back to the signal space
        __data = self._denorm(data)
        # enforce types
        return tf.cast(__data, dtype=tf.int32)

    def denoise(self, noisy_data: tf.Tensor, noise_rates: tf.Tensor, signal_rates: tf.Tensor) -> tuple:
        # predict noise component
        __noises = self.call((noisy_data, noise_rates**2), training=False)
        # remove noise component from data
        __data = (noisy_data - noise_rates * __noises) / signal_rates
        # return both
        return __noises, __data

    def reverse_diffusion(self, initial_noises: tf.Tensor, step_num: int, dtype: tf.DType=None) -> tf.Tensor:
        __dtype = dtype or self.compute_dtype
        __cast = functools.partial(tf.cast, dtype=__dtype)
        # reverse diffusion = sampling
        __shape = (int(initial_noises.shape[0]),) + (len(self._shape) - 1) * (1,)
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
            __noises, __data = self.denoise(__data, __alpha, __beta)
        return __data

    def generate(self, sample_num: int, step_num: int, dtype: tf.DType=None) -> tf.Tensor:
        __dtype = dtype or self.compute_dtype
        __cast = functools.partial(tf.cast, dtype=__dtype)
        # adapt the batch dimension
        __shape = (sample_num,) + tuple(self._shape)[1:]
        # sample the initial noise
        __noises = tf.random.normal(shape=__shape, dtype=__dtype)
        # remove the noise
        __data = self.reverse_diffusion(__noises, step_num=step_num)
        # denormalize
        return self.postprocess(__data)

    def train_step(self, data: tf.Tensor) -> dict:
        __dtype = self.compute_dtype
        __shape_n = tuple(data.shape)
        __shape_a = mlable.shapes.filter(data.shape, axes=[0])
        # normalize data to have standard deviation of 1, like the noises
        __data = self._norm(data, dtype=__dtype)
        # sample the noises = targets
        __noises = tf.random.normal(shape=__shape_n, dtype=__dtype)
        # sample the diffusion angles
        __angles = tf.random.uniform(shape=__shape_a, minval=0.0, maxval=1.0, dtype=__dtype)
        # compute the signal to noise ratio
        __noise_rates, __signal_rates = self._schedule(__angles, dtype=__dtype)
        # mix the data with noises
        __data = __signal_rates * __data + __noise_rates * __noises
        # train to predict the noise from scrambled data
        return super(NormalizedDiffusionModel, self).train_step(((__data, __noise_rates**2), __noises))

    def test_step(self, data: tf.Tensor) -> dict:
        __dtype = self.compute_dtype
        __shape_n = tuple(data.shape)
        __shape_a = mlable.shapes.filter(data.shape, axes=[0])
        # normalize data to have standard deviation of 1, like the noises
        __data = self._norm(data, dtype=__dtype)
        # sample the noises = targets
        __noises = tf.random.normal(shape=__shape_n, dtype=__dtype)
        # sample the diffusion angles
        __angles = tf.random.uniform(shape=__shape_a, minval=0.0, maxval=1.0, dtype=__dtype)
        # compute the signal to noise ratio
        __noise_rates, __signal_rates = self._schedule(__angles, dtype=__dtype)
        # mix the data with noises
        __data = __signal_rates * __data + __noise_rates * __noises
        # train to predict the noise from scrambled data
        return super(NormalizedDiffusionModel, self).test_step(((__data, __noise_rates**2), __noises))

    def get_config(self) -> dict:
        __config = super(NormalizedDiffusionModel, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)

# LATENT DIFFUSION #############################################################

@tf.keras.utils.register_keras_serializable(package='models')
class LatentDiffusionModel(tf.keras.models.Model): # mlable.models.ContrastModel
    def __init__(
        self,
        start_rate: float=START_RATE, # signal rate at the start of the forward diffusion process
        end_rate: float=END_RATE, # signal rate at the start of the forward diffusion process
        **kwargs
    ) -> None:
        # init
        super(LatentDiffusionModel, self).__init__(**kwargs)
        # save config for IO
        self._config = {'start_rate': start_rate, 'end_rate': end_rate,}
        # diffusion schedule
        self._schedule = functools.partial(mlable.schedules.cosine_rates, start_rate=start_rate, end_rate=end_rate)
        # scale the data to a normal distribution and back
        self._mean = tf.cast(0.0, dtype=self.compute_dtype)
        self._std = tf.cast(1.0, dtype=self.compute_dtype)
        # save the data shape for generation
        self._shape = ()
        # encoding / decoding model
        self._vae = None

    def build(self, input_shape: tuple) -> None:
        self._shape = tuple(input_shape)

    # SHAPES ###################################################################

    def compute_output_shape(self, input_shape: tuple=(), batch_dim: int=0) -> tuple:
        __shape = tuple(input_shape) or tuple(self._shape)
        __batch_dim = int(batch_dim or __shape[0])
        return (__batch_dim,) + __shape[1:]

    def compute_variance_shape(self, input_shape: tuple=(), batch_dim: int=0) -> tuple:
        __shape = self.compute_output_shape(input_shape=input_shape, batch_dim=batch_dim)
        return tuple(mlable.shapes.filter(__shape, axes=[0]))

    # LATENT <=> SIGNAL SPACES #################################################

    def _encode(self, data: tf.Tensor, dtype: tf.DType=None) -> tf.Tensor:
        __dtype = dtype or self.compute_dtype
        __cast = functools.partial(tf.cast, dtype=__dtype)
        return __cast(self._vae.encode(data, training=False))

    def _decode(self, data: tf.Tensor, logits: bool=True, dtype: tf.DType=None) -> tf.Tensor:
        __dtype = dtype or self.compute_dtype
        __cast = functools.partial(tf.cast, dtype=__dtype)
        return __cast(self._vae.decode(data, logits=logits, training=False))

    def get_vae(self) -> tf.keras.Model:
        return self._vae

    def set_vae(self, model: tf.keras.Model, trainable: bool=False) -> None:
        self._vae = model
        self._vae.trainable = trainable

    # NORMALIZE ################################################################

    def _norm(self, data: tf.Tensor, dtype: tf.DType=None) -> tf.Tensor:
        __dtype = dtype or self.compute_dtype
        __cast = functools.partial(tf.cast, dtype=__dtype)
        return (__cast(data) - __cast(self._mean)) / __cast(self._std)

    def _denorm(self, data: tf.Tensor, dtype: tf.DType=None) -> tf.Tensor:
        __dtype = dtype or self.compute_dtype
        __cast = functools.partial(tf.cast, dtype=__dtype)
        return __cast(self._mean) + __cast(self._std) * __cast(data)

    def adapt(self, dataset: tf.data.Dataset, dtype: tf.DType=None) -> None:
        __dtype = dtype or self.compute_dtype
        __cast = functools.partial(tf.cast, dtype=__dtype)
        # compute the dataset cardinality
        __scale = dataset.reduce(0, lambda __c, _: __c + 1)
        __scale = __cast(1.0) / __cast(tf.maximum(1, __scale))
        # compute the mean, in the latent space
        self._mean = __scale * dataset.reduce(__cast(0.0), lambda __m, __x: __m + __cast(reduce_mean(self._encode(__x))))
        self._mean = __cast(self._mean)
        # compute the standard deviation, in the latent space
        self._std = __scale * dataset.reduce(__cast(0.0), lambda __m, __x: __m + __cast(reduce_std(self._encode(__x))))
        self._std = __cast(self._std)

    # PRE / POST PROCESSING ####################################################

    def preprocess(self, data: tf.Tensor, dtype: tf.DType=None) -> tf.Tensor:
        # encode in the latent space
        __data = self._encode(data, dtype=dtype)
        # scale to N(0, I)
        return self._norm(__data, dtype=dtype)

    def postprocess(self, data: tf.Tensor, logits: bool=True, dtype: tf.DType=None) -> tf.Tensor:
        # scale the pixel values back to the latent space
        __data = self._denorm(data, dtype=dtype)
        # decode back to the signal space
        return self._decode(__data, logits=logits, dtype=dtype)

    # DENOISING ################################################################

    def denoise(self, noisy_data: tf.Tensor, noise_rates: tf.Tensor, signal_rates: tf.Tensor) -> tuple:
        # predict noise component
        __noises = self.call((noisy_data, noise_rates**2), training=False)
        # remove noise component from data
        __data = (noisy_data - noise_rates * __noises) / signal_rates
        # return both
        return __noises, __data

    def reverse_diffusion(self, initial_noises: tf.Tensor, step_num: int, dtype: tf.DType=None) -> tf.Tensor:
        __dtype = dtype or self.compute_dtype
        __cast = functools.partial(tf.cast, dtype=__dtype)
        # reverse diffusion = sampling
        __shape = self.compute_variance_shape(input_shape=initial_noises.shape)
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
            __noises, __data = self.denoise(__data, __alpha, __beta)
        return __data

    # SAMPLING #################################################################

    def generate(self, sample_num: int, step_num: int, logits: bool=True, dtype: tf.DType=None) -> tf.Tensor:
        __dtype = dtype or self.compute_dtype
        __cast = functools.partial(tf.cast, dtype=__dtype)
        # adapt the batch dimension
        __shape = self.compute_output_shape(batch_dim=sample_num)
        # sample the initial noise
        __noises = tf.random.normal(shape=__shape, dtype=__dtype)
        # remove the noise
        __data = self.reverse_diffusion(__noises, step_num=step_num)
        # denormalize and decode
        return self.postprocess(__data, logits=logits, dtype=__dtype)

    # TRAINING #################################################################

    def train_step(self, data: tf.Tensor) -> dict:
        __dtype = self.compute_dtype
        # normalize data to have standard deviation of 1, like the noises
        __data = self.preprocess(data, dtype=__dtype)
        # compute the shapes in the latent space
        __shape_n = self.compute_output_shape(input_shape=__data.shape)
        __shape_a = self.compute_variance_shape(input_shape=__data.shape)
        # sample the noises = targets
        __noises = tf.random.normal(shape=__shape_n, dtype=__dtype)
        # sample the diffusion angles
        __angles = tf.random.uniform(shape=__shape_a, minval=0.0, maxval=1.0, dtype=__dtype)
        # compute the signal to noise ratio
        __noise_rates, __signal_rates = self._schedule(__angles, dtype=__dtype)
        # mix the data with noises
        __data = __signal_rates * __data + __noise_rates * __noises
        # train to predict the noise from scrambled data
        return super(LatentDiffusionModel, self).train_step(((__data, __noise_rates**2), __noises))

    def test_step(self, data: tf.Tensor) -> dict:
        __dtype = self.compute_dtype
        # normalize data to have standard deviation of 1, like the noises
        __data = self.preprocess(data, dtype=__dtype)
        # compute the shapes in the latent space
        __shape_n = self.compute_output_shape(input_shape=__data.shape)
        __shape_a = self.compute_variance_shape(input_shape=__data.shape)
        # sample the noises = targets
        __noises = tf.random.normal(shape=__shape_n, dtype=__dtype)
        # sample the diffusion angles
        __angles = tf.random.uniform(shape=__shape_a, minval=0.0, maxval=1.0, dtype=__dtype)
        # compute the signal to noise ratio
        __noise_rates, __signal_rates = self._schedule(__angles, dtype=__dtype)
        # mix the data with noises
        __data = __signal_rates * __data + __noise_rates * __noises
        # train to predict the noise from scrambled data
        return super(LatentDiffusionModel, self).test_step(((__data, __noise_rates**2), __noises))

    # CONFIG ###################################################################

    def get_config(self) -> dict:
        __config = super(LatentDiffusionModel, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config: dict) -> tf.keras.layers.Layer:
        return cls(**config)
