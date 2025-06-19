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
        # linear time schedule
        self._time_schedule = functools.partial(mlable.schedules.linear_rate, start_step=0, start_rate=0.0, end_rate=1.0)
        # cosine diffusion schedule
        self._rate_schedule = functools.partial(mlable.schedules.cosine_rates, start_rate=start_rate, end_rate=end_rate)
        # scale the data to a normal distribution and back
        self._latent_mean = tf.cast(0.0, dtype=self.compute_dtype)
        self._latent_std = tf.cast(1.0, dtype=self.compute_dtype)
        # save the data shape for generation
        self._latent_shape = ()

    # WEIGHTS ##################################################################

    def build(self, inputs_shape: tuple) -> None:
        self._latent_shape = tuple(inputs_shape)

    # SHAPES ###################################################################

    def compute_data_shape(self, inputs_shape: tuple=(), batch_dim: int=0) -> tuple:
        __shape = tuple(inputs_shape) or tuple(self._latent_shape)
        __batch_dim = int(batch_dim or __shape[0])
        return (__batch_dim,) + __shape[1:]

    def compute_rate_shape(self, inputs_shape: tuple=(), batch_dim: int=0) -> tuple:
        __shape = BaseDiffusionModel.compute_data_shape(self, inputs_shape=inputs_shape, batch_dim=batch_dim)
        return tuple(mlable.shapes.filter(__shape, axes=[0]))

    # NORMALIZE ################################################################

    def _norm_latent(self, data: tf.Tensor, dtype: tf.DType=None) -> tf.Tensor:
        __dtype = dtype or self.compute_dtype
        __cast = functools.partial(tf.cast, dtype=__dtype)
        return (__cast(data) - __cast(self._latent_mean)) / __cast(self._latent_std)

    def _denorm_latent(self, data: tf.Tensor, dtype: tf.DType=None) -> tf.Tensor:
        __dtype = dtype or self.compute_dtype
        __cast = functools.partial(tf.cast, dtype=__dtype)
        return __cast(self._latent_mean) + __cast(self._latent_std) * __cast(data)

    def adapt_latent(self, dataset: tf.data.Dataset, mean_fn: callable=reduce_mean, std_fn: callable=reduce_std, batch_num: int=2 ** 10, dtype: tf.DType=None) -> None:
        __dtype = dtype or self.compute_dtype
        __cast = functools.partial(tf.cast, dtype=__dtype)
        # process only a subset for speed
        __dataset = dataset.take(batch_num)
        # compute the dataset cardinality
        __scale = __dataset.reduce(0, lambda __c, _: __c + 1)
        __scale = __cast(1.0) / __cast(tf.maximum(1, __scale))
        # compute the mean
        self._latent_mean = __scale * __dataset.reduce(__cast(0.0), mean_fn)
        # compute the standard deviation
        self._latent_std = __scale * __dataset.reduce(__cast(0.0), std_fn)

    # END-TO-END PRE / POST PROCESSING #########################################

    def to_latent(self, data: tf.Tensor, dtype: tf.DType=None, **kwargs) -> tf.Tensor:
        # scale to N(0, I)
        return BaseDiffusionModel._norm_latent(self, data, dtype=dtype)

    def from_latent(self, data: tf.Tensor, **kwargs) -> tf.Tensor:
        # scale back to the signal space
        __data = BaseDiffusionModel._denorm_latent(self, data)
        # enforce types
        return tf.cast(__data, dtype=tf.int32)

    # NOISE ####################################################################

    def ennoise_latent(self, data: tf.Tensor, data_rates: tf.Tensor, noise_rates: tf.Tensor) -> tuple:
        # random values by default
        __noises = tf.random.normal(data.shape, dtype=data.dtype)
        # mix the components
        return (data_rates * data + noise_rates * __noises), __noises

    def denoise_latent(self, data: tf.Tensor, data_rates: tf.Tensor, noise_rates: tf.Tensor, training: bool=False, **kwargs) -> tuple:
        # predict noise component
        __noises = self.call((data, noise_rates), training=training, **kwargs)
        # remove noise component from data
        __data = (data - noise_rates * __noises) / data_rates
        # return both
        return __data, __noises

    # DIFFUSION META ###########################################################

    def diffusion_schedule(self, data_shape: tuple, current_step: int=None, total_step: int=None, dtype: tf.DType=None) -> tuple:
        __dtype = dtype or self.compute_dtype
        # reverse diffusion = sampling
        __shape = BaseDiffusionModel.compute_rate_shape(self, inputs_shape=data_shape)
        # random values for the training process
        __times = tf.random.uniform(shape=__shape, minval=0.0, maxval=1.0, dtype=__dtype)
        # timesteps as a ratio of the diffusion process
        if (current_step is not None) and (total_step is not None):
            # factor by 1 to expand the batch dimension
            __match = tf.ones(__shape, dtype=__dtype)
            # always between 0 and 1
            __times = __match * self._time_schedule(current_step=current_step, end_step=total_step, dtype=__dtype)
        # signal rate, noise rate (never null because of the init bounds)
        return self._rate_schedule(__times, dtype=__dtype)

    # FORWARD DIFFUSION ########################################################

    def forward_step(self, current_data: tf.Tensor, current_step: int, total_step: int) -> tuple:
        # signal rate, noise rate for the current timestep
        __alpha_t, __beta_t = self.diffusion_schedule(current_step=current_step, total_step=total_step, data_shape=current_data.shape, dtype=current_data.dtype)
        __alpha_t1, __beta_t1 = self.diffusion_schedule(current_step=current_step - 1, total_step=total_step, data_shape=current_data.shape, dtype=current_data.dtype)
        # iterative rates
        __alpha = __alpha_t / __alpha_t1
        __beta = tf.sqrt(1.0 - __alpha ** 2)
        # fresh iterative noise e_t
        __noises = tf.random.normal(current_data.shape, dtype=current_data.dtype)
        # mix the components to sample x_t = a_t * x_t-1 + b_t * e_t
        __data = __alpha * current_data + __beta * __noises
        # x_t, e_t
        return __data, __noises

    def forward_diffusion(self, initial_data: tf.Tensor, total_step: int=256, dtype: tf.DType=None) -> tf.Tensor:
        __dtype = dtype or getattr(initial_data, 'dtype', self.compute_dtype)
        __cast = functools.partial(tf.cast, dtype=__dtype)
        # the original data x_0
        __data = __cast(initial_data)
        for __i in range(1, total_step + 1):
            # compute x_t by adding iterative noise
            __data, _ = BaseDiffusionModel.forward_step(self, current_data=__data, current_step=__i, total_step=total_step)
        return __data

    # REVERSE DIFFUSION ########################################################

    def current_rate(self, current_step: int, total_step: int, data_shape: tuple, eta_rate: float=1.0, dtype: tf.DType=None) -> tf.Tensor:
        # signal and noise rates from the previous iteration: a_t+1, b_t+1
        __alpha_t1, __beta_t1 = self.diffusion_schedule(current_step=current_step + 1, total_step=total_step, data_shape=data_shape, dtype=dtype)
        # current signal and noise rates: a_t, b_t
        __alpha_t, __beta_t = self.diffusion_schedule(current_step=current_step, total_step=total_step, data_shape=data_shape, dtype=dtype)
        # the rates (deviations) are the square roots of the variances
        return eta_rate * (__beta_t / __beta_t1) * tf.sqrt(1.0 - (__alpha_t1 / __alpha_t) ** 2)

    def current_estimation(self, initial_data: tf.Tensor, cumulated_noises: tf.Tensor, current_step: int, total_step: int, eta_rate: float=1.0) -> tuple:
        __dtype = initial_data.dtype
        __shape = tuple(initial_data.shape)
        # current signal and noise rates: a_t, b_t
        __alpha_t, __beta_t = self.diffusion_schedule(current_step=current_step, total_step=total_step, data_shape=__shape, dtype=__dtype)
        # standard deviation of x_t
        __std_t = self.current_rate(current_step=current_step, total_step=total_step, eta_rate=eta_rate, data_shape=__shape, dtype=__dtype)
        # mean of x_t, computed from the predictions of x_0 and e_t, the cumulative noise
        __mean_t = (__alpha_t * initial_data) + (tf.sqrt((__beta_t ** 2) - (__std_t ** 2)) * cumulated_noises)
        # fresh noise
        __noises_t = __std_t * tf.random.normal(__shape, dtype=__dtype)
        # estimation of x_t, with fresh noise e_t
        return __mean_t + __noises_t, __noises_t

    def reverse_step(self, current_data: tf.Tensor, current_noises: tf.Tensor, current_step: int, total_step: int, eta_rate: float=1.0, **kwargs) -> tuple:
        # signal rate, noise rate for the current timestep
        __alpha, __beta = self.diffusion_schedule(current_step=current_step, total_step=total_step, data_shape=current_data.shape, dtype=current_data.dtype)
        # estimate x_t from the predicted (cumulated) noise and x_0 estimation
        __data, __noises = BaseDiffusionModel.current_estimation(self, initial_data=current_data, cumulated_noises=current_noises, current_step=current_step, total_step=total_step, eta_rate=eta_rate)
        # predict the cumulated noise and estimate x_0
        return BaseDiffusionModel.denoise_latent(self, data=__data, data_rates=__alpha, noise_rates=__beta, training=False, **kwargs)

    def reverse_diffusion(self, initial_noises: tf.Tensor, total_step: int=256, eta_rate: float=1.0, dtype: tf.DType=None, **kwargs) -> tf.Tensor:
        __dtype = dtype or getattr(initial_noises, 'dtype', self.compute_dtype)
        __cast = functools.partial(tf.cast, dtype=__dtype)
        # the current predictions for the noise and the signal
        __noises = __cast(initial_noises)
        __data = __cast(initial_noises)
        for __i in reversed(range(total_step + 1)):
            # predict the cumulated noise and estimate x_0
            __data, __noises = BaseDiffusionModel.reverse_step(self, current_data=__data, current_noises=__noises, current_step=__i, total_step=total_step, eta_rate=eta_rate, **kwargs)
        return __data

    # SAMPLING #################################################################

    def generate_samples(self, sample_num: int, total_step: int=256, eta_rate: float=1.0, dtype: tf.DType=None, **kwargs) -> tf.Tensor:
        __dtype = dtype or self.compute_dtype
        # adapt the batch dimension
        __shape = BaseDiffusionModel.compute_data_shape(self, batch_dim=sample_num)
        # sample the initial noise
        __noises = tf.random.normal(shape=__shape, dtype=__dtype)
        # remove the noise
        __data = self.reverse_diffusion(__noises, total_step=total_step, eta_rate=eta_rate, **kwargs)
        # denormalize
        return self.from_latent(__data, training=False)

    # TRAINING #################################################################

    def train_step(self, data: tf.Tensor) -> dict:
        __dtype = self.compute_dtype
        # normalize data to have standard deviation of 1, like the noises
        __data = self.to_latent(data, training=True, dtype=__dtype)
        # random rates in the range [end_rate, start_rate] defined on init
        __alpha, __beta = self.diffusion_schedule(current_step=None, total_step=None, data_shape=__data.shape, dtype=__dtype)
        # mix the data with noises
        __data, __noises = self.ennoise_latent(data=__data, data_rates=__alpha, noise_rates=__beta)
        # train to predict the noise from scrambled data
        return super(BaseDiffusionModel, self).train_step(((__data, __beta), __noises))

    def test_step(self, data: tf.Tensor) -> dict:
        __dtype = self.compute_dtype
        # normalize data to have standard deviation of 1, like the noises
        __data = self.to_latent(data, training=False, dtype=__dtype)
        # random rates in the range [end_rate, start_rate] defined on init
        __alpha, __beta = self.diffusion_schedule(current_step=None, total_step=None, data_shape=__data.shape, dtype=__dtype)
        # mix the data with noises
        __data, __noises = self.ennoise_latent(data=__data, data_rates=__alpha, noise_rates=__beta)
        # train to predict the noise from scrambled data
        return super(BaseDiffusionModel, self).test_step(((__data, __beta), __noises))

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

    # VAE ######################################################################

    def get_vae(self) -> tf.keras.Model:
        return self._vae

    def set_vae(self, model: tf.keras.Model) -> None:
        self._vae = model

    # LATENT <=> SIGNAL SPACES #################################################

    def _encode(self, data: tf.Tensor, training: bool=False, sample: bool=True, dtype: tf.DType=None, **kwargs) -> tf.Tensor:
        __dtype = dtype or self.compute_dtype
        __cast = functools.partial(tf.cast, dtype=__dtype)
        __latents = self._vae.encode(data, training=training, **kwargs)
        if not isinstance(__latents, tf.Tensor):
            __latents = self._vae.sample(*__latents, random=sample)
        return __cast(__latents)

    def _decode(self, data: tf.Tensor, training: bool=False, dtype: tf.DType=None, **kwargs) -> tf.Tensor:
        __dtype = dtype or self.compute_dtype
        __cast = functools.partial(tf.cast, dtype=__dtype)
        return __cast(self._vae.decode(data, training=training, **kwargs))

    # PRE / POST PROCESSING ####################################################

    def to_latent(self, data: tf.Tensor, training: bool=False, sample: bool=True, dtype: tf.DType=None, **kwargs) -> tf.Tensor:
        # encode in the latent space
        __data = LatentDiffusionModel._encode(self, data, training=training, sample=sample, dtype=dtype, **kwargs)
        # scale to N(0, I)
        return BaseDiffusionModel._norm_latent(self, __data, dtype=dtype)

    def from_latent(self, data: tf.Tensor, training: bool=False, dtype: tf.DType=None, **kwargs) -> tf.Tensor:
        # scale the pixel values back to the latent space
        __data = BaseDiffusionModel._denorm_latent(self, data, dtype=dtype)
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
