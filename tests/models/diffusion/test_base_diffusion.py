import tensorflow as tf

import mlable.blocks.normalization
import mlable.data
import mlable.maths.ops
import mlable.models.diffusion
import mlable.shapes
import mlable.shaping.axes

# DUMMY ########################################################################

@tf.keras.utils.register_keras_serializable(package='models')
class DummyDiffusionModel(mlable.models.diffusion.BaseDiffusionModel):
    def __init__(self, latent_dim: int=64, start_rate: float=0.95, end_rate: float=0.05, **kwargs) -> None:
        super(DummyDiffusionModel, self).__init__(start_rate=start_rate, end_rate=end_rate, **kwargs)
        self._config.update({'latent_dim': latent_dim,})

    def build(self, inputs_shape: tuple) -> None:
        __shape_o, __shape_c = tuple(tuple(__i) for __i in inputs_shape)
        super(DummyDiffusionModel, self).build(__shape_o)
        self._norm0 = mlable.blocks.normalization.AdaptiveGroupNormalization(groups=8, axis=-1)
        self._proj0 = tf.keras.layers.Dense(units=self._config['latent_dim'])
        self._proj1 = tf.keras.layers.Dense(units=__shape_o[-1])
        self._norm0.build(__shape_o, __shape_c)
        self._proj0.build(__shape_o)
        self._proj1.build(self._proj0.compute_output_shape(__shape_o))
        self.built = True

    def call(self, inputs: tuple, **kwargs) -> tf.Tensor:
        __outputs, __contexts = inputs
        __outputs = self._norm0(__outputs, __contexts)
        return self._proj1(self._proj0(__outputs))

# PREPROCESS ###################################################################

def preprocess_bytes(inputs: tf.Tensor, factor: int=4) -> tuple:
    __inputs = mlable.shaping.axes.divide(inputs, axis=-1, factor=factor, insert=True, right=True)
    return tf.cast(__inputs, tf.float32)

# BASE #########################################################################

class BaseDiffusionModelTest(tf.test.TestCase):
    def setUp(self):
        super(BaseDiffusionModelTest, self).setUp()
        # dataset of random bytes
        self._dataset = mlable.data.random_dataset_of_bytes(sample_count=4 * 256, sample_size= 4 * 64)
        self._dataset = self._dataset.map(preprocess_bytes).batch(batch_size=4)
        # input data in the latent space
        self._cases = [
            {
                'inputs': tf.random.normal((2, 16, 16, 8), dtype=tf.float16),
                'contexts': tf.random.normal((2, 4), dtype=tf.float16),},
            {
                'inputs': tf.random.normal((2, 16, 16, 8), dtype=tf.float32),
                'contexts': tf.random.normal((2, 1), dtype=tf.float32),},
            {
                'inputs': tf.random.normal((2, 16, 16, 8), dtype=tf.float32),
                'contexts': tf.random.normal((2, 1, 1, 8), dtype=tf.float32),},
            # {
            #     'inputs': tf.random.normal((2, 16, 16, 8), dtype=tf.float32),
            #     'contexts': None,},
            {
                'inputs': tf.random.normal((2, 16, 16, 8), dtype=tf.float32),
                'contexts': tf.random.normal((1, 8), dtype=tf.float32),},
            {
                'inputs': tf.random.normal((2, 16, 16, 8), dtype=tf.float32),
                'contexts': tf.random.normal((1, 1), dtype=tf.float32),},]

    # SHAPES ###################################################################

    def test_inputs_and_outputs_shapes_are_equal(self):
        for __case in self._cases:
            __model = DummyDiffusionModel(latent_dim=32)
            __outputs = __model((__case['inputs'], __case['contexts']), training=False)
            self.assertEqual(tuple(__outputs.shape), tuple(__case['inputs'].shape))
            self.assertEqual(tuple(__outputs.shape), __model.compute_data_shape(__case['inputs'].shape))

    def test_model_saves_the_data_and_rate_shapes(self):
        for __case in self._cases:
            __model = DummyDiffusionModel(latent_dim=32)
            __outputs = __model((__case['inputs'], __case['contexts']), training=False)
            # as saved
            self.assertEqual(__model.compute_data_shape(), tuple(__case['inputs'].shape))
            self.assertEqual(__model.compute_rate_shape(), tuple(mlable.shapes.filter(__case['inputs'].shape, axes=[0])))
            # change the batch dimensions
            self.assertEqual(__model.compute_data_shape(batch_dim=5), (5,) + tuple(__case['inputs'].shape[1:]))
            self.assertEqual(__model.compute_rate_shape(batch_dim=5), (5,) + (len(__case['inputs'].shape) - 1) * (1,))

    def test_normalization_doesnt_change_shapes(self):
        for __case in self._cases:
            __model = DummyDiffusionModel(latent_dim=32)
            __outputs = __model._norm(__case['inputs'])
            self.assertEqual(tuple(__outputs.shape), tuple(__case['inputs'].shape))
            __outputs = __model._denorm(__outputs)
            self.assertEqual(tuple(__outputs.shape), tuple(__case['inputs'].shape))

    # NORMALIZATION ############################################################

    def test_normalization_has_no_effect_on_model_init(self):
        for __case in self._cases:
            __model = DummyDiffusionModel(latent_dim=32)
            __outputs = __model._norm(__case['inputs'])
            self.assertAllEqual(__outputs, __case['inputs'])
            __outputs = __model._denorm(__outputs)
            self.assertAllEqual(__outputs, __case['inputs'])

    def test_normalization_can_be_reversed(self):
        for __case in self._cases:
            __model = DummyDiffusionModel(latent_dim=32)
            # change the defaults
            __model._mean = tf.cast(-1.0, tf.float32)
            __model._std = tf.cast(4.0, tf.float32)
            # back and forth
            __outputs = __model._norm(__case['inputs'])
            __outputs = __model._denorm(__outputs)
            # check
            self.assertAllClose(__outputs, __case['inputs'])

    def test_dataset_adaptation_on_random_data(self):
        __model = DummyDiffusionModel(latent_dim=32)
        __model.adapt(dataset=self._dataset, batch_num=256)
        __dataset = self._dataset.map(__model._norm)
        __dim = int(__dataset.element_spec.shape[-1])
        __mean = __dataset.reduce(tf.cast(0.0, tf.float32), mlable.models.diffusion.reduce_mean)
        __std = __dataset.reduce(tf.cast(0.0, tf.float32), mlable.models.diffusion.reduce_std)
        # tests moments of the original dataset
        self.assertEqual(tf.reduce_prod(__model._mean.shape), __dim)
        self.assertEqual(tf.reduce_prod(__model._std.shape), __dim)
        # the dataset is made of random bytes => mean = 255 / 2
        self.assertAllClose(__model._mean, 0.5 * 255.0 * tf.ones(__model._mean.shape), rtol=1e-1)
        self.assertAllClose(__model._std, 73.9 * tf.ones(__model._mean.shape), rtol=1e-1) #  std = tf.sqrt((255.0 * 256.0 * 511.0 / (256.0 * 6.0)) - (127.5 ** 2))
        # test moments of the normalized dataset
        self.assertAllClose(__mean / 256.0, 0.0 * tf.ones(__mean.shape), atol=1e-1)
        self.assertAllClose(__std / 256.0, 1.0 * tf.ones(__std.shape), rtol=1e-1)
