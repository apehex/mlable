import tensorflow as tf

import mlable.blocks.normalization
import mlable.shapes

# BASE #########################################################################

class AdaptiveGroupNormalizationTest(tf.test.TestCase):
    def setUp(self):
        super(AdaptiveGroupNormalizationTest, self).setUp()
        self._cases = [
            {
                'inputs': tf.random.normal((2, 16, 16, 8), dtype=tf.float16),
                'contexts': tf.random.normal((2, 4), dtype=tf.float16),
                'args': {'groups': 4, 'axis':-1},},
            {
                'inputs': tf.random.normal((2, 16, 16, 8), dtype=tf.float32),
                'contexts': tf.random.normal((2, 1), dtype=tf.float32),
                'args': {'groups': 4, 'axis':-1},},
            {
                'inputs': tf.random.normal((2, 16, 16, 8), dtype=tf.float32),
                'contexts': tf.random.normal((2, 1, 1, 8), dtype=tf.float32),
                'args': {'groups': 8, 'axis':-1},},
            {
                'inputs': tf.random.normal((2, 16, 16, 8), dtype=tf.float32),
                'contexts': None,
                'args': {'groups': 4, 'axis':-1},},]

    def test_shape(self):
        for __case in self._cases:
            __layer = mlable.blocks.normalization.AdaptiveGroupNormalization(**__case['args'])
            __outputs = __layer(__case['inputs'], contexts=__case['contexts'], training=False)
            self.assertEqual(tuple(__outputs.shape), tuple(__case['inputs'].shape))
            self.assertEqual(tuple(__outputs.shape), __layer.compute_output_shape(__case['inputs'].shape))

    def test_invariant_on_layer_init(self):
        for __case in self._cases:
            # init with null projection weights
            __layer = mlable.blocks.normalization.AdaptiveGroupNormalization(**__case['args'])
            # so the block shoud be equivalent to a regular group norm
            __norm = tf.keras.layers.GroupNormalization(**__case['args'])
            # process
            __outputs_ada = __layer(__case['inputs'], contexts=__case['contexts'], training=False)
            __outputs_reg = __norm(__case['inputs'], training=False)
            # check
            self.assertAllEqual(__outputs_ada, __outputs_reg)

    def test_invariant_when_context_is_null(self):
        for __case in self._cases:
            __dtype = __case['inputs'].dtype
            __shape = tuple(__case['inputs'].shape)
            __dim = int(__shape[-1])
            # init with null projection weights
            __layer = mlable.blocks.normalization.AdaptiveGroupNormalization(**__case['args'])
            __layer.build(__case['inputs'].shape, None)
            __norm = tf.keras.layers.GroupNormalization(**__case['args'])
            # overwrite the projection weights
            __layer._proj.set_weights([tf.ones((__dim, 2 * __dim), dtype=__dtype), tf.zeros((2 * __dim,), dtype=__dtype)])
            # now the scale and shift should be non null on most contexts but None
            __outputs_ada = __layer(__case['inputs'], contexts=None, training=False)
            __outputs_reg = __norm(__case['inputs'], training=False)
            # null scale and shift => no variation
            self.assertAllEqual(__outputs_ada, __outputs_reg)

    def test_specific_cases(self):
        for __case in self._cases:
            __dtype = __case['inputs'].dtype
            __shape = tuple(__case['inputs'].shape)
            __dim = int(__shape[-1])
            # init with null projection weights
            __layer = mlable.blocks.normalization.AdaptiveGroupNormalization(**__case['args'])
            __layer.build(__case['inputs'].shape, mlable.shapes.filter(__shape, axes=[0, -1]))
            __norm = tf.keras.layers.GroupNormalization(**__case['args'])
            # overwrite the projection weights: scale = 2, shift = 0
            __weights = tf.concat([tf.linalg.diag(__dim * [2.0]), tf.linalg.diag(__dim * [0.0])], axis=-1)
            __layer._proj.set_weights([__weights, tf.zeros((2 * __dim,), dtype=__dtype)])
            # replace the contexts with ones
            __contexts = tf.ones(mlable.shapes.filter(__shape, axes=[0, -1]), dtype=__dtype)
            # now the scale and shift should be non null on most contexts but None
            __outputs_ada = __layer(__case['inputs'], contexts=__contexts, training=False)
            __outputs_reg = __norm(__case['inputs'], training=False)
            # null scale and shift => no variation
            self.assertAllEqual(__outputs_ada, 3.0 * __outputs_reg)
