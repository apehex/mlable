"""Test the agent on a fork"""

import numpy as np
import tensorflow as tf

import mlable.layers.embedding

# EINSUM ######################################################################

class RopeTest(tf.test.TestCase):
    def setUp(self):
        super(RopeTest, self).setUp()
        self._test_cases = [
            {
                'input_embedding_shape': (2, 1, 2, 4),
                'position': 3,
                'head_dim': 4,
                'sequence_axis': 1,
                'max_wavelength': 100,
                'expected': [
                    [[
                        [-1.1311126, 0.6598157, -0.8488725, 1.2508571],
                        [-1.1311126, 0.6598157, -0.8488725, 1.2508571],]],
                    [[
                        [-1.1311126, 0.6598157, -0.8488725, 1.2508571],
                        [-1.1311126, 0.6598157, -0.8488725, 1.2508571],]],],}]

    def test_rope_shape(self):
        for __case in self._test_cases:
            __layer = mlable.layers.embedding.RotaryPositionalEmbedding(
                head_dim=__case["head_dim"],
                sequence_axis=__case["sequence_axis"],
                max_wavelength=__case["max_wavelength"])
            __output = __layer(
                inputs=tf.ones(__case['input_embedding_shape']),
                positions=tf.convert_to_tensor([[__case['position']]], dtype=tf.dtypes.float32))
            np.testing.assert_array_almost_equal(__output, tf.convert_to_tensor(__case["expected"], dtype=tf.dtypes.float32))
