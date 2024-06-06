import numpy as np
import tensorflow as tf

import mlable.layers.linear

# EINSUM ######################################################################

class EinsumTest(tf.test.TestCase):
    def setUp(self):
        super(EinsumTest, self).setUp()
        self._test_cases = [
            {
                "inputs_shape": (1, 4),
                "params_shape": (3, 2, 4, 3),
                "equation": "TD,SNDH->STNH",
                "expected_shape": (3, 1, 2, 3),},
            {
                "inputs_shape": (1, 2, 4),
                "params_shape": (2, 4, 8),
                "equation": "ANH,NHD->AD",
                "expected_shape": (1, 8),},]

    def test_einsum_shape(self):
        for __case in self._test_cases:
            __layer = mlable.layers.linear.Einsum(equation=__case["equation"], shape=__case["params_shape"])
            __layer.build(__case["inputs_shape"])
            __output = __layer(tf.ones(__case["inputs_shape"]))
            self.assertEqual(__output.shape, __case["expected_shape"])

# FF ##########################################################################

class FeedForwardBlockTest(tf.test.TestCase):
    def setUp(self):
        super(FeedForwardBlockTest, self).setUp()
        self._test_cases = [
            {
                'input_dim': 2,
                'hidden_dim': 3,
                'batch_size': 2,
                'expected_val': [11.726998, 47.998482],
                'expected_shape': (2, 1, 2),},]

    def test_ffn(self):
        for __case in self._test_cases:
            # inputs
            __inputs = tf.reshape(tf.range(1, __case['batch_size'] + 1, dtype=tf.float32), (__case['batch_size'], 1, 1))
            __inputs = tf.repeat(__inputs, __case['input_dim'], axis=-1)
            # init
            __layer = mlable.layers.linear.FeedForwardGate(
                input_dim=__case["input_dim"],
                hidden_dim=__case["hidden_dim"])
            # build
            _ = __layer(__inputs)
            # set weights
            __layer._gelu.set_weights([np.ones((__case['input_dim'], __case['hidden_dim']))])
            __layer._linear.set_weights([np.ones((__case['input_dim'], __case['hidden_dim']))])
            __layer._output.set_weights([np.ones((__case['hidden_dim'], __case['input_dim']))])
            # compute
            __output = __layer(__inputs)
            # test
            np.testing.assert_array_almost_equal(__output[:, 0, 0].numpy().tolist(), __case["expected_val"])
            self.assertEqual(tuple(__output.shape), __case['expected_shape'])