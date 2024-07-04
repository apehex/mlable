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
