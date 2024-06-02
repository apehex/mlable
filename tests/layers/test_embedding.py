import numpy as np
import tensorflow as tf

import mlable.layers.embedding

# EINSUM ######################################################################

class RopeTest(tf.test.TestCase):
    def setUp(self):
        super(RopeTest, self).setUp()
        self._test_cases = [
            {
                'init': {
                    'sequence_axis': 1,
                    'feature_axis': -1,
                    'max_wavelength': 100,
                    'scaling_factor': 1.},
                'input': {
                    'inputs': tf.random.uniform(shape=(2, 1, 2, 4), dtype=tf.dtypes.float32),
                    'offset': 3},
                'output': {
                    'shape': (2, 1, 2, 4),}},
            {
                'init': {},
                'input': {
                    'inputs': tf.ones(shape=(1, 4, 6)),
                    'offset': 0},
                'output': {
                    'shape': (1, 4, 6),
                    'values': tf.convert_to_tensor([[
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [-0.30116868, 0.95252377, 0.99784327, 1.3817732, 1.0453222, 1.0021521],
                        [-1.3254442, 0.9029957, 0.9956818, 0.4931506, 1.0883927, 1.0042995],
                        [-1.1311125, 0.85152256, 0.9935159, -0.8488725, 1.1291188, 1.0064424]]])}},
            {
                'init': {},
                'input': {
                    'inputs': tf.zeros(shape=(4, 32, 8)),
                    'offset': 4},
                'output': {
                    'shape': (4, 32, 8),
                    'values': tf.zeros(shape=(4, 32, 8))}},
            {
                'init': {
                    'scaling_factor': 2.},
                'input': {
                    'inputs': tf.ones(shape=(1, 3, 2, 4)),
                    'offset': 0},
                'output': {
                    'shape': (1, 3, 2, 4),
                    'values': tf.convert_to_tensor([
                        [
                            [
                                [1.000000000, 1.000000000, 1.000000000, 1.000000000],
                                [1.000000000, 1.000000000, 1.000000000, 1.000000000],],
                            [
                                [0.398157001, 0.994987488, 1.357008100, 1.004987478],
                                [0.398157001, 0.994987488, 1.357008100, 1.004987478],],
                            [
                                [-0.301168621, 0.989950180, 1.381773233, 1.009949803],
                                [-0.301168621, 0.989950180, 1.381773233, 1.009949803],],]])}},]

    def test_rope_shape(self):
        for __case in self._test_cases:
            __layer = mlable.layers.embedding.RotaryPositionalEmbedding(**__case['init'])
            __outputs = __layer(**__case['input'])
            if 'shape' in __case['output']:
                self.assertEqual(tuple(__outputs.shape), __case['output']['shape'])
            if 'values' in __case['output']:
                np.testing.assert_array_almost_equal(__outputs, __case['output']['values'])
