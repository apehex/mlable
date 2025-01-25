import tensorflow as tf

import mlable.blocks.convolution

# CONVOLUTION ##################################################################

class ConvolutionBlockTest(tf.test.TestCase):
    def setUp(self):
        super(ConvolutionBlockTest, self).setUp()
        self._null_cases = [
            {
                'inputs': tf.ones((2, 16, 16, 8), dtype=tf.float16),
                'args': {'channel_dim': 8, 'kernel_dim': 5, 'stride_dim': 1, 'dropout_rate': 0.1, 'padding': 'same', 'epsilon': 1e-6,},
                'outputs': {'values': tf.zeros((2, 16, 16, 8), dtype=tf.float32),},},
            {
                'inputs': tf.convert_to_tensor([30 * [30 * [8 * [__i]]] for __i in range(4)], dtype=tf.int32),
                'args': {'channel_dim': 16, 'kernel_dim': 4, 'stride_dim': 2, 'dropout_rate': 0.1, 'padding': 'valid', 'epsilon': 1e-6,},
                'outputs': {'values': tf.zeros((4, 14, 14, 16), dtype=tf.float32),},},]

    def test_null_on_constant_inputs(self): # because of the layer norm
        for __case in self._null_cases:
            __layer = mlable.blocks.convolution.ConvolutionBlock(**__case['args'])
            __outputs = __layer(__case['inputs'], training=False)
            # test
            if 'shape' in __case['outputs']:
                self.assertEqual(tuple(__outputs.shape), __case['outputs']['shape'])
            if 'values' in __case['outputs']:
                self.assertAllEqual(__outputs, __case['outputs']['values'])

# RESIDUAL #####################################################################

class ResidualBlockTest(tf.test.TestCase):
    def setUp(self):
        super(ResidualBlockTest, self).setUp()
        self._null_cases = [
            {
                'inputs': tf.ones((2, 16, 16, 8), dtype=tf.float16),
                'args': {'channel_dim': 8, 'kernel_dim': 5, 'stride_dim': 1, 'dropout_rate': 0.1, 'padding': 'same', 'epsilon': 1e-6,},
                'outputs': {'values': tf.ones((2, 16, 16, 8), dtype=tf.float32),},},]

    def test_null_on_constant_inputs(self): # because of the layer norm
        for __case in self._null_cases:
            __layer = mlable.blocks.convolution.ResidualBlock(**__case['args'])
            __outputs = __layer(__case['inputs'], training=False)
            # test
            if 'shape' in __case['outputs']:
                self.assertEqual(tuple(__outputs.shape), __case['outputs']['shape'])
            if 'values' in __case['outputs']:
                self.assertAllEqual(__outputs, __case['outputs']['values'])

# TRANSPOSE ####################################################################

class TransposeConvolutionBlockTest(tf.test.TestCase):
    def setUp(self):
        super(TransposeConvolutionBlockTest, self).setUp()
        self._null_cases = [
            {
                'inputs': tf.ones((2, 16, 16, 8), dtype=tf.float16),
                'args': {'channel_dim': 8, 'kernel_dim': 5, 'stride_dim': 1, 'dropout_rate': 0.1, 'padding': 'same', 'epsilon': 1e-6,},
                'outputs': {'values': tf.zeros((2, 16, 16, 8), dtype=tf.float32),},},
            {
                'inputs': tf.convert_to_tensor([2 * [2 * [8 * [__i]]] for __i in range(4)], dtype=tf.int32),
                'args': {'channel_dim': 16, 'kernel_dim': 4, 'stride_dim': 2, 'dropout_rate': 0.1, 'padding': 'valid', 'epsilon': 1e-6,},
                'outputs': {'values': tf.zeros((4, 6, 6, 16), dtype=tf.float32),},},]

    def test_null_on_constant_inputs(self): # because of the layer norm
        for __case in self._null_cases:
            __layer = mlable.blocks.convolution.TransposeConvolutionBlock(**__case['args'])
            __outputs = __layer(__case['inputs'], training=False)
            # test
            if 'shape' in __case['outputs']:
                self.assertEqual(tuple(__outputs.shape), __case['outputs']['shape'])
            if 'values' in __case['outputs']:
                self.assertAllEqual(__outputs, __case['outputs']['values'])
