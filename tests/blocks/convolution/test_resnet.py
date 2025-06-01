import tensorflow as tf

import mlable.blocks.convolution.resnet

# BASE #########################################################################

class ResnetBlockTest(tf.test.TestCase):
    def setUp(self):
        super(ResnetBlockTest, self).setUp()
        self._cases = [
            {
                'inputs': tf.ones((2, 16, 16, 8), dtype=tf.float16),
                'args': {'channel_dim': 8, 'group_dim': 2, 'dropout_rate': 0.1, 'epsilon_rate': 1e-6,},
                'outputs': {'shape': (2, 16, 16, 8),},},
            {
                'inputs': tf.convert_to_tensor([16 * [16 * [8 * [__i]]] for __i in range(4)], dtype=tf.float32),
                'args': {'channel_dim': 16, 'group_dim': None, 'dropout_rate': 0.0, 'epsilon_rate': 1e-6,},
                'outputs': {'shape': (4, 16, 16, 16),},},]

    def test_shape(self):
        for __case in self._cases:
            __layer = mlable.blocks.convolution.resnet.ResnetBlock(**__case['args'])
            __outputs = __layer(__case['inputs'], training=False)
            self.assertEqual(tuple(__outputs.shape), __case['outputs']['shape'])

# BASE #########################################################################

class EncoderBlockTest(tf.test.TestCase):
    def setUp(self):
        super(EncoderBlockTest, self).setUp()
        self._cases = [
            {
                'inputs': tf.ones((2, 16, 16, 8), dtype=tf.float16),
                'args': {'channel_dim': 8, 'group_dim': 2, 'layer_num': 1, 'dropout_rate': 0.1, 'epsilon_rate': 1e-6, 'downsample_on': False,},
                'outputs': {'shape': (2, 16, 16, 8),},},
            {
                'inputs': tf.convert_to_tensor([16 * [16 * [8 * [__i]]] for __i in range(4)], dtype=tf.float32),
                'args': {'channel_dim': 16, 'group_dim': None, 'layer_num': 2, 'dropout_rate': 0.0, 'epsilon_rate': 1e-6, 'downsample_on': True,},
                'outputs': {'shape': (4, 8, 8, 16),},},]

    def test_shape(self):
        for __case in self._cases:
            __layer = mlable.blocks.convolution.resnet.EncoderBlock(**__case['args'])
            __outputs = __layer(__case['inputs'], training=False)
            self.assertEqual(tuple(__outputs.shape), __case['outputs']['shape'])

# BASE #########################################################################

class TransformerBlockTest(tf.test.TestCase):
    def setUp(self):
        super(TransformerBlockTest, self).setUp()
        self._cases = [
            {
                'inputs': tf.ones((2, 16, 16, 8), dtype=tf.float16),
                'args': {'channel_dim': 8, 'group_dim': 2, 'head_dim': 1, 'layer_num': 1, 'dropout_rate': 0.1, 'epsilon_rate': 1e-6,},
                'outputs': {'shape': (2, 16, 16, 8),},},
            {
                'inputs': tf.convert_to_tensor([16 * [16 * [8 * [__i]]] for __i in range(4)], dtype=tf.float32),
                'args': {'channel_dim': 8, 'group_dim': 4, 'head_dim': 2, 'layer_num': 2, 'dropout_rate': 0.0, 'epsilon_rate': 1e-6,},
                'outputs': {'shape': (4, 16, 16, 8),},},]

    def test_shape(self):
        for __case in self._cases:
            __layer = mlable.blocks.convolution.resnet.TransformerBlock(**__case['args'])
            __outputs = __layer(__case['inputs'], training=False)
            self.assertEqual(tuple(__outputs.shape), __case['outputs']['shape'])

# BASE #########################################################################

class DecoderBlockTest(tf.test.TestCase):
    def setUp(self):
        super(DecoderBlockTest, self).setUp()
        self._cases = [
            {
                'inputs': tf.ones((2, 16, 16, 8), dtype=tf.float16),
                'args': {'channel_dim': 8, 'group_dim': 2, 'layer_num': 1, 'dropout_rate': 0.1, 'epsilon_rate': 1e-6, 'upsample_on': False,},
                'outputs': {'shape': (2, 16, 16, 8),},},
            {
                'inputs': tf.convert_to_tensor([16 * [16 * [8 * [__i]]] for __i in range(4)], dtype=tf.float32),
                'args': {'channel_dim': 16, 'group_dim': None, 'layer_num': 2, 'dropout_rate': 0.0, 'epsilon_rate': 1e-6, 'upsample_on': True,},
                'outputs': {'shape': (4, 32, 32, 16),},},]

    def test_shape(self):
        for __case in self._cases:
            __layer = mlable.blocks.convolution.resnet.DecoderBlock(**__case['args'])
            __outputs = __layer(__case['inputs'], training=False)
            self.assertEqual(tuple(__outputs.shape), __case['outputs']['shape'])
