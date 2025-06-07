import tensorflow as tf

import mlable.blocks.convolution.resnet

# BASE #########################################################################

class ResnetBlockTest(tf.test.TestCase):
    def setUp(self):
        super(ResnetBlockTest, self).setUp()
        self._cases = [
            {
                'inputs': tf.ones((2, 16, 16, 8), dtype=tf.float16),
                'contexts': tf.ones((2, 1, 1, 4), dtype=tf.float16),
                'args': {'channel_dim': 8, 'group_dim': 2, 'dropout_rate': 0.1, 'epsilon_rate': 1e-6,},
                'outputs': {'shape': (2, 16, 16, 8),},},
            {
                'inputs': tf.convert_to_tensor([16 * [16 * [8 * [__i]]] for __i in range(4)], dtype=tf.float32),
                'contexts': None,
                'args': {'channel_dim': 16, 'group_dim': None, 'dropout_rate': 0.0, 'epsilon_rate': 1e-6,},
                'outputs': {'shape': (4, 16, 16, 16),},},]

    def test_shape(self):
        for __case in self._cases:
            __layer = mlable.blocks.convolution.resnet.ResnetBlock(**__case['args'])
            __outputs = __layer(__case['inputs'], contexts=__case['contexts'], training=False)
            self.assertEqual(tuple(__outputs.shape), __case['outputs']['shape'])
