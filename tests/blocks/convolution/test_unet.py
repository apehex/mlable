import tensorflow as tf

import mlable.blocks.convolution.unet

# ATTENTION ####################################################################

class AttentionBlockTest(tf.test.TestCase):
    def setUp(self):
        super(AttentionBlockTest, self).setUp()
        self._cases = [
            {
                'inputs': tf.ones((2, 16, 16, 8), dtype=tf.float16),
                'contexts': None,
                'args': {'group_dim': None, 'head_dim': None, 'head_num': None, 'dropout_rate': 0.1, 'epsilon_rate': 1e-6,},},
            {
                'inputs': tf.convert_to_tensor([16 * [16 * [8 * [__i]]] for __i in range(4)], dtype=tf.float32),
                'contexts': tf.ones((4, 1, 1, 1), dtype=tf.float32),
                'args': {'group_dim': None, 'head_dim': None, 'head_num': None, 'dropout_rate': 0.0, 'epsilon_rate': 1e-6,},},
            {
                'inputs': tf.random.normal((2, 16, 16, 8), dtype=tf.float32),
                'contexts': tf.random.normal((2, 1, 1, 4), dtype=tf.float32),
                'args': {'group_dim': None, 'head_dim': None, 'head_num': None, 'dropout_rate': 0.0, 'epsilon_rate': 1e-6,},},
            {
                'inputs': tf.random.normal((2, 16, 16, 8), dtype=tf.float32),
                'contexts': tf.random.normal((2, 4), dtype=tf.float32),
                'args': {'group_dim': 8, 'head_dim': 8, 'head_num': 4, 'dropout_rate': 0.0, 'epsilon_rate': 1e-6,},},
            {
                'inputs': tf.random.normal((2, 16, 16, 8), dtype=tf.float32),
                'contexts': tf.random.normal((2, 1, 1, 1), dtype=tf.float32),
                'args': {'group_dim': 8, 'head_dim': None, 'head_num': None, 'dropout_rate': 0.0, 'epsilon_rate': 1e-6,},},
            {
                'inputs': tf.random.normal((2, 16, 16, 8), dtype=tf.float32),
                'contexts': tf.random.normal((2, 1, 1, 8), dtype=tf.float32),
                'args': {'group_dim': None, 'head_dim': None, 'head_num': None, 'dropout_rate': 0.0, 'epsilon_rate': 1e-6,},
                'outputs': {'shape': (2, 32, 32, 4),},},
            {
                'inputs': tf.random.normal((2, 16, 16, 8), dtype=tf.float32),
                'contexts': tf.random.normal((2, 1), dtype=tf.float32),
                'args': {'group_dim': None, 'head_dim': None, 'head_num': None, 'dropout_rate': 0.0, 'epsilon_rate': 1e-6,},},
            {
                'inputs': tf.random.normal((2, 16, 16, 8), dtype=tf.float32),
                'contexts': tf.random.normal((2, 8), dtype=tf.float32),
                'args': {'group_dim': None, 'head_dim': None, 'head_num': None, 'dropout_rate': 0.0, 'epsilon_rate': 1e-6,},},]

    def test_shape(self):
        for __case in self._cases:
            __layer = mlable.blocks.convolution.unet.AttentionBlock(**__case['args'])
            __outputs = __layer(__case['inputs'], contexts=__case['contexts'], training=False)
            self.assertEqual(tuple(__outputs.shape), tuple(__case['inputs'].shape))

# GENERIC ######################################################################

class UnetBlockTest(tf.test.TestCase):
    def setUp(self):
        super(UnetBlockTest, self).setUp()
        self._cases = [
            {
                'inputs': tf.ones((2, 16, 16, 8), dtype=tf.float16),
                'contexts': None,
                'args': {'channel_dim': None, 'group_dim': None, 'head_dim': None, 'head_num': None, 'layer_num': None, 'add_attention': False, 'add_downsampling': False, 'add_upsampling': False, 'dropout_rate': 0.1, 'epsilon_rate': 1e-6,},
                'outputs': {'shape': (2, 16, 16, 8),},},
            {
                'inputs': tf.convert_to_tensor([16 * [16 * [8 * [__i]]] for __i in range(4)], dtype=tf.float32),
                'contexts': tf.ones((4, 1, 1, 1), dtype=tf.float32),
                'args': {'channel_dim': 32, 'group_dim': None, 'head_dim': None, 'head_num': None, 'layer_num': 1, 'add_attention': True, 'add_downsampling': False, 'add_upsampling': False, 'dropout_rate': 0.0, 'epsilon_rate': 1e-6,},
                'outputs': {'shape': (4, 16, 16, 32),},},
            {
                'inputs': tf.random.normal((2, 16, 16, 8), dtype=tf.float32),
                'contexts': tf.random.normal((2, 1, 1, 4), dtype=tf.float32),
                'args': {'channel_dim': 32, 'group_dim': None, 'head_dim': None, 'head_num': None, 'layer_num': 2, 'add_attention': True, 'add_downsampling': False, 'add_upsampling': False, 'dropout_rate': 0.0, 'epsilon_rate': 1e-6,},
                'outputs': {'shape': (2, 16, 16, 32),},},
            {
                'inputs': tf.random.normal((2, 16, 16, 8), dtype=tf.float32),
                'contexts': tf.random.normal((2, 4), dtype=tf.float32),
                'args': {'channel_dim': 32, 'group_dim': 8, 'head_dim': 8, 'head_num': 4, 'layer_num': 4, 'add_attention': False, 'add_downsampling': False, 'add_upsampling': False, 'dropout_rate': 0.0, 'epsilon_rate': 1e-6,},
                'outputs': {'shape': (2, 16, 16, 32),},},
            {
                'inputs': tf.random.normal((2, 16, 16, 8), dtype=tf.float32),
                'contexts': tf.random.normal((2, 1, 1, 1), dtype=tf.float32),
                'args': {'channel_dim': 32, 'group_dim': 8, 'head_dim': None, 'head_num': None, 'layer_num': 2, 'add_attention': False, 'add_downsampling': True, 'add_upsampling': False, 'dropout_rate': 0.0, 'epsilon_rate': 1e-6,},
                'outputs': {'shape': (2, 8, 8, 32),},},
            {
                'inputs': tf.random.normal((2, 16, 16, 8), dtype=tf.float32),
                'contexts': tf.random.normal((2, 1, 1, 8), dtype=tf.float32),
                'args': {'channel_dim': 4, 'group_dim': None, 'head_dim': None, 'head_num': None, 'layer_num': 2, 'add_attention': False, 'add_downsampling': False, 'add_upsampling': True, 'dropout_rate': 0.0, 'epsilon_rate': 1e-6,},
                'outputs': {'shape': (2, 32, 32, 4),},},
            {
                'inputs': tf.random.normal((2, 16, 16, 8), dtype=tf.float32),
                'contexts': tf.random.normal((2, 1), dtype=tf.float32),
                'args': {'channel_dim': 32, 'group_dim': None, 'head_dim': None, 'head_num': None, 'layer_num': 2, 'add_attention': True, 'add_downsampling': True, 'add_upsampling': False, 'dropout_rate': 0.0, 'epsilon_rate': 1e-6,},
                'outputs': {'shape': (2, 8, 8, 32),},},
            {
                'inputs': tf.random.normal((2, 16, 16, 8), dtype=tf.float32),
                'contexts': tf.random.normal((2, 8), dtype=tf.float32),
                'args': {'channel_dim': 4, 'group_dim': None, 'head_dim': None, 'head_num': None, 'layer_num': 2, 'add_attention': True, 'add_downsampling': False, 'add_upsampling': True, 'dropout_rate': 0.0, 'epsilon_rate': 1e-6,},
                'outputs': {'shape': (2, 32, 32, 4),},},]

    def test_shape(self):
        for __case in self._cases:
            __layer = mlable.blocks.convolution.unet.UnetBlock(**__case['args'])
            __outputs = __layer(__case['inputs'], contexts=__case['contexts'], training=False)
            self.assertEqual(tuple(__outputs.shape), __case['outputs']['shape'])
