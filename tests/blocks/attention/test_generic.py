import numpy as np
import tensorflow as tf

import mlable.blocks.attention.generic

# ATTENTION ####################################################################

class AttentionBlockTest(tf.test.TestCase):
    def setUp(self):
        super(AttentionBlockTest, self).setUp()
        self._null_cases = [
            {
                'query': tf.random.uniform((2, 8, 16), minval=-1., maxval=1.),
                'key': tf.ones((2, 4, 16), dtype=tf.float16),
                'value': tf.ones((2, 4, 16), dtype=tf.float16),
                'args': {
                    'head_num': 2,
                    'key_dim': 4,
                    'attention_axes': [1],
                    'use_position': True,},
                'outputs': {
                    'values': tf.zeros((2, 8, 16), dtype=tf.float32),
                },},
            {
                'query': tf.random.uniform((2, 8, 16), minval=-1., maxval=1.),
                'key': tf.stack(2 * [tf.stack(16 * [tf.range(4)], axis=1)], axis=0),
                'value': tf.stack(2 * [tf.stack(16 * [tf.range(4)], axis=1)], axis=0),
                'args': {
                    'head_num': 2,
                    'key_dim': 4,
                    'attention_axes': [1],
                    'use_position': False,},
                'outputs': {
                    'values': tf.zeros((2, 8, 16), dtype=tf.float32),
                },},]

    def test_null_on_constant_inputs(self): # because of the layer norm
        for __case in self._null_cases:
            __layer = mlable.blocks.attention.generic.AttentionBlock(**__case['args'])
            __outputs = __layer(query=__case['query'], key=__case['key'], value=__case['value'])
            # test
            if 'shape' in __case['outputs']:
                self.assertEqual(tuple(__outputs.shape), __case['outputs']['shape'])
            if 'values' in __case['outputs']:
                self.assertAllEqual(__outputs, __case['outputs']['values'])
