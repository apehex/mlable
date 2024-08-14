import numpy as np
import tensorflow as tf

import mlable.blocks.transformer

# FF ##########################################################################

class FeedForwardBlockTest(tf.test.TestCase):
    def setUp(self):
        super(FeedForwardBlockTest, self).setUp()
        self._null_cases = [
            {
                'inputs': tf.ones((2, 4, 6), dtype=tf.float16),
                'args': {
                    'embed_dim': 6,
                    'hidden_dim': 24,},
                'outputs': {
                    'values': tf.zeros((2, 4, 6), dtype=tf.float32),
                },},
            {
                'inputs': tf.stack(4 * [tf.range(16)], axis=1),
                'args': {
                    'embed_dim': 4,
                    'hidden_dim': 16,},
                'outputs': {
                    'values': tf.zeros((16, 4), dtype=tf.float32),
                },},]

    def test_null_on_constant_inputs(self): # because of the layer norm
        for __case in self._null_cases:
            __layer = mlable.blocks.transformer.FeedForwardBlock(**__case['args'])
            __outputs = __layer(__case['inputs'])
            # test
            if 'shape' in __case['outputs']:
                self.assertEqual(tuple(__outputs.shape), __case['outputs']['shape'])
            if 'values' in __case['outputs']:
                self.assertAllEqual(__outputs, __case['outputs']['values'])

# SELF ATTENTION ##############################################################

class SelfAttentionBlockTest(tf.test.TestCase):
    def setUp(self):
        super(SelfAttentionBlockTest, self).setUp()
        self._null_cases = [
            {
                'inputs': tf.ones((2, 4, 8), dtype=tf.float16),
                'args': {
                    'num_heads': 2,
                    'head_dim': 4,
                    'sequence_axis': 1,},
                'outputs': {
                    'values': tf.zeros((2, 4, 8), dtype=tf.float32),
                },},
            {
                'inputs': tf.reshape(tf.stack(4 * [tf.range(16)], axis=1), (1, 16, 4)),
                'args': {
                    'num_heads': 2,
                    'head_dim': 4,
                    'sequence_axis': 1,},
                'outputs': {
                    'values': tf.zeros((1, 16, 4), dtype=tf.float32),
                },},]

    def test_null_on_constant_inputs(self): # because of the layer norm
        for __case in self._null_cases:
            __layer = mlable.blocks.transformer.SelfAttentionBlock(**__case['args'])
            __outputs = __layer(__case['inputs'])
            # test
            if 'shape' in __case['outputs']:
                self.assertEqual(tuple(__outputs.shape), __case['outputs']['shape'])
            if 'values' in __case['outputs']:
                self.assertAllEqual(__outputs, __case['outputs']['values'])

# CROSS ATTENTION #############################################################

class CrossAttentionBlockTest(tf.test.TestCase):
    def setUp(self):
        super(CrossAttentionBlockTest, self).setUp()
        self._null_cases = [
            {
                'inputs': tf.random.uniform((2, 8, 16), minval=-1., maxval=1.),
                'contexts': tf.ones((2, 4, 8), dtype=tf.float16),
                'args': {
                    'num_heads': 2,
                    'head_dim': 4,
                    'sequence_axis': 1,},
                'outputs': {
                    'values': tf.zeros((2, 8, 16), dtype=tf.float32),
                },},]

    def test_null_on_constant_inputs(self): # because of the layer norm
        for __case in self._null_cases:
            __layer = mlable.blocks.transformer.CrossAttentionBlock(**__case['args'])
            __outputs = __layer(__case['inputs'], __case['contexts'])
            # test
            if 'shape' in __case['outputs']:
                self.assertEqual(tuple(__outputs.shape), __case['outputs']['shape'])
            if 'values' in __case['outputs']:
                self.assertAllEqual(__outputs, __case['outputs']['values'])
