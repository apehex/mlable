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
                    'hidden_dim': 24,},
                'outputs': {
                    'values': tf.zeros((2, 4, 6), dtype=tf.float32),
                },},
            {
                'inputs': tf.stack(4 * [tf.range(16)], axis=1),
                'args': {
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
            __layer = mlable.blocks.transformer.AttentionBlock(**__case['args'])
            __outputs = __layer(query=__case['query'], key=__case['key'], value=__case['value'])
            # test
            if 'shape' in __case['outputs']:
                self.assertEqual(tuple(__outputs.shape), __case['outputs']['shape'])
            if 'values' in __case['outputs']:
                self.assertAllEqual(__outputs, __case['outputs']['values'])

# DECODER ######################################################################

class DecoderBLockTest(tf.test.TestCase):
    def setUp(self):
        super(DecoderBLockTest, self).setUp()
        self._null_cases = [
            {
                'query': tf.random.uniform((2, 8, 16), minval=-1., maxval=1.),
                'key': tf.ones((2, 4, 16), dtype=tf.float16),
                'value': tf.ones((2, 4, 16), dtype=tf.float16),
                'args': {
                    'head_num': 2,
                    'hidden_dim': 64,
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
                    'hidden_dim': 64,
                    'key_dim': 4,
                    'attention_axes': [1],
                    'use_position': False,},
                'outputs': {
                    'values': tf.zeros((2, 8, 16), dtype=tf.float32),
                },},]

    def test_null_on_constant_inputs(self): # because of the layer norm
        for __case in self._null_cases:
            __layer = mlable.blocks.transformer.DecoderBlock(**__case['args'])
            __outputs = __layer(query=__case['query'], key=__case['key'], value=__case['value'])
            # test
            if 'shape' in __case['outputs']:
                self.assertEqual(tuple(__outputs.shape), __case['outputs']['shape'])
            if 'values' in __case['outputs']:
                self.assertAllEqual(__outputs, __case['outputs']['values'])
