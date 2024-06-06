import numpy as np
import tensorflow as tf

import mlable.layers.transformer
import mlable.utils

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
            __layer = mlable.layers.transformer.FeedForwardGate(
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

# CACHED ATTENTION ############################################################

class CachedAttentionTest(tf.test.TestCase):

    def test_masked_attention(self):
        """Test with a mask tensor."""
        __batch_dim, __seq_dim, __num_heads, __head_dim = 3, 4, 2, 2
        # GPU/CPU case.
        __init_dim = 0
        # Directly tests the keras layer.
        __cache = mlable.utils.create_cache(__batch_dim, __init_dim, __num_heads, __head_dim)
        # basic attention
        __layer = mlable.layers.transformer.CachedMultiHeadAttention(num_heads=__num_heads, key_dim=__head_dim)
        # input data
        __from_inputs = tf.zeros((__batch_dim, __seq_dim, 8), dtype=np.float32)
        # random mask
        __mask = np.random.randint(2, size=(__batch_dim, __seq_dim, __seq_dim))
        # call
        __output, __cache = __layer(query=__from_inputs, value=__from_inputs, cache=__cache, attention_mask=__mask)
        # checks
        self.assertEqual(__output.shape, (3, 4, 8))
        self.assertEqual(__cache.shape, (2, 3, 4, 2, 2))
        # without cache
        __output, __cache = __layer(query=__from_inputs, value=__from_inputs, attention_mask=__mask)
        self.assertEqual(__output.shape, (3, 4, 8))
        self.assertEqual(__cache.shape, (2, 3, 4, 2, 2))

    def test_padded_decode(self):
        """Test with a mask tensor."""
        __num_heads, __head_dim, __batch_dim, __seq_dim = 2, 2, 3, 4
        # GPU/CPU case.
        __init_dim = __seq_dim
        # Directly tests the keras layer.
        __cache = mlable.utils.create_cache(__batch_dim, __init_dim, __num_heads, __head_dim)
        # basic attention
        __layer = mlable.layers.transformer.CachedMultiHeadAttention(num_heads=__num_heads, key_dim=__head_dim)
        # Generate data for the input (non-mask) tensors.
        __from_inputs = tf.zeros((__batch_dim, __seq_dim, 8), dtype=np.float32)
        # decode step
        __step = 2
        # random mask
        __mask = np.random.randint(2, size=(__batch_dim, __seq_dim, __seq_dim), dtype=np.int32)
        # call
        __output, __cache = __layer(
                query=__from_inputs,
                value=__from_inputs,
                attention_mask=__mask,
                cache=__cache,
                step=__step)
        # checks
        self.assertEqual(__output.shape, (3, 4, 8))
        self.assertEqual(__cache.shape, (2, 3, 4, 2, 2))
