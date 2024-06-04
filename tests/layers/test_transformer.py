import numpy as np
import tensorflow as tf

import mlable.layers.transformer

# CACHED ATTENTION ############################################################

def _create_cache(__batch_dim, __init_dim, __num_heads, __head_dim):
        return tf.zeros([2, __batch_dim, __init_dim, __num_heads, __head_dim], dtype=tf.float32)

class CachedAttentionTest(tf.test.TestCase):

    def test_masked_attention(self):
        """Test with a mask tensor."""
        __batch_dim, __seq_dim, __num_heads, __head_dim = 3, 4, 2, 2
        # GPU/CPU case.
        __init_dim = 0
        # Directly tests the keras layer.
        __cache = _create_cache(__batch_dim, __init_dim, __num_heads, __head_dim)
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
        __cache = _create_cache(__batch_dim, __init_dim, __num_heads, __head_dim)
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
                decode_loop_step=__step)
        # checks
        self.assertEqual(__output.shape, (3, 4, 8))
        self.assertEqual(__cache.shape, (2, 3, 4, 2, 2))
