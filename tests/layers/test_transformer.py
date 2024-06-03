import numpy as np
import tensorflow as tf

import mlable.layers.transformer

# CACHED ATTENTION ############################################################

def _create_cache(batch_dim, init_decode_length, num_heads, head_dim):
        return tf.zeros([batch_dim, init_decode_length, num_heads, head_dim], dtype=tf.float32)

class CachedAttentionTest(tf.test.TestCase):

    def test_masked_attention(self):
        """Test with a mask tensor."""
        num_heads, head_dim = 2, 2
        # Create a 3-dimensional input (the first dimension is implicit).
        batch_dim, seq_dim = 3, 4
        # GPU/CPU case.
        init_decode_length = 0
        # Directly tests the keras layer.
        key_cache = _create_cache(batch_dim, init_decode_length, num_heads, head_dim)
        value_cache = _create_cache(batch_dim, init_decode_length, num_heads, head_dim)
        layer = mlable.layers.transformer.CachedMultiHeadAttention(num_heads=num_heads, key_dim=head_dim)

        # Generate data for the input (non-mask) tensors.
        from_data = tf.zeros((batch_dim, seq_dim, 8), dtype=np.float32)
        # Invoke the data with a random set of mask data. This should mask at least
        # one element.
        mask_data = np.random.randint(2, size=(batch_dim, seq_dim, seq_dim))
        masked_output_data, key_cache, value_cache = layer(query=from_data, value=from_data, key_cache=key_cache, value_cache=value_cache, attention_mask=mask_data)
        self.assertEqual(masked_output_data.shape, (3, 4, 8))
        self.assertEqual(value_cache.shape, (3, 4, 2, 2))

        # Tests inputs without cache.
        masked_output_data, key_cache, value_cache = layer(query=from_data, value=from_data, attention_mask=mask_data)
        self.assertEqual(masked_output_data.shape, (3, 4, 8))

    def test_padded_decode(self):
        """Test with a mask tensor."""
        num_heads, head_dim = 2, 2
        # TPU decoding should pre-allocate the entire sequence.
        batch_dim, seq_dim = 3, 4
        # GPU/CPU case.
        init_decode_length = seq_dim

        # Directly tests the keras layer.
        key_cache = _create_cache(batch_dim, init_decode_length, num_heads, head_dim)
        value_cache = _create_cache(batch_dim, init_decode_length, num_heads, head_dim)
        layer = mlable.layers.transformer.CachedMultiHeadAttention(num_heads=num_heads, key_dim=head_dim)

        # Generate data for the input (non-mask) tensors.
        from_data = tf.zeros((batch_dim, seq_dim, 8), dtype=np.float32)
        decode_loop_step = 2
        mask_data = np.random.randint(2, size=(batch_dim, seq_dim, seq_dim), dtype=np.int32)
        # Testing the invocation directly as Keras cannot consume inputs correctly.
        masked_output_data, key_cache, value_cache = layer(
                query=from_data,
                value=from_data,
                attention_mask=mask_data,
                key_cache=key_cache,
                value_cache=value_cache,
                decode_loop_step=decode_loop_step)
        self.assertEqual(masked_output_data.shape, (3, 4, 8))
        self.assertEqual(value_cache.shape, (3, 4, 2, 2))
