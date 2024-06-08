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
            __outputs = __layer(__inputs)
            # test
            np.testing.assert_array_almost_equal(__outputs[:, 0, 0].numpy().tolist(), __case["expected_val"])
            self.assertEqual(tuple(__outputs.shape), __case['expected_shape'])

# CACHED ATTENTION ############################################################

class CachedAttentionTest(tf.test.TestCase):

    def test_batch_decode(self):
        __batch_dim, __seq_dim, __embed_dim, __num_heads, __head_dim = 2, 4, 6, 2, 3
        # training mode
        __cache = None
        __mask = None
        __step = None
        # basic attention
        __layer = mlable.layers.transformer.CachedMultiHeadAttention(num_heads=__num_heads, key_dim=__head_dim)
        # input data
        __inputs = tf.zeros((__batch_dim, __seq_dim, __embed_dim), dtype=np.float32)
        # call
        __outputs, __scores, __cache = __layer(query=__inputs, value=__inputs, cache=__cache, attention_mask=__mask, step=__step, return_attention_scores=True)
        # check
        self.assertEqual(__outputs.shape, __inputs.shape)
        self.assertEqual(__scores.shape, (__batch_dim, __num_heads, __seq_dim, __seq_dim))
        self.assertEqual(__cache, None)

    def test_sequential_decode_update(self):
        __batch_dim, __seq_dim, __embed_dim, __num_heads, __head_dim = 1, 4, 6, 2, 3
        # sampling mode
        __cache = mlable.utils.create_cache(batch_dim=__batch_dim, cache_dim=__seq_dim, num_heads=__num_heads, head_dim=__head_dim)
        # random mask
        __mask = np.random.randint(2, size=(__batch_dim, 1, __seq_dim))
        # update index
        __step = __seq_dim // 2
        # basic attention
        __layer = mlable.layers.transformer.CachedMultiHeadAttention(num_heads=__num_heads, key_dim=__head_dim)
        # input data
        __inputs = tf.ones((__batch_dim, 1, __embed_dim), dtype=np.float32)
        # call
        __outputs, __scores, __cache = __layer(query=__inputs, value=__inputs, cache=__cache, attention_mask=__mask, step=__step, return_attention_scores=True)
        # check
        self.assertEqual(__outputs.shape, __inputs.shape)
        self.assertEqual(__scores.shape, (__batch_dim, __num_heads, 1, __seq_dim))
        self.assertEqual(__cache.shape, (2, __batch_dim, __seq_dim, __num_heads, __head_dim))

    def test_sequential_decode_append(self):
        __batch_dim, __seq_dim, __embed_dim, __num_heads, __head_dim = 1, 4, 6, 2, 3
        # sampling mode
        __cache = mlable.utils.create_cache(batch_dim=__batch_dim, cache_dim=__seq_dim, num_heads=__num_heads, head_dim=__head_dim)
        # random mask
        __mask = np.random.randint(2, size=(__batch_dim, 1, __seq_dim + 1))
        # update index
        __step = None
        # basic attention
        __layer = mlable.layers.transformer.CachedMultiHeadAttention(num_heads=__num_heads, key_dim=__head_dim)
        # input data
        __inputs = tf.ones((__batch_dim, 1, __embed_dim), dtype=np.float32)
        # call
        __outputs, __scores, __cache = __layer(query=__inputs, value=__inputs, cache=__cache, attention_mask=__mask, step=__step, return_attention_scores=True)
        # check
        self.assertEqual(__outputs.shape, __inputs.shape)
        self.assertEqual(__scores.shape, (__batch_dim, __num_heads, 1, __seq_dim + 1))
        self.assertEqual(__cache.shape, (2, __batch_dim, __seq_dim + 1, __num_heads, __head_dim))

    # TODO test mask with cache appending
    def test_random_masking(self):
        __batch_dim, __seq_dim, __embed_dim, __num_heads, __head_dim = 1, 4, 6, 2, 3
        # sampling mode
        __cache = mlable.utils.create_cache(batch_dim=__batch_dim, cache_dim=__seq_dim, num_heads=__num_heads, head_dim=__head_dim)
        # random mask
        __mask = np.random.randint(2, size=(__batch_dim, __seq_dim, __seq_dim))
        # update index
        __step = __seq_dim // 2
        # basic attention
        __layer = mlable.layers.transformer.CachedMultiHeadAttention(num_heads=__num_heads, key_dim=__head_dim)
        # input data
        __inputs = tf.ones((__batch_dim, 1, __embed_dim), dtype=np.float32)
        # call
        __outputs, __scores, __cache = __layer(query=__inputs, value=__inputs, cache=__cache, attention_mask=__mask, step=__step, return_attention_scores=True)
        # check
        self.assertEqual(__outputs.shape, (__batch_dim, __seq_dim, __embed_dim))
        self.assertEqual(__scores.shape, (__batch_dim, __num_heads, __seq_dim, __seq_dim))
        self.assertEqual(__cache.shape, (2, __batch_dim, __seq_dim, __num_heads, __head_dim))
