import numpy as np
import tensorflow as tf

import mlable.layers.transformer
import mlable.utils

# FF ##########################################################################

class FeedForwardGateTest(tf.test.TestCase):
    def setUp(self):
        super(FeedForwardGateTest, self).setUp()
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
            self.assertAllClose(__outputs[:, 0, 0].numpy().tolist(), __case["expected_val"])
            self.assertEqual(tuple(__outputs.shape), __case['expected_shape'])

# CACHED ATTENTION ############################################################

class CachedAttentionTest(tf.test.TestCase):

    def test_internals(self):
        __batch_dim, __seq_dim, __embed_dim, __num_heads, __head_dim = 2, 4, 6, 2, 4
        # training mode
        __cache = None
        __mask = None
        __step = None
        # basic attention
        __layer = mlable.layers.transformer.CachedMultiHeadAttention(num_heads=__num_heads, key_dim=__head_dim, dropout=0.1)
        # input data
        __inputs = tf.zeros((__batch_dim, __seq_dim, __embed_dim), dtype=tf.dtypes.float32)
        # call
        __outputs, __scores, __cache = __layer(query=__inputs, value=__inputs, cache=__cache, attention_mask=__mask, step=__step, return_attention_scores=True)
        # check
        self.assertEqual(len(__layer.trainable_variables), 8)
        self.assertEqual(len(__layer.non_trainable_variables), 1)

    def test_batch_decode(self):
        __batch_dim, __seq_dim, __embed_dim, __num_heads, __head_dim = 2, 4, 6, 2, 3
        # training mode
        __cache = None
        __mask = None
        __step = None
        # basic attention
        __layer = mlable.layers.transformer.CachedMultiHeadAttention(num_heads=__num_heads, key_dim=__head_dim)
        # input data
        __inputs = tf.zeros((__batch_dim, __seq_dim, __embed_dim), dtype=tf.dtypes.float32)
        # call
        __outputs, __scores, __cache = __layer(query=__inputs, value=__inputs, cache=__cache, attention_mask=__mask, step=__step, return_attention_scores=True)
        # check
        self.assertEqual(__outputs.shape, __inputs.shape)
        self.assertEqual(__scores.shape, (__batch_dim, __num_heads, __seq_dim, __seq_dim))
        self.assertEqual(__cache, None)

    def test_cache_is_ignored_during_training(self):
        __batch_dim, __seq_dim, __embed_dim, __num_heads, __head_dim = 2, 4, 6, 2, 3
        # training mode
        __training = True
        __cache = mlable.utils.create_cache(batch_dim=__batch_dim, cache_dim=__seq_dim, num_heads=__num_heads, head_dim=__head_dim)
        __mask = None
        __step = 0
        # basic attention
        __layer = mlable.layers.transformer.CachedMultiHeadAttention(num_heads=__num_heads, key_dim=__head_dim)
        # input data
        __inputs = tf.zeros((__batch_dim, __seq_dim, __embed_dim), dtype=tf.dtypes.float32)
        # call
        __outputs, __scores, __cache = __layer(query=__inputs, value=__inputs, cache=__cache, attention_mask=__mask, step=__step, return_attention_scores=True, training=__training)
        # check
        self.assertEqual(__outputs.shape, __inputs.shape)
        self.assertEqual(__scores.shape, (__batch_dim, __num_heads, __seq_dim, __seq_dim))
        self.assertEqual(__cache, None)

    def test_sequential_decode_update(self):
        __batch_dim, __seq_dim, __embed_dim, __num_heads, __head_dim = 1, 4, 6, 2, 3
        # sampling mode
        __cache = mlable.utils.create_cache(batch_dim=__batch_dim, cache_dim=__seq_dim, num_heads=__num_heads, head_dim=__head_dim)
        # random mask
        __mask = tf.random.uniform(shape=(__batch_dim, 1, __seq_dim), minval=0, maxval=2, dtype=tf.dtypes.int32)
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
        __mask = tf.random.uniform(shape=(__batch_dim, 1, __seq_dim + 1), minval=0, maxval=2, dtype=tf.dtypes.int32)
        # update index
        __step = None
        # basic attention
        __layer = mlable.layers.transformer.CachedMultiHeadAttention(num_heads=__num_heads, key_dim=__head_dim)
        # input data
        __inputs = tf.ones((__batch_dim, 1, __embed_dim), dtype=tf.dtypes.float32)
        # call
        __outputs, __scores, __cache = __layer(query=__inputs, value=__inputs, cache=__cache, attention_mask=__mask, step=__step, return_attention_scores=True)
        # check
        self.assertEqual(__outputs.shape, __inputs.shape)
        self.assertEqual(__scores.shape, (__batch_dim, __num_heads, 1, __seq_dim + 1))
        self.assertEqual(__cache.shape, (2, __batch_dim, __seq_dim + 1, __num_heads, __head_dim))

    # def test_batch_decode_equivalence_to_sequential_decode(self):
    #     __batch_dim, __seq_dim, __embed_dim, __num_heads, __head_dim = 2, 5, 8, 2, 4
    #     # sampling mode
    #     __input_cache = mlable.utils.create_cache(batch_dim=__batch_dim, cache_dim=__seq_dim, num_heads=__num_heads, head_dim=__head_dim)
    #     # random mask
    #     __input_mask = tf.ones((__seq_dim, __seq_dim))
    #     # update index
    #     __step = 0
    #     # basic attention
    #     __layer = mlable.layers.transformer.CachedMultiHeadAttention(num_heads=__num_heads, key_dim=__head_dim)
    #     # input data
    #     __input_x = tf.random.uniform(shape=(__batch_dim, __seq_dim, __embed_dim), dtype=tf.dtypes.float32)
    #     # batch call
    #     __output_values, __output_scores, __output_cache = __layer(query=__input_x, value=__input_x, cache=__input_cache, attention_mask=__input_mask, step=0, return_attention_scores=True, use_causal_mask=True, training=True)
    #     # loop decode
    #     __loop_values = tf.zeros(shape=(__batch_dim, __seq_dim, __embed_dim), dtype=tf.dtypes.float32)
    #     __iteration_cache = mlable.utils.create_cache(batch_dim=__batch_dim, cache_dim=__seq_dim, num_heads=__num_heads, head_dim=__head_dim)
    #     for __i in range(__seq_dim):
    #         __iteration_x = tf.slice(__input_x, (0, __i, 0), (__batch_dim, 1, __embed_dim))
    #         __iteration_mask = __input_mask # tf.slice(__input_mask, (__i, 0), (1, __seq_dim))
    #         __iteration_values, __iteration_scores, __iteration_cache = __layer(query=__iteration_x, value=__iteration_x, cache=__iteration_cache, attention_mask=__iteration_mask, step=__i, return_attention_scores=True, use_causal_mask=True)
    #         __loop_values = tf.keras.ops.slice_update(__loop_values, (0, __i, 0), __iteration_values)
    #     # checks
    #     self.assertAllClose(__output_values, __loop_values)

    def test_training_process(self):
        __batch_dim, __seq_dim, __embed_dim, __num_heads, __head_dim = 2, 5, 8, 2, 4
        # training mode
        __cache = None
        __mask = None
        __step = None
        # zero the output on the last instruction of `_compute_attention`
        __layer = mlable.layers.transformer.CachedMultiHeadAttention(num_heads=__num_heads, key_dim=__head_dim, dropout=0.99999)
        # input data
        __inputs = tf.random.uniform(shape=(__batch_dim, __seq_dim, __embed_dim), dtype=tf.dtypes.float32)
        # call
        __output_values, __output_scores, __output_cache = __layer(query=__inputs, value=__inputs, cache=__cache, attention_mask=__mask, step=__step, return_attention_scores=True, training=True)
        # manual processing
        __manual_x = __layer._value_dense(__inputs)
        __manual_scores = tf.zeros((__batch_dim, __num_heads, __seq_dim, __seq_dim))
        __manual_values = tf.einsum(__layer._combine_equation, __manual_scores, __manual_x)
        __manual_values = __layer._output_dense(__manual_values)
        # check
        self.assertAllClose(__output_values, __manual_values, atol=1e-5)
