import numpy as np
import tensorflow as tf

import mlable.layers.transformer
import mlable.utils

# CACHE #######################################################################

class CacheToolingTest(tf.test.TestCase):

    def test_shapes(self):
        __batch_dim, __seq_dim, __num_heads, __head_dim= 3, 4, 2, 2
        # execution step
        __step = 1
        # init
        __cache = mlable.utils.create_cache(__batch_dim, __seq_dim, __num_heads, __head_dim)
        # input
        __key_1 = tf.ones((__batch_dim, 1, __num_heads, __head_dim), dtype=tf.dtypes.float32)
        __key_s = tf.ones((__batch_dim, __seq_dim, __num_heads, __head_dim), dtype=tf.dtypes.float32)
        # update
        __output_1 = mlable.utils.update_cache(tensor=__key_1, cache=__cache[0], axis=1, step=None)
        __output_s = mlable.utils.update_cache(tensor=__key_s, cache=__cache[0], axis=1, step=__step)
        # check shapes
        self.assertEqual(__output_1.shape, (__batch_dim, __seq_dim + 1, __num_heads, __head_dim))
        self.assertEqual(__output_s.shape, (__batch_dim, __seq_dim, __num_heads, __head_dim))
        # check values
        np.testing.assert_array_almost_equal(__output_1[:, __seq_dim, :, :], tf.ones((__batch_dim, __num_heads, __head_dim))) # appended
        np.testing.assert_array_almost_equal(__output_s[:, __step, :, :], tf.ones((__batch_dim, __num_heads, __head_dim))) # updated at index `__step`
