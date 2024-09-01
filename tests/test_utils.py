import math
import random

import tensorflow as tf

import mlable.utils

# FN COMPOSITION ##############################################################

class ComposeTest(tf.test.TestCase):
    def setUp(self):
        super(ComposeTest, self).setUp()
        self._random = [random.uniform(-8., 8.) for _ in range(32)]

    def test_identity(self):
        __f = lambda __x: __x
        __g = lambda __x: -__x
        __h = lambda __x: (__x, -__x)
        __i = lambda __t: tuple(reversed(__t))
        __f4 = mlable.utils.compose([__f, __f, __f, __f])
        __g2 = mlable.utils.compose([__g, __g])
        __hi2 = mlable.utils.compose([__h, __i, __i])
        self.assertEqual(self._random, [__f4(__e) for __e in self._random])
        self.assertEqual(self._random, [__g2(__e) for __e in self._random])
        self.assertEqual([__h(__e) for __e in self._random], [__hi2(__e) for __e in self._random])
        self.assertEqual(self._random, __f4(self._random))

# FN MAP ######################################################################

class DistributeTest(tf.test.TestCase):
    def setUp(self):
        super(DistributeTest, self).setUp()
        self._random = [random.uniform(-8., 8.) for _ in range(32)]

    def test_values(self):
        __f = lambda __x: __x ** 2
        __g = lambda __x: -__x
        __fn = mlable.utils.distribute(__f)
        __gn = mlable.utils.distribute(__g)
        self.assertEqual([(__e ** 2, __e ** 2, __e ** 2) for __e in self._random], [__fn(__e, __e, __e) for __e in self._random])
        self.assertEqual([(-__e,) for __e in self._random], [__gn(__e) for __e in self._random])

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
        self.assertAllClose(__output_1[:, __seq_dim, :, :], tf.ones((__batch_dim, __num_heads, __head_dim))) # appended
        self.assertAllClose(__output_s[:, __step, :, :], tf.ones((__batch_dim, __num_heads, __head_dim))) # updated at index `__step`
