import math
import random

import tensorflow as tf

import mlable.utils

# SHAPES ######################################################################

class ReshapingTests(tf.test.TestCase):
    def setUp(self):
        super(ReshapingTests, self).setUp()
        self._shapes = [
            range(4, 9, 1),
            (4, 4, 4, 4),
            (None, 4, 16),
            [1, 8],
            tf.ones((2, 16)).shape,]

    def test_filter(self):
        for __s in self._shapes:
            self.assertEqual(len(__s), len(mlable.utils.filter_shape(shape=__s, axes=[])))
            self.assertEqual(len(__s) * [1], mlable.utils.filter_shape(shape=__s, axes=[]))
            self.assertEqual(list(__s)[1], mlable.utils.filter_shape(shape=__s, axes=[1])[1])
            self.assertEqual((len(__s) - 1) * [1], mlable.utils.filter_shape(shape=__s, axes=[0])[1:])

    def test_normalize(self):
        for __s in self._shapes:
            assert all([isinstance(__d, int) for __d in mlable.utils.normalize_shape(shape=__s)])
            self.assertEqual(len(__s), len(mlable.utils.normalize_shape(shape=__s)))

    def test_divide(self):
        for __s in self._shapes:
            __d_same_rank = mlable.utils.divide_shape(shape=__s, input_axis=-1, output_axis=0, factor=4, insert=False)
            __d_add_axis = mlable.utils.divide_shape(shape=__s, input_axis=-1, output_axis=0, factor=4, insert=True)
            # same rank
            self.assertEqual(len(__s), len(__d_same_rank))
            # add an axis
            self.assertEqual(len(__s) + 1, len(__d_add_axis))
            # keeps cardinality
            if all(isinstance(__d, int) for __d in list(__s)):
                self.assertEqual(math.prod(list(__s)), math.prod(__d_same_rank))
                self.assertEqual(math.prod(list(__s)), math.prod(__d_add_axis))

    def test_merge(self):
        for __s in self._shapes:
            __m = mlable.utils.merge_shape(shape=__s, left_axis=-2, right_axis=-1, left=True)
            # one less axis
            self.assertEqual(len(__s) - 1, len(__m))
            # keeps cardinality
            if all(isinstance(__d, int) for __d in list(__s)):
                self.assertEqual(math.prod(list(__s)), math.prod(__m))

    def test_reciprocity(self):
        for __s in self._shapes:
            __d = mlable.utils.divide_shape(shape=__s, input_axis=-2, output_axis=-1, factor=4, insert=True)
            __m = mlable.utils.merge_shape(shape=__s, left_axis=-2, right_axis=-1, left=True)
            __dm = mlable.utils.merge_shape(shape=__d, left_axis=-2, right_axis=-1, left=True)
            __md = mlable.utils.divide_shape(shape=__m, input_axis=-2, output_axis=-1, factor=list(__s)[-1], insert=True)
            self.assertEqual(len(__s), len(__dm))
            self.assertEqual(len(__s), len(__md))
            self.assertEqual(mlable.utils.normalize_shape(__s), __dm)
            self.assertEqual(mlable.utils.normalize_shape(__s), __md)

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
