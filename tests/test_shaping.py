import math

import tensorflow as tf

import mlable.shaping

# SHAPES ######################################################################

class ReshapingTests(tf.test.TestCase):
    def setUp(self):
        super(ReshapingTests, self).setUp()
        self._shapes = [
            range(4, 9, 1),
            (4, 4, 4, 4),
            (None, 4, 16),
            [1, 8],
            tf.ones((2, 16)).shape,
            tf.keras.Input((32, 32, 8)).shape]

    def test_filter(self):
        for __s in self._shapes:
            self.assertEqual(len(__s), len(mlable.shaping.filter_shape(shape=__s, axes=[])))
            self.assertEqual(len(__s) * [1], mlable.shaping.filter_shape(shape=__s, axes=[]))
            self.assertEqual(list(__s)[1], mlable.shaping.filter_shape(shape=__s, axes=[1])[1])
            self.assertEqual((len(__s) - 1) * [1], mlable.shaping.filter_shape(shape=__s, axes=[0])[1:])

    def test_normalize(self):
        for __s in self._shapes:
            assert all([isinstance(__d, int) for __d in mlable.shaping.normalize_shape(shape=__s)])
            self.assertEqual(len(__s), len(mlable.shaping.normalize_shape(shape=__s)))

    def test_symbolic(self):
        for __s in self._shapes:
            __integer_shape = mlable.shaping.normalize_shape(shape=__s)
            __symbolic_shape = mlable.shaping.symbolic_shape(shape=__integer_shape)
            assert tuple(__s) == tuple(__symbolic_shape)
            assert all([isinstance(__d, (int, None.__class__)) for __d in __symbolic_shape])

    def test_divide(self):
        for __s in self._shapes:
            __d_same_rank = mlable.shaping.divide_shape(shape=__s, input_axis=-1, output_axis=0, factor=4, insert=False)
            __d_add_axis = mlable.shaping.divide_shape(shape=__s, input_axis=-1, output_axis=0, factor=4, insert=True)
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
            __m = mlable.shaping.merge_shape(shape=__s, left_axis=-2, right_axis=-1, left=True)
            # one less axis
            self.assertEqual(len(__s) - 1, len(__m))
            # keeps cardinality
            if all(isinstance(__d, int) for __d in list(__s)):
                self.assertEqual(math.prod(list(__s)), math.prod(__m))

    def test_reciprocity(self):
        for __s in self._shapes:
            __d = mlable.shaping.divide_shape(shape=__s, input_axis=-2, output_axis=-1, factor=4, insert=True)
            __m = mlable.shaping.merge_shape(shape=__s, left_axis=-2, right_axis=-1, left=True)
            __dm = mlable.shaping.merge_shape(shape=__d, left_axis=-2, right_axis=-1, left=True)
            __md = mlable.shaping.divide_shape(shape=__m, input_axis=-2, output_axis=-1, factor=list(__s)[-1], insert=True)
            self.assertEqual(len(__s), len(__dm))
            self.assertEqual(len(__s), len(__md))
            self.assertEqual(mlable.shaping.normalize_shape(__s), __dm)
            self.assertEqual(mlable.shaping.normalize_shape(__s), __md)
