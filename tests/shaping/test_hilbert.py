import tensorflow as tf

import densecurves.hilbert
import mlable.shapes
import mlable.shaping.hilbert

# FOLD #########################################################################

class FoldTest(tf.test.TestCase):
    def setUp(self):
        super(FoldTest, self).setUp()
        self._cases = [
            {
                'args': {
                    'data': tf.random.uniform((4, 4096, 3), minval=-1.0, maxval=1.0, dtype=tf.bfloat16),
                    'order': 4,
                    'rank': 3,
                    'axis': -2,},
                'shape': (4, 16, 16, 16, 3),},
            {
                'args': {
                    'data': tf.cast(tf.random.uniform((1, 2 ** (3 * 3), 1), minval=0, maxval=256, dtype=tf.int32), dtype=tf.uint8),
                    'order': 3,
                    'rank': 3,
                    'axis': 1,},
                'shape': (1, 8, 8, 8, 1),},]

    def test_shape_and_dtype(self):
        for __case in self._cases:
            __folded = mlable.shaping.hilbert.fold(**__case['args'])
            assert tuple(__folded.shape) == __case['shape']
            assert __folded.dtype == __case['args']['data'].dtype

    def test_invariance_with_rank_1(self):
        __data = tf.random.uniform((4, 128, 3), minval=-1.0, maxval=1.0, dtype=tf.float32)
        __folded = mlable.shaping.hilbert.fold(__data, order=7, rank=1, axis=1)
        self.assertAllEqual(__data, __folded)

    def test_reciprocity(self):
        for __case in self._cases:
            __shape = list(__case['args']['data'].shape)
            __order = __case['args']['order']
            __rank = __case['args']['rank']
            __axis = __case['args']['axis'] % len(__shape)
            __axes = range(__axis, __axis + __rank)
            __folded = mlable.shaping.hilbert.fold(**__case['args'])
            __back = mlable.shaping.hilbert.unfold(__folded, order=__order, rank=__rank, axes=__axes)
            self.assertAllEqual(__case['args']['data'], __back)

    def test_each_value_along_the_curve(self):
        for __case in self._cases:
            __shape_flat = list(__case['args']['data'].shape)
            __order = __case['args']['order']
            __rank = __case['args']['rank']
            __axis = __case['args']['axis'] % len(__shape_flat)
            __axes = list(range(__axis, __axis + __rank))
            __folded = mlable.shaping.hilbert.fold(**__case['args'])
            __curve = [densecurves.hilbert.point(__i, order=__order, rank=__rank) for __i in range(2 ** (__order * __rank))]
            for __i, __p in enumerate(__curve):
                __shape_folded = list(__folded.shape)
                __size_flat = [__d if (__j != __axis) else 1 for __j, __d in enumerate(__shape_flat)]
                __size_folded = [__d if (__j not in __axes) else 1 for __j, __d in enumerate(__shape_folded)]
                __begin_flat = [__i if (__j == __axis) else 0 for __j in range(len(__shape_flat))]
                __begin_folded = [__p[__j - __axis] if (__j in __axes) else 0 for __j in range(len(__shape_folded))]
                self.assertAllEqual(
                    tf.squeeze(tf.slice(__case['args']['data'], __begin_flat, __size_flat)),
                    tf.squeeze(tf.slice(__folded, __begin_folded, __size_folded)))

# UNFOLD #######################################################################

class UnfoldTest(tf.test.TestCase):
    def setUp(self):
        super(UnfoldTest, self).setUp()
        self._cases = [
            {
                'args': {
                    'data': tf.random.uniform((4, 16, 16, 16, 3), minval=-1.0, maxval=1.0, dtype=tf.bfloat16),
                    'order': 4,
                    'rank': 3,
                    'axes': [1, 2, -2],},
                'shape': (4, 4096, 3),},
            {
                'args': {
                    'data': tf.cast(tf.random.uniform((1, 8, 8, 8, 8, 1), minval=0, maxval=256, dtype=tf.int32), dtype=tf.uint8),
                    'order': 3,
                    'rank': 4,
                    'axes': [-2, 1, 3, 2],},
                'shape': (1, 4096, 1),},]

    def test_shape_and_dtype(self):
        for __case in self._cases:
            __unfolded = mlable.shaping.hilbert.unfold(**__case['args'])
            assert tuple(__unfolded.shape) == __case['shape']
            assert __unfolded.dtype == __case['args']['data'].dtype

    def test_invariance_with_rank_1(self):
        __data = tf.random.uniform((4, 5, 128, 6, 3), minval=-1.0, maxval=1.0, dtype=tf.float32)
        __unfolded = mlable.shaping.hilbert.unfold(__data, order=7, rank=1, axes=[2])
        self.assertAllEqual(__data, __unfolded)

    def test_reciprocity(self):
        for __case in self._cases:
            __axes = [__a % len(tuple(__case['args']['data'].shape)) for __a in __case['args']['axes']]
            __unfolded = mlable.shaping.hilbert.unfold(**__case['args'])
            __back = mlable.shaping.hilbert.fold(__unfolded, order=__case['args']['order'], rank=__case['args']['rank'], axis=min(__axes))
            self.assertAllEqual(__case['args']['data'], __back)
