import tensorflow as tf

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
                    'axis': 1,},
                'shape': (4, 16, 16, 16, 3),},
            {
                'args': {
                    'data': tf.cast(tf.random.uniform((1, 2 ** (3 * 5), 1), minval=0, maxval=256, dtype=tf.int32), dtype=tf.uint8),
                    'order': 3,
                    'rank': 5,
                    'axis': 1,},
                'shape': (1, 8, 8, 8, 8, 8, 1),},]

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
            __args = {__k: __v for __k, __v in __case['args'].items() if __k in ['order', 'rank']}
            __axes = range(__case['args']['axis'], __case['args']['axis'] + __case['args']['rank'])
            __folded = mlable.shaping.hilbert.fold(**__case['args'])
            __back = mlable.shaping.hilbert.unfold(__folded, **__args, axes=__axes)
            self.assertAllEqual(__case['args']['data'], __back)

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
