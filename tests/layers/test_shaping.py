import tensorflow as tf

import mlable.layers.shaping

# BOTH ########################################################################

class ReshapingTest(tf.test.TestCase):
    def setUp(self):
        super(ReshapingTest, self).setUp()
        self._tensors = [
            tf.ones(shape=range(4, 9, 1)),
            tf.zeros(shape=(4, 4, 4, 4)),
            tf.keras.Input(shape=(None, 4, 16)),
            tf.random.uniform(shape=[2, 16], minval=0., maxval=32),]

    def test_reciprocity(self):
        for __t in self._tensors:
            __d = mlable.layers.shaping.Divide(axis=-1, factor=list(__t.shape)[-1], insert=True, right=True)
            __m = mlable.layers.shaping.Merge(axis=-1, right=False)
            if None not in list(__t.shape):
                self.assertAllEqual(__t, __d(__m(__t)))
                self.assertAllEqual(__t, __m(__d(__t)))

    def test_values_are_unchanged(self):
        for __t in self._tensors:
            __d = mlable.layers.shaping.Divide(axis=-1, factor=4, insert=True, right=True)
            __m = mlable.layers.shaping.Merge(axis=-1, right=False)
            if None not in list(__t.shape):
                self.assertAllEqual(tf.reshape(__t, shape=(-1,)), tf.reshape(__d(__t), shape=(-1,)))
                self.assertAllEqual(tf.reshape(__t, shape=(-1,)), tf.reshape(__m(__t), shape=(-1,)))

    def test_divide_shape(self):
        for __t in self._tensors:
            __d = mlable.layers.shaping.Divide(axis=-1, factor=4, insert=True, right=True)
            __si = list(__t.shape)
            __sd = list(__d(__t).shape)
            self.assertEqual(__sd[-1], 4)
            self.assertEqual(__sd[-2] * 4, __si[-1])
            self.assertEqual(__sd[:-2], __si[:-1])

    def test_merge_shape(self):
        for __t in self._tensors:
            __m = mlable.layers.shaping.Merge(axis=-1, right=False)
            __si = list(__t.shape)
            __sm = list(__m(__t).shape)
            self.assertEqual(__sm[-1], __si[-1] * __si[-2])
            self.assertEqual(__sm[:-1], __si[:-2])
