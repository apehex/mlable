import tensorflow as tf

import mlable.masking

# REDUCE ######################################################################

class MaskingTest(tf.test.TestCase):

    def setUp(self):
        super(MaskingTest, self).setUp()
        # all groups have one non-zero value
        self._ascii = tf.convert_to_tensor([
            '1111111111111111',
            '2222222222222222'])
        # each sample has null groups
        self._padded = tf.convert_to_tensor([
            '11111111111111\x00\x00',
            '\x00\x00\x00\x002222222222\x002'])
        # encode
        self._ascii = tf.strings.unicode_transcode(input=self._ascii, input_encoding='UTF-8', output_encoding='UTF-32-BE')
        self._ascii = tf.io.decode_raw(self._ascii, out_type=tf.int8, fixed_length=64)
        self._padded = tf.strings.unicode_transcode(input=self._padded, input_encoding='UTF-8', output_encoding='UTF-32-BE')
        self._padded = tf.io.decode_raw(self._padded, out_type=tf.int8, fixed_length=64)
        # mask
        self._ascii = tf.not_equal(self._ascii, 0)
        self._padded = tf.not_equal(self._padded, 0)

    def test_reduce_shapes(self):
        # reduced along a single axis
        __m_0_any = mlable.masking._reduce_any(mask=self._ascii, axis=0, keepdims=False)
        __m_0_all = mlable.masking._reduce_all(mask=self._ascii, axis=0, keepdims=False)
        __m_1_any = mlable.masking._reduce_any(mask=self._ascii, axis=-1, keepdims=False)
        __m_1_all = mlable.masking._reduce_all(mask=self._ascii, axis=-1, keepdims=False)
        self.assertEqual(list(__m_0_any.shape), [64,])
        self.assertEqual(list(__m_0_all.shape), [64,])
        self.assertEqual(list(__m_1_any.shape), [2,])
        self.assertEqual(list(__m_1_all.shape), [2,])
        # reduced on all axes
        __m_any = mlable.masking._reduce_any(mask=self._ascii, axis=None, keepdims=False)
        __m_all = mlable.masking._reduce_all(mask=self._ascii, axis=None, keepdims=False)
        self.assertEqual(list(__m_any.shape), [])
        self.assertEqual(list(__m_all.shape), [])
        # same shape as input
        __m_0_any_keep = mlable.masking._reduce_any(mask=self._ascii, axis=0, keepdims=True)
        __m_0_all_keep = mlable.masking._reduce_all(mask=self._ascii, axis=0, keepdims=True)
        __m_1_any_keep = mlable.masking._reduce_any(mask=self._ascii, axis=-1, keepdims=True)
        __m_1_all_keep = mlable.masking._reduce_all(mask=self._ascii, axis=-1, keepdims=True)
        __m_any_keep = mlable.masking._reduce_any(mask=self._ascii, axis=None, keepdims=True)
        __m_all_keep = mlable.masking._reduce_all(mask=self._ascii, axis=None, keepdims=True)
        self.assertEqual(list(__m_0_any_keep.shape), list(self._ascii.shape))
        self.assertEqual(list(__m_0_all_keep.shape), list(self._ascii.shape))
        self.assertEqual(list(__m_1_any_keep.shape), list(self._ascii.shape))
        self.assertEqual(list(__m_1_all_keep.shape), list(self._ascii.shape))
        self.assertEqual(list(__m_any_keep.shape), list(self._ascii.shape))
        self.assertEqual(list(__m_all_keep.shape), list(self._ascii.shape))

    def test_reduce_on_ascii_values(self):
        # reduced along a single axis
        __m_0_any = mlable.masking._reduce_any(mask=self._ascii, axis=0, keepdims=False)
        __m_0_all = mlable.masking._reduce_all(mask=self._ascii, axis=0, keepdims=False)
        __m_1_any = mlable.masking._reduce_any(mask=self._ascii, axis=-1, keepdims=False)
        __m_1_all = mlable.masking._reduce_all(mask=self._ascii, axis=-1, keepdims=False)
        self.assertAllEqual(__m_0_any, __m_0_all)
        self.assertAllEqual(__m_1_any, tf.convert_to_tensor([True, True]))
        self.assertAllEqual(__m_1_all, tf.convert_to_tensor([False, False]))
        # reduced on all axes
        __m_any = mlable.masking._reduce_any(mask=self._ascii, axis=None, keepdims=False)
        __m_all = mlable.masking._reduce_all(mask=self._ascii, axis=None, keepdims=False)
        self.assertEqual(__m_any.numpy(), True)
        self.assertEqual(__m_all.numpy(), False)
        # same shape as input
        __m_0_any_keep = mlable.masking._reduce_any(mask=self._ascii, axis=0, keepdims=True)
        __m_0_all_keep = mlable.masking._reduce_all(mask=self._ascii, axis=0, keepdims=True)
        __m_1_any_keep = mlable.masking._reduce_any(mask=self._ascii, axis=-1, keepdims=True)
        __m_1_all_keep = mlable.masking._reduce_all(mask=self._ascii, axis=-1, keepdims=True)
        __m_any_keep = mlable.masking._reduce_any(mask=self._ascii, axis=None, keepdims=True)
        __m_all_keep = mlable.masking._reduce_all(mask=self._ascii, axis=None, keepdims=True)
        self.assertAllEqual(__m_0_any_keep, __m_0_all_keep)
        self.assertAllEqual(__m_1_any_keep, tf.ones(shape=(2, 64), dtype=tf.dtypes.bool))
        self.assertAllEqual(__m_1_all_keep, tf.zeros(shape=(2, 64), dtype=tf.dtypes.bool))
        self.assertAllEqual(__m_any_keep, tf.ones(shape=(2, 64), dtype=tf.dtypes.bool))
        self.assertAllEqual(__m_all_keep, tf.zeros(shape=(2, 64), dtype=tf.dtypes.bool))

    def test_group_shapes(self):
        # reduced along a single axis
        __m_0_any = mlable.masking._reduce_group_by_group_any(mask=self._ascii, group=2, axis=0, keepdims=False)
        __m_0_all = mlable.masking._reduce_group_by_group_all(mask=self._ascii, group=2, axis=0, keepdims=False)
        __m_1_any = mlable.masking._reduce_group_by_group_any(mask=self._ascii, group=4, axis=-1, keepdims=False)
        __m_1_all = mlable.masking._reduce_group_by_group_all(mask=self._ascii, group=4, axis=-1, keepdims=False)
        self.assertEqual(list(__m_0_any.shape), [1, 64])
        self.assertEqual(list(__m_0_all.shape), [1, 64])
        self.assertEqual(list(__m_1_any.shape), [2, 16])
        self.assertEqual(list(__m_1_all.shape), [2, 16])
        # same shape as input
        __m_0_any_keep = mlable.masking._reduce_group_by_group_any(mask=self._ascii, group=2, axis=0, keepdims=True)
        __m_0_all_keep = mlable.masking._reduce_group_by_group_all(mask=self._ascii, group=2, axis=0, keepdims=True)
        __m_1_any_keep = mlable.masking._reduce_group_by_group_any(mask=self._ascii, group=4, axis=-1, keepdims=True)
        __m_1_all_keep = mlable.masking._reduce_group_by_group_all(mask=self._ascii, group=4, axis=-1, keepdims=True)
        self.assertEqual(list(__m_0_any_keep.shape), list(self._ascii.shape))
        self.assertEqual(list(__m_0_all_keep.shape), list(self._ascii.shape))
        self.assertEqual(list(__m_1_any_keep.shape), list(self._ascii.shape))
        self.assertEqual(list(__m_1_all_keep.shape), list(self._ascii.shape))

    def test_group_on_ascii_values(self):
        # reduce 4 by 4
        __m_1_any = mlable.masking._reduce_group_by_group_any(mask=self._ascii, group=4, axis=-1, keepdims=False)
        __m_1_all = mlable.masking._reduce_group_by_group_all(mask=self._ascii, group=4, axis=-1, keepdims=False)
        self.assertAllEqual(__m_1_any, tf.ones(shape=(2, 16), dtype=tf.dtypes.bool))
        self.assertAllEqual(__m_1_all, tf.zeros(shape=(2, 16), dtype=tf.dtypes.bool))
        # same shape as input
        __m_1_any_keep = mlable.masking._reduce_group_by_group_any(mask=self._ascii, group=4, axis=-1, keepdims=True)
        __m_1_all_keep = mlable.masking._reduce_group_by_group_all(mask=self._ascii, group=4, axis=-1, keepdims=True)
        self.assertAllEqual(__m_1_any_keep, tf.ones(shape=(2, 64), dtype=tf.dtypes.bool))
        self.assertAllEqual(__m_1_all_keep, tf.zeros(shape=(2, 64), dtype=tf.dtypes.bool))

    def test_group_on_padded_values(self):
        # reduce 4 by 4
        __m_1_any = mlable.masking.reduce_any(mask=self._padded, group=4, axis=-1, keepdims=False)
        __m_1_all = mlable.masking.reduce_all(mask=self._padded, group=4, axis=-1, keepdims=False)
        self.assertAllEqual(__m_1_any, tf.convert_to_tensor([14 * [True] + [False, False], 4 * [False] + 10 * [True] + [False, True]], dtype=tf.dtypes.bool))
        self.assertAllEqual(__m_1_all, tf.zeros(shape=(2, 16), dtype=tf.dtypes.bool))
        # same shape as input
        __m_1_any_keep = mlable.masking.reduce_any(mask=self._padded, group=4, axis=-1, keepdims=True)
        __m_1_all_keep = mlable.masking.reduce_all(mask=self._padded, group=4, axis=-1, keepdims=True)
        self.assertAllEqual(__m_1_any_keep, tf.convert_to_tensor([14 * 4 * [True] + 2 * 4 * [False], 4 * 4 * [False] + 10 * 4 * [True] + 4 * [False] + 4 * [True]], dtype=tf.dtypes.bool))
        self.assertAllEqual(__m_1_all_keep, tf.zeros(shape=(2, 64), dtype=tf.dtypes.bool))
